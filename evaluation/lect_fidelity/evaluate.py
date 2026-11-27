#!/usr/bin/env python
"""Calibrate on the calibration split and measure on the evaluation split.

Usage, from this directory, in a fresh process per provider (``tugaphone`` registers its
lexicon with ``orthography2ipa`` for the whole process)::

    python evaluate.py orthography2ipa allosaurus,wav2vec2_espeak data/clips-clul.jsonl clul
    python evaluate.py tugaphone allosaurus,wav2vec2_espeak data/clips-clul.jsonl clul

The clips file names each clip's role, calibration or evaluation, and its corpus.
Writes ``data/eval-<tag>-<provider>.json``: per backend and decoding mode, the
calibration fitted on the calibration clips, and per evaluation corpus the AUC of the
clip log-likelihood ratio with a stratified bootstrap 95% interval and per duration
bin, clip accuracy at the calibrated decision point with Wilson intervals, accuracy per
recording or speaker with all its clips pooled, accuracy of pools of at least 30
seconds, per-class readings and median shares.
"""
import importlib.metadata as md
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from speechonnxmetrics.lect_fidelity import expect, fit, pool, score_phones
from speechonnxmetrics.lect_fidelity.calibration import table_entry

provider = sys.argv[1]
LECTS = ("pt-PT", "pt-BR")
BACKENDS = tuple(sys.argv[2].split(","))
clips = {c["id"]: c for c in (json.loads(l) for l in open(sys.argv[3]))}
TAG = sys.argv[4]
phones = {b: {r["id"]: r for r in (json.loads(l) for l in open(f"data/phones-{b}.jsonl"))} for b in BACKENDS}
expectations = {}
for c in clips.values():
    if c["text"] not in expectations:
        expectations[c["text"]] = expect(c["text"], LECTS, provider)
site_stats = Counter()
for e in expectations.values():
    site_stats.update(s.cls for s in e.sites)
    site_stats.update({f"excluded:{k}": v for k, v in e.excluded.items()})
print(provider, "texts", len(expectations), dict(site_stats), flush=True)


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def bootstrap_auc(scores, labels, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    scores, labels = np.array(scores), np.array(labels)
    pos, neg = np.where(labels)[0], np.where(~labels)[0]
    vals = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(auc(scores[i], labels[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def wilson(k, n):
    if not n:
        return [float("nan")] * 2
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


DURATION_BINS = ((0, 3), (3, 6), (6, 10), (10, 1e9))


def measure(cal, ev_ids, results):
    out = {}
    if len({clips[i]["lect"] for i in ev_ids}) < 2:
        out["single_lect"] = True
    labels = np.array([clips[i]["lect"] == LECTS[0] for i in ev_ids])
    llr = [cal.log_likelihood_ratio(results[i].readings) for i in ev_ids]
    if not out.get("single_lect"):
        out["auc"] = auc(llr, labels)
        out["auc_ci"] = bootstrap_auc(llr, labels)
        out["auc_per_site"] = auc([v / results[i].n_sites for v, i in zip(llr, ev_ids)], labels)
        bins = {}
        for lo, hi in DURATION_BINS:
            k = [n for n, i in enumerate(ev_ids) if lo <= results[i].duration < hi]
            sub = labels[k]
            bins[f"{lo}-{hi if hi < 1e9 else 'inf'} s"] = {
                "pt-PT": int(sub.sum()), "pt-BR": int((~sub).sum()),
                "auc": auc([llr[n] for n in k], sub) if sub.any() and (~sub).any() else None}
        out["auc_by_duration"] = bins
    acc = {}
    for lect in LECTS:
        sel = [k for k, i in enumerate(ev_ids) if clips[i]["lect"] == lect]
        right = sum(cal.decide(llr[k]) == lect for k in sel)
        acc[lect] = {"right": right, "n": len(sel), "ci": wilson(right, len(sel))}
    out["clip_accuracy"] = acc
    groups = {}
    for i in ev_ids:
        groups.setdefault(clips[i].get("group") or i, []).append(i)
    per_group = {lect: {"right": 0, "n": 0} for lect in LECTS}
    for g, ids in groups.items():
        if len(ids) < 2 and not clips[ids[0]].get("group"):
            continue
        lect = clips[ids[0]]["lect"]
        p = pool([results[i] for i in ids], cal)
        per_group[lect]["n"] += 1
        per_group[lect]["right"] += p.decision == lect
        per_group[lect].setdefault("seconds", []).append(round(p.duration, 1))
    out["group_accuracy"] = per_group
    pooled = {}
    rng = random.Random(7)
    for lect in LECTS:
        sel = [i for i in ev_ids if clips[i]["lect"] == lect]
        rng.shuffle(sel)
        groups30, cur, dur = [], [], 0.0
        for i in sel:
            cur.append(results[i]); dur += results[i].duration
            if dur >= 30.0:
                groups30.append(cur); cur, dur = [], 0.0
        pr = [pool(g, cal) for g in groups30]
        right = sum(p.decision == lect for p in pr)
        pooled[lect] = {"right": right, "n": len(pr), "ci": wilson(right, len(pr))}
    out["pooled_30s_accuracy"] = pooled
    per_class = defaultdict(lambda: {l: Counter() for l in LECTS})
    for i in ev_ids:
        for r in results[i].readings:
            per_class[r.site.cls][clips[i]["lect"]][r.label] += 1
    out["per_class_readings"] = {k: {l: dict(v[l]) for l in LECTS} for k, v in per_class.items()}
    out["median_shares"] = {lect: {k: float(np.median([results[i].shares[k] for i in ev_ids if clips[i]["lect"] == lect] or [0]))
                                   for k in (*LECTS, "neither")} for lect in LECTS}
    return out


def run(backend, mode):
    out = {"backend": backend, "mode": mode}
    results = {}
    reasons = Counter()
    for cid, c in clips.items():
        e = expectations[c["text"]]
        row = phones[backend].get(cid)
        if row is None:
            reasons["not recognised"] += 1
            continue
        if not e.sites:
            reasons[f"{c['role']} no site"] += 1
            continue
        r = score_phones(row["phones"][mode], e, None, row["duration"])
        if r is None:
            reasons[f"{c['role']} too little speech"] += 1
            continue
        results[cid] = r
    out["skipped"] = dict(reasons)
    cal_ids = [i for i in results if clips[i]["role"] == "calibration"]
    ev_ids = [i for i in results if clips[i]["role"] == "evaluation"]
    sizes = {}
    for role, ids in (("calibration", cal_ids), ("evaluation", ev_ids)):
        for lect in LECTS:
            sel = [i for i in ids if clips[i]["lect"] == lect]
            sizes[f"{role} {lect}"] = {"clips": len(sel), "seconds": round(sum(results[i].duration for i in sel), 1),
                                       "sites": sum(results[i].n_sites for i in sel),
                                       "corpus": sorted({clips[i]["corpus"] for i in sel})}
    out["sizes"] = sizes
    provenance = {
        "calibration_split": {k: v for k, v in sizes.items() if k.startswith("calibration")},
        "inventory": "restricted to the union of the two lects' phones" if mode != "none" else "unrestricted",
        "corpora": {role: sorted({clips[i]["corpus"] for i in ids})
                    for role, ids in (("calibration", cal_ids), ("evaluation", ev_ids))},
        "packages": {p: md.version(p) for p in ("orthography2ipa", "tugaphone", "scriptconv", "onnxruntime")},
    }
    cal = fit([(clips[i]["lect"], results[i].readings) for i in cal_ids], LECTS, provider, backend, provenance, restrict_inventory=mode != "none")
    out["calibration"] = cal.to_dict()
    out["evaluation"] = {}
    for corpus in sorted({clips[i]["corpus"] for i in ev_ids}):
        out["evaluation"][corpus] = measure(cal, [i for i in ev_ids if clips[i]["corpus"] == corpus], results)
    corpora = {clips[i]["corpus"] for i in ev_ids}
    if len(corpora) > 1 and len({clips[i]["lect"] for i in ev_ids}) == 2 and all(
            len({clips[i]["lect"] for i in ev_ids if clips[i]["corpus"] == c}) == 1 for c in corpora):
        out["evaluation"]["cross-corpus: " + " + ".join(sorted(corpora))] = measure(cal, ev_ids, results)
    cov = {}
    for lect in LECTS:
        sel = [results[i] for i in cal_ids + ev_ids if clips[i]["lect"] == lect]
        ratio = [len(r.realised) / min(len(p) for p in r.expected.values()) for r in sel]
        cov[lect] = [float(np.percentile(ratio, q)) for q in (1, 5, 50, 95)]
    out["realised_to_expected_ratio_p1_p5_p50_p95"] = cov
    return out, cal


summary = []
for backend in phones:
    for mode in ("none", provider):
        res, cal = run(backend, mode)
        summary.append(res)
        for corpus, m in res["evaluation"].items():
            print(json.dumps({"backend": backend, "mode": mode, "corpus": corpus,
                              **{k: m.get(k) for k in ("auc", "auc_ci", "auc_per_site", "clip_accuracy", "group_accuracy", "pooled_30s_accuracy")}},
                             ensure_ascii=False), flush=True)
Path(f"data/eval-{TAG}-{provider}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
