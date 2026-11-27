#!/usr/bin/env python
"""Select and extract the real-speech clips the lect-fidelity calibration and evaluation use.

The headline measurement uses one corpus for both lects: the CLUL corpus "Spoken
Portuguese: Geographical and Social Varieties", split by recording with ``tugalect
prepare spgsv`` and ``tugalect split``, and compare-accents-pt, one paragraph read by
twenty speakers::

    python prepare.py clul TUGALECT_DATA   # writes data/clips-clul.jsonl
    python prepare.py wav2vec2             # writes data/clips-clul-w2v.jsonl

``TUGALECT_DATA`` holds tugalect's ``spgsv/train.jsonl``, ``spgsv/test.jsonl`` and
``compare-accents/manifest.jsonl``. The training recordings calibrate; the held-out
recordings and compare-accents-pt evaluate. ``clul-split.json`` names the 37 training
and 13 held-out recordings with their clip counts, and the ``clul`` part refuses a split
that differs from it. The wav2vec2 model runs several times slower than Allosaurus, so
it decodes the subset of those clips named in ``wav2vec2-clips.json``.

The secondary, cross-corpus measurement takes each lect from a different corpus. Run
from this directory, once per part; each writes ``data/clips-<part>.jsonl``::

    python prepare.py fleurs        # FLEURS pt_br: dev for calibration, test for evaluation
    python prepare.py eurospeech    # EuroSpeech Portugal validation shards 0, 3, 6: calibration
    python prepare.py massive 400   # Speech-MASSIVE pt-PT: evaluation only, never redistributed

then ``cat data/clips-*.jsonl > data/clips.jsonl``. Selections are seeded, so the same
clips come back from the same dataset revisions.
"""
import csv
import io
import json
import random
import tarfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
out = Path("data")
clips = []


def fleurs(split, role, n, seed):
    tsv = hf_hub_download("google/fleurs", f"data/pt_br/{split}.tsv", repo_type="dataset")
    tar = hf_hub_download("google/fleurs", f"data/pt_br/audio/{split}.tar.gz", repo_type="dataset")
    rows = [r for r in csv.reader(open(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)]
    rng = random.Random(seed)
    rng.shuffle(rows)
    seen_text = set()
    keep = []
    for r in rows:
        if r[2] in seen_text:
            continue
        seen_text.add(r[2])
        keep.append(r)
        if len(keep) == n:
            break
    want = {r[1]: r for r in keep}
    d = out / "fleurs" / split
    d.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        for m in t:
            name = Path(m.name).name
            if name in want:
                (d / name).write_bytes(t.extractfile(m).read())
    for name, r in want.items():
        clips.append({"id": f"fleurs-{split}-{name}", "role": role, "lect": "pt-BR", "corpus": f"FLEURS pt_br {split}",
                      "path": str(d / name), "text": r[2], "gender": r[6] if len(r) > 6 else ""})


def eurospeech(n_per_shard, seed):
    d = out / "eurospeech"
    d.mkdir(parents=True, exist_ok=True)
    for shard in (0, 3, 6):
        p = hf_hub_download("disco-eth/EuroSpeech", f"portugal/validation-0000{shard}-of-00008.parquet", repo_type="dataset")
        t = pq.read_table(p, columns=["key", "duration_seconds", "human_transcript", "cer", "audio", "video_id"])
        df = t.drop_columns(["audio"]).to_pandas()
        ok = df[(df.cer <= 0.08) & (df.duration_seconds <= 20)]
        print("eurospeech shard", shard, "rows", len(df), "eligible", len(ok), "sessions", sorted(ok.video_id.unique()))
        pick = ok.sample(n=min(n_per_shard, len(ok)), random_state=seed + shard)
        audio = t.column("audio")
        for i, row in pick.iterrows():
            x, sr = sf.read(io.BytesIO(audio[i].as_py()["bytes"]))
            f = d / f"{row.key}.wav"
            sf.write(f, x, sr)
            clips.append({"id": f"eurospeech-{row.key}", "role": "calibration", "lect": "pt-PT",
                          "corpus": "EuroSpeech Portugal validation", "path": str(f), "text": row.human_transcript,
                          "speaker_group": row.video_id})


def massive(n, seed):
    p = hf_hub_download("TigreGotico/speech_MASSIVE_pt-PT", "default/train/0001.parquet", repo_type="dataset",
                        revision="refs/convert/parquet")
    t = pq.read_table(p)
    df = t.drop_columns(["audio"]).to_pandas()
    print("massive parquet rows", len(df), list(df.columns), df.split.value_counts().to_dict() if "split" in df else "")
    pick = df.sample(n=min(n, len(df)), random_state=seed)
    d = out / "massive"
    d.mkdir(parents=True, exist_ok=True)
    audio = t.column("audio")
    for i, row in pick.iterrows():
        a = audio[i].as_py()
        f = d / Path(a.get("path") or f"{i}.wav").name
        f.write_bytes(a["bytes"])
        clips.append({"id": f"massive-{f.stem}", "role": "evaluation", "lect": "pt-PT",
                      "corpus": "Speech-MASSIVE pt-PT", "path": str(f), "text": row.text})


def clul(root):
    root = Path(root)
    split = json.loads((HERE / "clul-split.json").read_text())
    parts = (("calibration", "spgsv/train.jsonl", "CLUL spoken varieties training recordings"),
             ("evaluation", "spgsv/test.jsonl", "CLUL spoken varieties held-out recordings"),
             ("evaluation", "compare-accents/manifest.jsonl", "compare-accents-pt"))
    for role, name, corpus in parts:
        for line in open(root / name):
            r = json.loads(line)
            audio = Path(r["audio"])
            clips.append({"id": r["id"] if corpus == "compare-accents-pt" else f"clul-{r['id']}", "role": role,
                          "lect": r["lect"], "corpus": corpus,
                          "path": str(audio if audio.is_absolute() else root / name.split("/")[0] / audio),
                          "text": r["text"], "group": r["group"]})
    for part, corpus in (("training", "CLUL spoken varieties training recordings"),
                         ("held_out", "CLUL spoken varieties held-out recordings")):
        found = {}
        for c in clips:
            if c["corpus"] == corpus:
                groups = found.setdefault(c["lect"], {})
                groups[c["group"]] = groups.get(c["group"], 0) + 1
        if found != split[part]:
            raise SystemExit(f"the {part} recordings differ from clul-split.json")


def wav2vec2():
    chosen = json.loads((HERE / "wav2vec2-clips.json").read_text())
    with open(out / "clips-clul.jsonl") as fh:
        by_id = {c["id"]: c for c in map(json.loads, fh)}
    for role in ("evaluation", "calibration"):
        clips.extend(by_id[i] for i in chosen[role])


import sys
part = sys.argv[1]
if part == "clul":
    clul(sys.argv[2])
elif part == "wav2vec2":
    wav2vec2()
    part = "clul-w2v"
elif part == "fleurs":
    fleurs("dev", "calibration", 330, 1)
    fleurs("test", "evaluation", 400, 2)
elif part == "eurospeech":
    eurospeech(110, 3)
else:
    massive(int(sys.argv[2]), 4)
with open(out / f"clips-{part}.jsonl", "w") as fh:
    for c in clips:
        fh.write(json.dumps(c, ensure_ascii=False) + "\n")
print({(c["role"], c["lect"]): sum(1 for d in clips if (d["role"], d["lect"]) == (c["role"], c["lect"])) for c in clips})
