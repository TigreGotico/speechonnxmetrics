#!/usr/bin/env python
"""Decode every clip with one backend, with and without the inventory restriction.

Usage, from this directory, with the ONNX exports in ``onnx/``::

    python recognise.py allosaurus data/clips-clul.jsonl
    python recognise.py wav2vec2_espeak data/clips-clul-w2v.jsonl

Appends to ``data/phones-<backend>.jsonl`` and skips clips already there.
"""
import json
import sys
import time
from pathlib import Path

import onnxruntime as ort

from speechonnxmetrics.lect_fidelity import AllosaurusBackend, Wav2Vec2EspeakBackend
from speechonnxmetrics.lect_fidelity.providers import Orthography2ipaProvider, TugaphoneProvider
from speechonnxmetrics._dsp.audio import load_audio

ONNX = Path("onnx")
name = sys.argv[1]
backend = {"allosaurus": lambda: AllosaurusBackend(model=str(ONNX / "allosaurus_uni2005.onnx")),
           "wav2vec2_espeak": lambda: Wav2Vec2EspeakBackend(model=str(ONNX / "wav2vec2_xlsr53_espeak_cv_ft.onnx"))}[name]()
opts = ort.SessionOptions()
opts.intra_op_num_threads = int(__import__("os").environ.get("OMP_NUM_THREADS", "4"))
opts.inter_op_num_threads = 1
backend._session = ort.InferenceSession(backend.model.hf_file, opts, providers=["CPUExecutionProvider"])
lects = ("pt-PT", "pt-BR")
inventories = {}
for prov in (Orthography2ipaProvider(), TugaphoneProvider()):
    inventories[prov.name] = prov.inventory(lects[0]) | prov.inventory(lects[1])
out = Path(f"data/phones-{name}.jsonl")
done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
clips = [json.loads(l) for f in sys.argv[2:] for l in open(f)]
t0 = time.time()
with out.open("a") as fh:
    for i, c in enumerate(clips):
        if c["id"] in done:
            continue
        x, sr = load_audio(c["path"])
        logits = backend.logits(x, sr)
        row = {"id": c["id"], "duration": len(x) / sr,
               "phones": {"none": backend.phones_from_logits(logits)}}
        for k, inv in inventories.items():
            row["phones"][k] = backend.phones_from_logits(logits, inv)
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        if i % 50 == 0:
            print(name, i, round(time.time() - t0), flush=True)
print("RECOGNISE DONE", name, flush=True)
