#!/usr/bin/env python
"""Export the Allosaurus universal phone recogniser to ONNX.

Allosaurus (X. Li et al., "Universal Phone Recognition with a Multilingual Allophone
System", ICASSP 2020) is a five-layer bidirectional LSTM over 120-dimensional stacked
MFCC frames, trained with CTC on 2,000 universal phones of which 229 are output units.
Code and pretrained model ``uni2005`` are GPL-3.0, from
https://github.com/xinjli/allosaurus, and the exported graph carries the same licence.

The graph covers only the acoustic model: ``feats`` ``(1, T, 120)`` float32 to
``logits`` ``(1, T, 230)``, unit 0 being the CTC blank. The MFCC frontend runs in numpy
inside ``speechonnxmetrics.lect_fidelity.backends``, written from the Kaldi MFCC
definition with the parameters of the model's ``pm_config.json``.

With ``--clips``, the script also writes ``allosaurus_reference.json``: the phones
Allosaurus's own recogniser emits for each test clip after the clip is resampled to
8 kHz by this package's resampler and stored as 16-bit PCM, against which the ONNX
pipeline is checked. Allosaurus resamples with ``resampy`` otherwise, and the choice of
resampler alone changes about one phone in six on these clips.

Usage, in an environment with ``allosaurus``, ``torch`` and ``onnx`` installed::

    python export_allosaurus.py --model-dir DIR --out allosaurus_uni2005.onnx \
        [--clips ../test/fixtures/audio/lect]

``DIR`` holds the extracted ``uni2005`` release
(https://github.com/xinjli/allosaurus/releases/download/v1.0/latest.tar.gz).
"""
import argparse
import json
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch


class AcousticModel(torch.nn.Module):
    """The Allosaurus BLSTM without sequence packing, for a single utterance."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.lstm = inner.blstm_layer
        self.proj = inner.phone_layer

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (1, T, 120) -> (1, T, 230)
        hidden, _ = self.lstm(feats.transpose(0, 1))
        return self.proj(hidden).transpose(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path, help="directory holding uni2005/")
    ap.add_argument("--out", default="allosaurus_uni2005.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--clips", type=Path, help="directory of 16 kHz WAV clips for the reference phones")
    args = ap.parse_args()

    from allosaurus.app import read_recognizer

    rec = read_recognizer("uni2005", alt_model_path=args.model_dir)
    rec.am.eval()
    model = AcousticModel(rec.am).eval()

    dummy = torch.randn(1, 100, 120)
    torch.onnx.export(
        model,
        (dummy,),
        args.out,
        input_names=["feats"],
        output_names=["logits"],
        dynamic_axes={"feats": {1: "time"}, "logits": {1: "time"}},
        opset_version=args.opset,
        dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    probe = torch.randn(1, 333, 120)
    with torch.no_grad():
        want = rec.am(probe, torch.tensor([333])).numpy()
    got = sess.run(None, {"feats": probe.numpy()})[0]
    print(f"wrote {args.out}; max |onnx - torch| = {np.abs(got - want).max():.2e}")

    if args.clips:
        from speechonnxmetrics._dsp.audio import load_audio
        from speechonnxmetrics._dsp.resample import kaiser_resample

        ref = {}
        with tempfile.TemporaryDirectory() as tmp:
            for wav in sorted(args.clips.glob("*.wav")):
                x, sr = load_audio(str(wav))
                pcm = (np.clip(kaiser_resample(x, sr, 8000), -1, 1) * 32767).astype(np.int16)
                path = Path(tmp) / wav.name
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(8000)
                    w.writeframes(pcm.tobytes())
                ref[wav.name] = rec.recognize(str(path), "ipa").split()
        out = Path(args.out).with_name("allosaurus_reference.json")
        out.write_text(json.dumps(ref, ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
