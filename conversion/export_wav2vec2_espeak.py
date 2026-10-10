#!/usr/bin/env python
"""Export ``facebook/wav2vec2-xlsr-53-espeak-cv-ft`` to ONNX.

The model is XLSR-53 (A. Conneau et al., "Unsupervised Cross-lingual Representation
Learning for Speech Recognition", 2020) fine-tuned with CTC on Common Voice
transcribed into espeak-ng phones (Q. Xu, A. Baevski, M. Auli, "Simple and Effective
Zero-shot Cross-lingual Phoneme Recognition", 2021). Weights are Apache-2.0, and the
exported graph carries the same licence.

The graph takes ``input_values`` ``(1, T)``, a 16 kHz waveform normalised to zero mean
and unit variance, and returns ``logits`` ``(1, T', 392)``; unit 0 is the CTC blank
``<pad>``. The normalisation runs in numpy inside
``speechonnxmetrics.lect_fidelity.backends``. ``vocab.json`` is copied beside the graph.

With ``--clips``, the script also writes ``wav2vec2_espeak_reference.json``: the symbols
the torch model decodes greedily for each test clip, against which the ONNX pipeline is
checked.

Usage, in an environment with ``transformers``, ``torch`` and ``onnx`` installed::

    python export_wav2vec2_espeak.py --out wav2vec2_xlsr53_espeak_cv_ft.onnx
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

REPO = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
REVISION = "2c733782da5604684829819a5eb744c193fe9398"


class Logits(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.inner(input_values).logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="wav2vec2_xlsr53_espeak_cv_ft.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--clips", type=Path, help="directory of 16 kHz WAV clips for the reference symbols")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2ForCTC

    inner = Wav2Vec2ForCTC.from_pretrained(REPO, revision=REVISION).eval()
    model = Logits(inner).eval()

    dummy = torch.randn(1, 32000)
    torch.onnx.export(
        model,
        (dummy,),
        args.out,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {1: "time"}, "logits": {1: "frames"}},
        opset_version=args.opset,
        dynamo=False,
    )
    shutil.copy(hf_hub_download(REPO, "vocab.json", revision=REVISION), Path(args.out).with_name("vocab.json"))

    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    probe = torch.randn(1, 51234)
    with torch.no_grad():
        want = model(probe).numpy()
    got = sess.run(None, {"input_values": probe.numpy()})[0]
    print(f"wrote {args.out}; max |onnx - torch| = {np.abs(got - want).max():.2e}")

    if args.clips:
        from speechonnxmetrics._dsp.audio import load_audio

        vocab = json.loads(Path(args.out).with_name("vocab.json").read_text())
        symbol = {i: s for s, i in vocab.items()}
        ref = {}
        for wav in sorted(args.clips.glob("*.wav")):
            x, _ = load_audio(str(wav), target_sr=16000)
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
            with torch.no_grad():
                best = model(torch.from_numpy(x.astype(np.float32))[None]).argmax(-1)[0].tolist()
            units = [u for k, u in enumerate(best) if u != 0 and (k == 0 or u != best[k - 1])]
            ref[wav.name] = [symbol[u] for u in units]
        out = Path(args.out).with_name("wav2vec2_espeak_reference.json")
        out.write_text(json.dumps(ref, ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
