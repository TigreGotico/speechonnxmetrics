#!/usr/bin/env python
"""Export UTMOS (SpeechMOS ``utmos22_strong``) to ONNX with a dynamic time axis.

The upstream module is ``torch.hub.load("tarepan/SpeechMOS", "utmos22_strong")``,
whose ``forward(wave, sr)`` resamples internally.  We fix ``sr=16000`` (a no-op
inside ``torchaudio.functional.resample``) so the exported graph takes a plain
16 kHz mono waveform and needs no sample-rate argument.

Usage:
    python export_utmos.py --out utmos22_strong.onnx [--opset 17]
"""
import argparse

import torch


def patch_pad_to_multiple() -> None:
    """Remove the data-dependent tail padding from the wav2vec2 encoder.

    ``fairseq_alt.pad_to_multiple`` pads the frame axis up to a multiple of 2 and
    branches on ``float.is_integer()``, which breaks under tracing (the size is a
    Tensor).  The padding is numerically inert: padded frames are masked out of
    self-attention and sliced off afterwards, so real positions attend only to
    real positions either way.  Forcing the no-pad path (``pad_length == 0``,
    ``padding_mask is None``) therefore yields identical outputs -- which the
    parity check against the unpatched torch model verifies empirically.
    """
    from speechmos.utmos22 import fairseq_alt

    fairseq_alt.pad_to_multiple = lambda x, multiple, dim=-1, value=0: (x, 0)


class UTMOS16k(torch.nn.Module):
    """Thin wrapper pinning the sample rate to 16 kHz."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, wave: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B,)
        return self.inner(wave, 16000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="utmos22_strong.onnx", help="output .onnx path")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    ap.add_argument("--dummy-seconds", type=float, default=4.0,
                    help="length of the tracing example (time axis is dynamic anyway)")
    args = ap.parse_args()

    inner = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
    inner.eval()
    patch_pad_to_multiple()
    model = UTMOS16k(inner).eval()

    dummy = torch.randn(1, int(args.dummy_seconds * 16000)) * 0.1

    torch.onnx.export(
        model,
        (dummy,),
        args.out,
        input_names=["wave"],
        output_names=["mos"],
        dynamic_axes={"wave": {0: "batch", 1: "time"}, "mos": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
