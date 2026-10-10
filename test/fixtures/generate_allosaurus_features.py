#!/usr/bin/env python
"""Write ``allosaurus_features_reference.npz``: Allosaurus's own MFCC frontend on one
test clip, the reference the numpy frontend in ``lect_fidelity.backends`` is checked
against without the model.

The clip is resampled to 8 kHz by this package's resampler and quantised to 16 bits,
as the backend does, then passed to Allosaurus's feature model with its configuration
from ``uni2005/pm_config.json`` and dithering off (Allosaurus dithers by default, which
makes its features random).

Usage, in an environment with ``allosaurus`` installed (its feature model needs only
numpy and scipy)::

    python generate_allosaurus_features.py --model-dir DIR/uni2005
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from speechonnxmetrics._dsp.audio import load_audio
from speechonnxmetrics._dsp.resample import kaiser_resample

HERE = Path(__file__).parent
CLIP = HERE / "audio" / "lect" / "pt-PT_0.wav"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path, help="the extracted uni2005 directory")
    args = ap.parse_args()

    from allosaurus.audio import Audio
    from allosaurus.pm.mfcc import MFCC

    config = json.loads((args.model_dir / "pm_config.json").read_text())
    config["dither"] = 0.0
    x, sr = load_audio(str(CLIP))
    pcm = (np.clip(kaiser_resample(x, sr, 8000), -1, 1) * 32767).astype(np.int16)
    feats = MFCC(Namespace(**config)).compute(Audio(pcm, 8000))
    np.savez_compressed(HERE / "allosaurus_features_reference.npz", pcm=pcm, feats=feats.astype(np.float32))
    print(f"{CLIP.name}: {feats.shape}")


if __name__ == "__main__":
    main()
