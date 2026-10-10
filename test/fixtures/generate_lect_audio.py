#!/usr/bin/env python
"""Synthesise the lect-fidelity test clips with two public Piper voices.

Source: the Piper voices ``pt_PT-tugão-medium`` and ``pt_BR-faber-medium`` from the
Hugging Face repo ``rhasspy/piper-voices`` at commit
``c10ece1aade47bb51c153c893d14e5bf8e5b7117``. The repo is MIT licensed and both voices
are trained on CC0 recordings from ``OHF-Voice/voice-datasets``, as their model cards
state. Both cards also say the voices were fine-tuned from the ``en_US`` lessac voice,
whose card points to the Blizzard 2013 Lessac licence: research purposes only, no
commercial use and no distribution of the materials. Whether that reaches audio
synthesised by a fine-tuned voice is unclear, so these clips are not described as freely
redistributable.

Each sentence is spoken once by each voice. Piper drives its voices with espeak-ng, so
the European voice realises the European variants of the test sites (coda ``s`` as
``ʃ``, reduced unstressed ``e``, velarised coda ``l``) and the Brazilian voice the
Brazilian ones (coda ``s`` as ``s``, ``t``/``d`` affricated before ``i``, vocalised
coda ``l``). The espeak-ng phonemes each voice received are kept in the manifest.

Usage, in an environment with ``piper-tts`` installed::

    python generate_lect_audio.py --voices DIR

``DIR`` holds the four voice files ``pt_PT-tugao-medium.onnx[.json]`` and
``pt_BR-faber-medium.onnx[.json]``.
"""
import argparse
import io
import json
import wave
from pathlib import Path

import numpy as np

from speechonnxmetrics._dsp.resample import kaiser_resample

OUT = Path(__file__).parent / "audio" / "lect"
REVISION = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"
LICENCE = (
    "voice repo MIT; voice training data CC0; the voice is fine-tuned from the en_US lessac voice, "
    "trained on Blizzard 2013 Lessac data licensed for non-commercial research only, so whether "
    "this audio may be redistributed freely is unclear"
)
VOICES = {
    "pt-PT": ("pt_PT-tugao-medium", "pt/pt_PT/tugão/medium/pt_PT-tugão-medium.onnx"),
    "pt-BR": ("pt_BR-faber-medium", "pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx"),
}
SENTENCES = [
    "As casas estão fechadas.",
    "O leite está quente.",
    "Tive de partir cedo.",
    "O papel é azul.",
    "Os meninos comem pão.",
    "Mais tarde vamos dormir.",
]
SR = 16000
#: The voices speak fast at their default pace; a slower pace gives the recognisers
#: clearer segments. Each clip is padded with silence and peak-normalised to 0.9.
LENGTH_SCALE = 1.5
PAD_SECONDS = 0.25
PEAK = 0.9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voices", required=True, type=Path)
    args = ap.parse_args()

    from piper import PiperVoice
    from piper.config import SynthesisConfig

    config = SynthesisConfig(length_scale=LENGTH_SCALE, noise_scale=0.0, noise_w_scale=0.0)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for lect, (stem, hub_path) in VOICES.items():
        voice = PiperVoice.load(str(args.voices / f"{stem}.onnx"))
        for i, text in enumerate(SENTENCES):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                voice.synthesize_wav(text, w, syn_config=config)
            buf.seek(0)
            with wave.open(buf, "rb") as r:
                sr = r.getframerate()
                pcm = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16)
            audio = kaiser_resample(pcm.astype(np.float32) / 32768.0, sr, SR)
            audio = PEAK * audio / np.abs(audio).max()
            pad = np.zeros(int(PAD_SECONDS * SR), dtype=np.float32)
            audio = np.concatenate([pad, audio, pad])
            name = f"{lect}_{i}.wav"
            with wave.open(str(OUT / name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
            manifest.append({
                "file": name,
                "text": text,
                "lect": lect,
                "voice": f"rhasspy/piper-voices@{REVISION}:{hub_path}",
                "licence": LICENCE,
                "length_scale": LENGTH_SCALE,
                "espeak_phonemes": " ".join("".join(p) for p in voice.phonemize(text)),
            })
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(manifest)} clips to {OUT}")


if __name__ == "__main__":
    main()
