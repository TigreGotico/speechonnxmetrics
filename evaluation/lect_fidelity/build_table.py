#!/usr/bin/env python
"""Write ``speechonnxmetrics/data/lect_calibration.json`` from the ``evaluate.py`` outputs.

Each entry is the calibration of one provider and one backend, decoded with the
inventory restricted to the two lects' phones and measured on the training recordings of
the CLUL spoken-varieties corpus. Usage, from this directory, with the outputs of
``evaluate.py`` for both providers and both backends::

    python build_table.py data/eval-clul-*.json
"""
import json
import sys
from pathlib import Path

from speechonnxmetrics.lect_fidelity.calibration import Calibration, table_entry

TABLE = Path(__file__).resolve().parents[2] / "speechonnxmetrics" / "data" / "lect_calibration.json"
LICENCES = {
    "CLUL spoken varieties training recordings": "Jarbas/SpokenPortugueseGeographicalSocialVarieties_splits, MIT",
    "CLUL spoken varieties held-out recordings": "Jarbas/SpokenPortugueseGeographicalSocialVarieties_splits, MIT",
    "compare-accents-pt": "TigreGotico/compare-accents-pt, licence not stated on the card; evaluation only",
    "EuroSpeech Portugal validation": "disco-eth/EuroSpeech; Portuguese parliament recordings, "
    "Portuguese Copyright Code article 75 per the dataset card",
    "FLEURS pt_br dev": "google/fleurs, CC-BY-4.0",
    "FLEURS pt_br test": "google/fleurs, CC-BY-4.0",
    "Speech-MASSIVE pt-PT": "TigreGotico/speech_MASSIVE_pt-PT (from FBK-MT/Speech-MASSIVE), CC-BY-NC-SA-4.0; "
    "evaluation only",
}

calibrations = {}
for path in sys.argv[1:]:
    for run in json.loads(Path(path).read_text()):
        if run["mode"] == "none":
            continue
        cal = Calibration.from_dict(run["calibration"])
        cal.provenance["licences"] = {c: LICENCES[c] for role in cal.provenance["corpora"].values() for c in role}
        cal.provenance["split"] = "by recording: no recording is in both the calibration and the evaluation part"
        cal.provenance["evaluation"] = {
            corpus: {"auc": round(m["auc"], 4), "auc_95ci": [round(x, 4) for x in m["auc_ci"]]}
            for corpus, m in run["evaluation"].items() if m.get("auc") is not None
        }
        key, value = table_entry(cal)
        calibrations[key] = value
TABLE.write_text(json.dumps({"calibrations": calibrations}, ensure_ascii=False, indent=1) + "\n")
print(f"wrote {len(calibrations)} calibrations to {TABLE}")
