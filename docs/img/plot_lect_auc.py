"""Plot the held-out clip AUC of each lect-fidelity calibration.

Reads speechonnxmetrics/data/lect_calibration.json. Run from the repository root:
    python docs/img/plot_lect_auc.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "speechonnxmetrics" / "data" / "lect_calibration.json"
CORPORA = ("CLUL spoken varieties held-out recordings", "compare-accents-pt")


def read():
    rows = []
    for entry in json.loads(TABLE.read_text(encoding="utf-8"))["calibrations"].values():
        evaluation = entry["provenance"]["evaluation"]
        rows.append({
            "provider": entry["provider"],
            "backend": entry["backend"],
            **{corpus: {"auc": evaluation[corpus]["auc"], "auc_95ci": evaluation[corpus]["auc_95ci"]}
               for corpus in CORPORA},
        })
    return rows


def draw(fig, theme, rows):
    ax = fig.subplots()
    backends = sorted({r["backend"] for r in rows})
    providers = sorted({r["provider"] for r in rows})
    height = 0.8 / len(providers)
    for i, provider in enumerate(providers):
        for j, backend in enumerate(backends):
            row = next(r for r in rows if r["provider"] == provider and r["backend"] == backend)
            m = row[CORPORA[0]]
            y = j + (i - (len(providers) - 1) / 2) * height
            low, high = m["auc_95ci"]
            ax.barh(y, m["auc"], height=height * 0.9, color=theme["series"][i],
                    xerr=[[m["auc"] - low], [high - m["auc"]]], ecolor=theme["muted"], capsize=3,
                    label=provider if j == 0 else None)
            ax.text(high + 0.01, y, f"{m['auc']:.2f}", va="center", color=theme["text"])
    ax.set_yticks(range(len(backends)), backends)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("clip AUC on the CLUL held-out recordings (bars: 95% interval)")
    ax.set_title("pt-PT against pt-BR, one clip at a time")
    ax.legend(title="provider", loc="lower right")


if __name__ == "__main__":
    save(str(ROOT / "docs" / "img" / "lect-auc"), draw, read())
