"""Dataset identification and integrity checks (Week 2-A), plus the numbers the dashboard shows.

Usage: python dataset_stats.py path/to/dataset.xlsx   (writes models/dataset_stats.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import composition

FEATURES = [
    "Ion_A", "Ion_B", "EN_A", "EN_B", "R_A", "R_B", "Sa_A", "Sa_B", "Delec_B", "t", "Ionic_A", "Ionic_B",
    "Avg_Ionic_Character", "SmithAcid_A", "SmithAcid_B", "1000/T", "Gap", "Oxide_vacancy", "E_above_hull",
    "TEC", "Work_Function", "Log_dieletric", "O_pband",
]
SOURCE = "k* Perovskite Oxygen Exchange Kinetics Dataset, Zenodo, DOI 10.5281/zenodo.16872672"


def load(path, sheet=0):
    path = str(path)
    if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(path, sheet_name=int(sheet) if str(sheet).isdigit() else sheet)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean(df, target="LogKstar"):
    """Drop unusable rows (missing or non-numeric values, exact duplicates) and report what happened."""
    rep = {"source": SOURCE, "license": "Check the licence on the Zenodo record before redistributing the data.",
           "rows_in": int(len(df))}
    num = df.copy()
    for c in FEATURES + [target]:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    bad = num[FEATURES + [target]].isna().any(axis=1) | ~np.isfinite(num[FEATURES + [target]]).all(axis=1)
    rep["dropped_missing_or_non_numeric"] = int(bad.sum())
    num = num[~bad]
    dup = num.duplicated()
    rep["dropped_duplicates"] = int(dup.sum())
    num = num[~dup].reset_index(drop=True)
    rep["rows_out"] = int(len(num))
    rep["materials"] = int(num["Material composition"].nunique()) if "Material composition" in num else None
    return num, rep


def _hist(values, bins=12):
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [round(float(e), 2) for e in edges], "counts": [int(c) for c in counts]}


def stats(df, report, target="LogKstar"):
    out = {"report": report, "logk_hist": _hist(df[target]), "temp_hist": _hist(1000 / df["1000/T"] - 273.15, 10)}
    for site in "AB":
        if site in df.columns:
            count = {}
            for s in df[site].drop_duplicates():
                for e in composition.parse_site(s):
                    count[e] = count.get(e, 0) + 1
            out[f"{site.lower()}_elements"] = dict(sorted(count.items(), key=lambda kv: -kv[1]))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    d, r = clean(load(sys.argv[1]))
    s = stats(d, r)
    print(json.dumps(s["report"], indent=2))
    out = Path(__file__).parent / "models"
    out.mkdir(exist_ok=True)
    (out / "dataset_stats.json").write_text(json.dumps(s))
