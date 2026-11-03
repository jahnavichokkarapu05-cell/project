"""
Train, validate and save the models. Also called by the developer panel.

    python train.py path/to/dataset.xlsx [--target LogKstar] [--seed 42] [--sheet NAME]

Splits are made by material (70/15/15, fixed seed) so no material appears in two splits.
Run it with the same scikit-learn version you serve with.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import composition
from dataset_stats import FEATURES, clean, load, stats

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
FILES = {("full", "svr"): "perovskite_svr_pipeline.pkl", ("full", "rf"): "perovskite_rf_model.pkl",
         ("name_only", "svr"): "perovskite_comp_svr_pipeline.pkl", ("name_only", "rf"): "perovskite_comp_rf_model.pkl"}


def split_by_material(n, groups, seed):
    tr, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(np.zeros(n), groups=groups))
    va, te = next(GroupShuffleSplit(1, test_size=0.50, random_state=seed).split(np.zeros(len(rest)), groups=groups[rest]))
    va, te = rest[va], rest[te]
    sets = [set(groups[i]) for i in (tr, va, te)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "A material appears in two splits"
    return tr, va, te


def make(kind, seed):
    if kind == "svr":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1))
    return RandomForestRegressor(n_estimators=100, random_state=seed)


def run(data, out_dir, seed=42, target="LogKstar", sheet=0, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = load(data, sheet)
    missing = [c for c in FEATURES + [target] if c not in raw.columns]
    if missing:
        raise ValueError(f"Columns not found in the data file: {missing}. File has: {list(raw.columns)}")
    df, report = clean(raw, target)
    log(f"Rows: {report['rows_in']} in, {report['rows_out']} usable")
    X, y = df[FEATURES], df[target]
    has_names = "Material composition" in df.columns
    groups = df["Material composition"].to_numpy() if has_names else df.index.to_numpy()
    tr, va, te = split_by_material(len(df), groups, seed)

    # Preprocessing outputs: standardised arrays per split (scaler fitted on train only) and the split lists.
    scaler = StandardScaler().fit(X.iloc[tr])
    (out / "processed").mkdir(exist_ok=True)
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        np.save(out / "processed" / f"X_{name}.npy", scaler.transform(X.iloc[idx]).astype("float32"))
        np.save(out / "processed" / f"y_{name}.npy", y.iloc[idx].to_numpy("float32"))
    (out / "splits.json").write_text(json.dumps({
        "seed": seed, "features": FEATURES,
        **{n: sorted({str(groups[i]) for i in idx}) for n, idx in (("train", tr), ("val", va), ("test", te))}}))

    metrics = {"created": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC", "seed": seed, "rows": len(df),
               "materials": int(len(set(groups))), "split_rows": {"train": len(tr), "val": len(va), "test": len(te)},
               "split_check": "no material appears in more than one split", "models": {}}
    for tag, cols in (("full", FEATURES), ("name_only", composition.COMP_FEATURES)):
        for kind in ("svr", "rf"):
            m = make(kind, seed).fit(X.iloc[tr][cols], y.iloc[tr])
            sc = {}
            for nm, idx in (("val", va), ("test", te)):
                p = m.predict(X.iloc[idx][cols])
                sc[nm] = {"r2": round(float(r2_score(y.iloc[idx], p)), 3), "mae": round(float(mean_absolute_error(y.iloc[idx], p)), 3)}
            cv = cross_val_predict(make(kind, seed), X[cols], y, groups=groups, cv=GroupKFold(5))
            sc["grouped_cv"] = {"r2": round(float(r2_score(y, cv)), 3), "mae": round(float(mean_absolute_error(y, cv)), 3)}
            metrics["models"][f"{tag}_{kind}"] = sc
            log(f"{tag:9s} {kind}: test R2={sc['test']['r2']}  grouped CV R2={sc['grouped_cv']['r2']}")
            joblib.dump(make(kind, seed).fit(X[cols], y), out / FILES[(tag, kind)])

    meta = {"logk_scale": [float(y.min()), float(y.max())], "n_rows": int(len(df)),
            "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES}}
    (out / "meta.json").write_text(json.dumps(meta))
    (out / "metrics.json").write_text(json.dumps(metrics))
    (out / "dataset_stats.json").write_text(json.dumps(stats(df, report, target)))
    if {"A", "B"} <= set(df.columns):
        (out / "elements.json").write_text(json.dumps(composition.fit_tables(df)))
    if has_names:
        uniq = df.drop_duplicates("Material composition").sort_values(target).reset_index(drop=True)
        examples = []
        for i in sorted({int(round(i)) for i in np.linspace(0, len(uniq) - 1, 6)}):
            r = uniq.iloc[i]
            t_c = 1000 / r["1000/T"] - 273.15
            examples.append({"label": f"{r['Material composition']} at {t_c:.0f} \u00b0C", "formula": str(r["Material composition"]),
                             "temp_c": round(t_c), "values": {c: float(r[c]) for c in FEATURES}, "true_log_kstar": float(r[target])})
        (out / "examples.json").write_text(json.dumps(examples))
    log(f"Saved to {out}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help=".xlsx or .csv file")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--target", default="LogKstar")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.data, MODEL_DIR, a.seed, a.target, a.sheet)
    dest = BASE / "data" / ("dataset" + Path(a.data).suffix.lower())  # lets the developer panel retrain later
    dest.parent.mkdir(exist_ok=True)
    if Path(a.data).resolve() != dest.resolve():
        shutil.copy(a.data, dest)


if __name__ == "__main__":
    main()
