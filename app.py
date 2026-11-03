"""
k* predictor: Flask backend for the perovskite oxygen-exchange model.

Loads the models saved at the end of the notebook (ps_1.ipynb):
  models/perovskite_svr_pipeline.pkl   StandardScaler + SVR(rbf, C=10, epsilon=0.1)   (required)
  models/perovskite_rf_model.pkl       RandomForestRegressor(n_estimators=100)         (optional)

Run:  python app.py   ->  http://127.0.0.1:5000
"""
import datetime
import hmac
import json
import math
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import composition
import pandas as pd
import sklearn
import train as trainer
from flask import Flask, jsonify, request

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
SVR_PATH = MODEL_DIR / "perovskite_svr_pipeline.pkl"
RF_PATH = MODEL_DIR / "perovskite_rf_model.pkl"

# Position of the result marker on the "slower -> faster" scale in the UI.
# Replace with the min / max of LogKstar in your dataset.
LOGK_SCALE = (-9.0, -4.0)  # fallback; overridden by models/meta.json written by train.py
RANGES, EXAMPLES = {}, []


def load_meta():
    global LOGK_SCALE, RANGES, EXAMPLES
    try:
        meta = json.loads((MODEL_DIR / "meta.json").read_text())
        LOGK_SCALE, RANGES = tuple(meta["logk_scale"]), meta.get("ranges", {})
    except Exception:
        pass
    try:
        EXAMPLES = json.loads((MODEL_DIR / "examples.json").read_text())
    except Exception:
        EXAMPLES = []


# Temperature sweep used for the Arrhenius-style chart (degrees C).
SWEEP_C = list(range(600, 1001, 50))

# Starting values: first row of Kstar_DataSet_Pro.xlsx (La0.9Sr0.1Ga0.8Mg0.2O3 at 1000/T = 0.852).
EXAMPLE = {
    "Ion_A": 18.365,
    "Ion_B": 25.406,
    "EN_A": 1.2118,
    "EN_B": 1.71,
    "R_A": 1.368,
    "R_B": 0.64,
    "Sa_A": 0.3309,
    "Sa_B": 0.5868,
    "Delec_B": 0.0,
    "t": 0.9594,
    "Ionic_A": 0.7529,
    "Ionic_B": 0.7037,
    "Avg_Ionic_Character": 0.7283,
    "SmithAcid_A": -6.43,
    "SmithAcid_B": -2.18,
    "1000/T": 0.852,
    "Gap": 5.3114,
    "Oxide_vacancy": -5.6757,
    "E_above_hull": 101.6109,
    "TEC": 13.5824,
    "Work_Function": 2.9974,
    "Log_dieletric": 1.189,
    "O_pband": -2.0251,
}

# Short hints shown under inputs. Edit freely; missing names just show no hint.
HINTS = {
    "t": "Goldschmidt tolerance factor",
    "1000/T": "Inverse temperature, 1000 / T(K)",
    "Sa_A": "Lewis acidity, A-site",
    "Sa_B": "Lewis acidity, B-site",
    "SmithAcid_A": "Smith acidity, A-site",
    "SmithAcid_B": "Smith acidity, B-site",
    "Delec_B": "d-electron count, B-site",
    "Log_dieletric": "Log of dielectric constant",
    "Gap": "Band gap",
    "E_above_hull": "Energy above the convex hull",
    "TEC": "Thermal expansion coefficient",
    "Work_Function": "Work function",
    "O_pband": "Oxygen p-band centre",
    "Oxide_vacancy": "Oxygen vacancy descriptor",
    "EN_A": "Electronegativity, A-site",
    "EN_B": "Electronegativity, B-site",
    "R_A": "Radius, A-site",
    "R_B": "Radius, B-site",
}

app = Flask(__name__, static_folder="static", static_url_path="")

_state = {"svr": None, "rf": None, "csvr": None, "crf": None, "tables": None, "explainer": None, "error": None}


def load_models():
    load_meta()
    _state["error"] = None
    try:
        _state["svr"] = joblib.load(SVR_PATH)
    except FileNotFoundError:
        _state["error"] = (
            f"Model file not found: {SVR_PATH.relative_to(BASE)}. "
            "Copy perovskite_svr_pipeline.pkl from your notebook folder into models/."
        )
    except Exception as exc:  # version mismatch, corrupt file, ...
        _state["error"] = f"Could not load {SVR_PATH.name}: {exc}"
    for key, name in (("csvr", "perovskite_comp_svr_pipeline.pkl"), ("crf", "perovskite_comp_rf_model.pkl")):
        try:
            _state[key] = joblib.load(MODEL_DIR / name)
        except Exception:
            _state[key] = None
    try:
        _state["tables"] = json.loads((MODEL_DIR / "elements.json").read_text())
    except Exception:
        _state["tables"] = None
    if RF_PATH.exists():
        try:
            _state["rf"] = joblib.load(RF_PATH)
        except Exception:
            _state["rf"] = None


def features():
    return list(_state["svr"].feature_names_in_)


def group_of(name):
    if name == "1000/T":
        return "Conditions"
    if name.endswith("_A"):
        return "A-site"
    if name.endswith("_B"):
        return "B-site"
    return "Structure and properties"


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/schema")
def schema():
    if _state["error"]:
        return jsonify(error=_state["error"]), 503
    names = features()
    return jsonify(
        features=[
            {
                "name": n,
                "group": group_of(n),
                "default": EXAMPLE.get(n),
                "hint": HINTS.get(n, ""),
            }
            for n in names
        ],
        models={"svr": True, "rf": _state["rf"] is not None},
        scale=LOGK_SCALE,
        examples=EXAMPLES,
        composition=_state["csvr"] is not None and _state["tables"] is not None,
        elements={s: sorted(_state["tables"][s]) for s in "AB"} if _state["tables"] else None,
        has_temperature="1000/T" in names,
        sklearn_version=sklearn.__version__,
    )


def frame_from(values):
    names = features()
    row, missing = {}, []
    for n in names:
        raw = values.get(n)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            missing.append(n)
            continue
        if not math.isfinite(v):
            missing.append(n)
            continue
        row[n] = v
    if missing:
        raise ValueError("Enter a number for: " + ", ".join(missing))
    return pd.DataFrame([row], columns=names)  # column order = training order


def rf_contributions(df):
    """Per-feature SHAP contributions for the Random Forest (optional dependency)."""
    try:
        import shap

        if _state["explainer"] is None:
            _state["explainer"] = shap.TreeExplainer(_state["rf"])
        sv = np.ravel(_state["explainer"].shap_values(df)[0])
        order = np.argsort(-np.abs(sv))[:10]
        cols = list(df.columns)
        return [
            {"feature": cols[i], "value": float(df.iloc[0, i]), "contribution": float(sv[i])}
            for i in order
        ]
    except Exception:
        return None


def respond(df, which, svr, rf, extra_warnings=()):
    model = rf if which == "rf" else svr
    t0 = time.perf_counter()
    y = float(model.predict(df)[0])
    latency_ms = (time.perf_counter() - t0) * 1000

    trace = {"columns": list(df.columns), "latency_ms": round(latency_ms, 2)}
    if which == "svr":
        scaler = svr.named_steps["standardscaler"]
        est = svr.named_steps["svr"]
        z = scaler.transform(df)[0]
        trace.update(
            scaled=[float(v) for v in z],
            n_support_vectors=int(len(est.support_)),
            params={"kernel": est.kernel, "C": est.C, "epsilon": est.epsilon},
        )
    else:
        trace.update(n_trees=int(len(model.estimators_)), contributions=rf_contributions(df) if rf is _state["rf"] else None)

    sweep = None
    if "1000/T" in df.columns:
        rows = []
        for c in SWEEP_C:
            r = df.iloc[0].copy()
            r["1000/T"] = 1000.0 / (c + 273.15)
            rows.append(r)
        sy = model.predict(pd.DataFrame(rows, columns=df.columns))
        sweep = [
            {"temp_c": c, "inv_t": float(1000.0 / (c + 273.15)), "log_kstar": float(v)}
            for c, v in zip(SWEEP_C, sy)
        ]

    warnings = list(extra_warnings) + [
        f"{n} = {df.iloc[0][n]:g} is outside the training range ({lo:g} to {hi:g})"
        for n, (lo, hi) in RANGES.items()
        if n in df.columns and not (lo <= df.iloc[0][n] <= hi)
    ]
    return {
        "model": which, "warnings": warnings, "log_kstar": y,
        "kstar": float(10.0 ** y),  # assumes LogKstar is log10
        "sweep": sweep, "trace": trace,
    }


@app.post("/api/predict")
def predict():
    if _state["error"]:
        return jsonify(error=_state["error"]), 503
    body = request.get_json(silent=True) or {}
    which = body.get("model", "svr")
    if which == "rf" and _state["rf"] is None:
        return jsonify(error="The Random Forest model file was not found in models/."), 400
    try:
        df = frame_from(body.get("values", {}))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(respond(df, which, _state["svr"], _state["rf"]))


@app.post("/api/predict_composition")
def predict_composition():
    if _state["csvr"] is None or _state["tables"] is None:
        return jsonify(error="Composition models not found in models/. Run train.py again."), 503
    body = request.get_json(silent=True) or {}
    which = body.get("model", "svr")
    if which == "rf" and _state["crf"] is None:
        return jsonify(error="The composition Random Forest was not found in models/."), 400
    try:
        temp = float(body.get("temp_c"))
        if not math.isfinite(temp) or not -100 < temp < 3000:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify(error="Enter a temperature in \u00b0C, for example 800."), 400
    try:
        feats, sites, warns = composition.descriptors(body.get("formula", ""), temp, _state["tables"])
    except (ValueError, KeyError) as exc:
        return jsonify(error=str(exc)), 400
    df = pd.DataFrame([feats], columns=list(_state["csvr"].feature_names_in_))
    out = respond(df, which, _state["csvr"], _state["crf"], warns)
    out.update(descriptors=feats, sites=sites, formula=str(body.get("formula", "")).strip())
    return jsonify(out)


# ---------------------------------------------------------------- operations, security, analytics
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN") or secrets.token_urlsafe(8)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
CKPT = MODEL_DIR / "checkpoints"
_M = {"start": time.time(), "requests": 0, "errors": 0, "lat": []}
_job = {"status": "idle", "message": "", "checkpoint": None}


@app.before_request
def _start_timer():
    request._t0 = time.perf_counter()


@app.after_request
def _after(resp):
    if request.path.startswith("/api/"):
        _M["requests"] += 1
        _M["errors"] += resp.status_code >= 400
        _M["lat"] = (_M["lat"] + [(time.perf_counter() - request._t0) * 1000])[-200:]
        allowed = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]
        origin = request.headers.get("Origin")
        val = "*" if "*" in allowed else (origin if origin in allowed else None)
        if val:
            resp.headers["Access-Control-Allow-Origin"] = val
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Admin-Token"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Vary"] = "Origin"
    return resp


@app.errorhandler(413)
def _too_big(_):
    return jsonify(error="That file is too large (limit 5 MB)."), 413


def _read_json(name):
    try:
        return json.loads((MODEL_DIR / name).read_text())
    except Exception:
        return None


def _active_name():
    try:
        return (MODEL_DIR / "active.txt").read_text().strip()
    except Exception:
        return "initial"


@app.get("/api/health")
def health():
    lat = _M["lat"]
    return jsonify(
        status="degraded" if _state["error"] else "ok",
        models={k: _state[k] is not None for k in ("svr", "rf", "csvr", "crf")},
        sklearn=sklearn.__version__, uptime_s=round(time.time() - _M["start"]),
        requests=_M["requests"], errors=_M["errors"],
        avg_latency_ms=round(sum(lat) / len(lat), 1) if lat else None,
        active_checkpoint=_active_name(), training=_job["status"],
    )


@app.get("/api/analytics")
def analytics():
    return jsonify(dataset=_read_json("dataset_stats.json"), metrics=_read_json("metrics.json"), system=health().get_json())


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Choose a CSV or Excel file with the columns formula and temp_c."), 400
    if _state["csvr"] is None or _state["tables"] is None:
        return jsonify(error="Composition models not found in models/. Run train.py again."), 503
    try:
        df = pd.read_excel(f) if f.filename.lower().endswith((".xlsx", ".xls")) else pd.read_csv(f)
    except Exception:
        return jsonify(error="Could not read that file. Use .csv or .xlsx."), 400
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"formula", "temp_c"} <= set(df.columns):
        return jsonify(error="The file needs the columns: formula, temp_c."), 400
    if not 0 < len(df) <= 500:
        return jsonify(error="Upload between 1 and 500 rows."), 400
    which = "rf" if request.form.get("model") == "rf" and _state["crf"] is not None else "svr"
    model = _state["crf"] if which == "rf" else _state["csvr"]
    rows, ok = [], []
    for _, r in df.iterrows():
        item = {"formula": str(r["formula"]).strip(), "temp_c": None if pd.isna(r["temp_c"]) else r["temp_c"]}
        try:
            try:
                t = float(r["temp_c"])
            except (TypeError, ValueError):
                raise ValueError("temp_c must be a number.")
            if not math.isfinite(t) or not -100 < t < 3000:
                raise ValueError("temp_c must be a number between -100 and 3000.")
            feats, _, warns = composition.descriptors(item["formula"], t, _state["tables"])
            item["warnings"] = warns
            ok.append((len(rows), feats))
        except (ValueError, TypeError, KeyError) as exc:
            item["error"] = str(exc)
        rows.append(item)
    if ok:
        X = pd.DataFrame([f for _, f in ok], columns=list(_state["csvr"].feature_names_in_))
        for (idx, _), v in zip(ok, model.predict(X)):
            rows[idx].update(log_kstar=float(v), kstar=float(10.0 ** v))
    return jsonify(model=which, n=len(rows), n_ok=len(ok), rows=rows)


def _admin():
    return hmac.compare_digest(request.headers.get("X-Admin-Token", ""), ADMIN_TOKEN)


DENY = ({"error": "Developer token missing or wrong."}, 401)


def _train_job(data, name, seed):
    _job.update(status="running", message="Training started", checkpoint=name)
    try:
        trainer.run(data, CKPT / name, seed=seed, log=lambda m: _job.update(message=m))
        _job.update(status="done", message="Finished. Activate the checkpoint to use it.")
    except Exception as exc:
        shutil.rmtree(CKPT / name, ignore_errors=True)
        _job.update(status="failed", message=str(exc))


@app.post("/api/admin/train")
def admin_train():
    if not _admin():
        return jsonify(DENY[0]), 401
    if _job["status"] == "running":
        return jsonify(error="A training job is already running."), 409
    data_dir = BASE / "data"
    data_dir.mkdir(exist_ok=True)
    data = next(iter(sorted(data_dir.glob("dataset.*"))), None)
    f = request.files.get("file")
    if f and f.filename:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in (".xlsx", ".csv"):
            return jsonify(error="Upload an .xlsx or .csv dataset."), 400
        data = data_dir / f"upload_{int(time.time())}{suffix}"
        f.save(data)
    if data is None:
        return jsonify(error="No dataset available yet. Upload the .xlsx file."), 400
    try:
        seed = int(request.form.get("seed", 42))
    except ValueError:
        return jsonify(error="Seed must be a whole number."), 400
    name = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    threading.Thread(target=_train_job, args=(data, name, seed), daemon=True).start()
    return jsonify(started=name)


@app.get("/api/admin/status")
def admin_status():
    if not _admin():
        return jsonify(DENY[0]), 401
    return jsonify(_job)


@app.get("/api/admin/checkpoints")
def admin_checkpoints():
    if not _admin():
        return jsonify(DENY[0]), 401
    out = []
    for d in sorted(CKPT.glob("*"), reverse=True) if CKPT.exists() else []:
        m = (json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else None)
        if m:
            out.append({"name": d.name, "created": m["created"], "seed": m["seed"], "rows": m["rows"],
                        "grouped_cv_r2": {k: v["grouped_cv"]["r2"] for k, v in m["models"].items()}, "active": d.name == _active_name()})
    return jsonify(checkpoints=out, active=_active_name())


@app.post("/api/admin/activate")
def admin_activate():
    if not _admin():
        return jsonify(DENY[0]), 401
    name = str((request.get_json(silent=True) or {}).get("name", ""))
    src = CKPT / name
    if not re.fullmatch(r"[\w-]+", name) or not src.is_dir():
        return jsonify(error="Unknown checkpoint."), 404
    for p in list(src.glob("*.pkl")) + list(src.glob("*.json")):
        shutil.copy(p, MODEL_DIR / p.name)
    (MODEL_DIR / "active.txt").write_text(name)
    load_models()
    return jsonify(active=name, error=_state["error"])


try:  # optional: fetch large model files from Hugging Face Hub on startup (see hf_sync.py)
    import hf_sync
    hf_sync.ensure_models(MODEL_DIR)
except Exception as exc:
    print("Model sync skipped:", exc)


load_models()

if __name__ == "__main__":
    if not os.environ.get("ADMIN_TOKEN"):
        print(f"Developer token for this run: {ADMIN_TOKEN}  (set ADMIN_TOKEN to choose your own)")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
