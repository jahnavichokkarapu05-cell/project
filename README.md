# k* predictor

Web front end for the perovskite oxygen-exchange model from `ps_1.ipynb`.

## Run

    python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python train.py path/to/dataset.xlsx     # trains SVR + Random Forest, writes models/
    python app.py

Open http://127.0.0.1:5000

Install the same scikit-learn version you trained with, otherwise the pickle may fail to load.

## Enter a material name instead of descriptors

The predict page opens on **By material name**: type a formula such as `La0.6Sr0.4Co0.2Fe0.8O3` and a temperature in degrees C.
`train.py` recovers per-element values from your dataset (every site descriptor there is an exact weighted average)
and trains a second pair of models on the 15 composition-only inputs. Limits: only ABO3 formulas, A-site elements
first, and only elements that occur in the training data. The seven computed properties (Gap, Oxide_vacancy,
E_above_hull, TEC, Work_Function, Log_dieletric, O_pband) are not used in this mode. **By descriptors** still uses all 23.

## Two modes

- **Normal user (sign-in page)**: form, predicted k*, position on a slower-to-faster scale, temperature curve.
- **Developer (sign-in page)**: the same, plus the request JSON, DataFrame build, scaling (z-scores), model
  details, output, and temperature sweep, each with its code. Model selector (SVR / Random Forest).

The choice is remembered in the browser; switch any time from the header.

## Things to adjust in app.py

- `LOGK_SCALE`: set automatically from your data by `train.py` (via `models/meta.json`).
- `HINTS`: short descriptions under each input. Edit to match your dataset's definitions.
- `SWEEP_C`: temperature range of the sweep chart (use a range your training data covers).
- k* is computed as 10**LogKstar, which assumes a base-10 log.

## Layout

    train.py               trains the models from the dataset
    app.py                 Flask API: /api/schema, /api/predict
    static/index.html      the whole front end
    models/                put the .pkl files here


## Operations and deployment

- `python dataset_stats.py data.xlsx` checks the dataset (row counts, duplicates, bad values) and writes the dashboard numbers.
- `python train.py data.xlsx` splits by material (70/15/15, fixed seed, checked for zero overlap), saves the standardised `.npy` arrays and `splits.json` under `models/`, and trains four models.
- API: `GET /api/health`, `GET /api/analytics`, `POST /api/upload` (CSV/XLSX), `POST /api/predict`, `POST /api/predict_composition`, and developer-only `POST /api/admin/train`, `GET /api/admin/checkpoints`, `POST /api/admin/activate`.
- Developer routes need the `X-Admin-Token` header. Set `ADMIN_TOKEN` in the environment (see `.env.example`); if it is unset, a random token is printed when the app starts.
- CORS: set `CORS_ORIGINS` to the frontend address(es), comma separated. When the pages are hosted apart from the API, set `window.API_BASE` in `static/config.js`.
- Hosting: `Dockerfile` and `render.yaml` are provided for a Render web service. `hf_sync.py` downloads model files from a Hugging Face repo when `HF_REPO` is set. These have not been run against a live account.
