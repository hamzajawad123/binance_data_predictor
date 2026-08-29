# binance_data_predictor

> Hourly **24-hour-ahead realized volatility** forecasts for **BTCUSDT, ETHUSDT, BNBUSDT, and SOLUSDT**.

[![MLOps Pipeline](https://github.com/hamzajawad123/binance_data_predictor/actions/workflows/mlops_pipeline.yml/badge.svg)](https://github.com/hamzajawad123/binance_data_predictor/actions/workflows/mlops_pipeline.yml)

This repository is a local-first forecasting stack: public Binance hourly candles → leakage-safe features → LightGBM / XGBoost (MLflow) → **FastAPI** + **Streamlit**.

It forecasts **how jumpy the next 24 hours may be**. It does **not** predict whether price will go up or down. It is **not** financial advice.

A committed `cloud/` bundle (about 90 days of data plus `model.joblib`) lets you run the dashboard after a clone, without fetching the full history first.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies used](#technologies-used)
- [Project architecture](#project-architecture)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Environment configuration](#environment-configuration)
- [Running the API](#running-the-api)
- [Running the frontend](#running-the-frontend)
- [Run the complete application](#run-the-complete-application)
- [How to stop](#how-to-stop)
- [Optional: fetch data and train locally](#optional-fetch-data-and-train-locally)
- [Docker](#docker)
- [GitHub Actions](#github-actions)
- [API endpoints](#api-endpoints)
- [Project structure](#project-structure)
- [Folder and file guide](#folder-and-file-guide)
- [Common commands](#common-commands)
- [Troubleshooting](#troubleshooting)
- [Useful links](#useful-links)
- [Contributing](#contributing)
- [Security](#security)

---

## Introduction

**Package name (pyproject.toml):** `crypto-vol-forecast` `0.1.0`

**What this project does**

At the close of hour `t` the model forecasts:

- `log_return_t = ln(close_t / close_{t-1})`
- `vol_24h = std(log_return_{t+1}, …, log_return_{t+24})`

The dashboard also shows a 1-day equivalent: `vol_24h * sqrt(24)` (not annualized).

**Why it exists**

Hourly crypto moves are noisy. This stack produces a **magnitude** (jumpiness) forecast with a Streamlit UI, a FastAPI contract, walk-forward evaluation artifacts, and a Cloud demo bundle.

**Who it is for**

This information is not available in the current repository as a named user persona. Practically: developers who clone the repo, and anyone you share a Streamlit Cloud URL with.

**What it does not do**

Crash-risk / extreme-drop classification is not in this serving stack.

---

## Features

- Public Binance klines (no Binance API key in `.env.example`)
- Four symbols: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT
- Leakage-safe features (no future bars in inputs at `t`)
- Serving model: `models/model.joblib` locally, or `cloud/model.joblib` after clone
- FastAPI: health, features, candles, predict, score
- Streamlit dashboard: coin nav, forecast metrics, candles, jumpiness charts
- If FastAPI is down, the dashboard loads the model **in-process** (this is how Streamlit Cloud runs)
- Optional Feast online store (SQLite); API falls back to parquet if Feast is empty or fails
- GitHub Actions: tests, 6-hour `cloud/` refresh, Docker Hub image push
- TreeSHAP write-up: [docs/SHAP.md](docs/SHAP.md)

---

## Technologies used

Verified from `requirements.txt`, `requirements-cloud.txt`, `Dockerfile`, and `pyproject.toml`:

| Area | Technology |
| --- | --- |
| Language | Python `>=3.10` (`pyproject.toml`). CI, Docker, and `runtime.txt` use **3.11**. |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Data | pandas, NumPy, PyArrow, requests |
| Models | scikit-learn, LightGBM, XGBoost, joblib |
| Tracking | MLflow (`mlflow.db` / `mlruns/`, gitignored) |
| Features store | Feast (offline parquet + SQLite online store) |
| Explainability | SHAP |
| Tests | pytest |
| Containers | Docker, Docker Compose |
| Pipeline hashes | DVC (`dvc.yaml`, `dvc.lock`) — the `dvc` CLI is **not** listed in `requirements.txt` |
| CI | GitHub Actions (`.github/workflows/mlops_pipeline.yml`) |

There is **no** Node.js frontend, **no** npm, and **no** PostgreSQL / MySQL in this repository. Feast’s online store is SQLite at `data/online_store.db` (gitignored).

---

## Project architecture

```text
Binance public klines API
        │
        ▼
 data/raw.parquet          (full history; gitignored)
        │
        ▼
 data/features.parquet     (gitignored)  ──optional──► Feast SQLite
        │
        ▼
 models/model.joblib       (gitignored) + MLflow
        │
        ├── FastAPI  (uvicorn, port 8000)
        └── Streamlit (frontend/dashboard.py)
                │
                └── if API is down: in-process load of cloud/ or data/ + models/

 cloud/   committed 90-day snapshot + model (used by Streamlit Cloud and as fallback)
```

---

## Prerequisites

Install these **before** cloning if you want a local run.

### Git (mandatory)

- Check: `git --version`
- Install: [https://git-scm.com/downloads](https://git-scm.com/downloads)

### Python 3.10+ (mandatory); 3.11 recommended

- Check: `python --version` (Windows) or `python3 --version` (macOS / Linux)
- CI and Docker use **3.11**. `runtime.txt` contains `python-3.11` (Streamlit Cloud).

### VS Code (optional)

- [https://code.visualstudio.com/](https://code.visualstudio.com/)

### Docker Desktop (optional)

Needed only for `docker compose` / image build.

- Check: `docker --version` and `docker compose version`

### Network

Data collection calls `BINANCE_BASE_URL` (default `https://api.binance.com`). No Binance API key is defined in `.env.example`.

---

## Quick start

Fastest path after clone: use the committed `cloud/` bundle (no 18-month download).

Run these from the folder where you want the project (PowerShell, Command Prompt, or VS Code terminal).

```bash
git clone https://github.com/hamzajawad123/binance_data_predictor.git
cd binance_data_predictor
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
copy .env.example .env
pip install -r requirements-cloud.txt
streamlit run frontend/dashboard.py
```

macOS / Linux:

```bash
source .venv/bin/activate
cp .env.example .env
pip install -r requirements-cloud.txt
streamlit run frontend/dashboard.py
```

Open the URL Streamlit prints in the terminal (Docker Compose publishes the UI on **8501**).

For the full local stack (Feast, MLflow, SHAP, pytest), install `requirements.txt` instead of `requirements-cloud.txt` and start FastAPI in a second terminal (see [Run the complete application](#run-the-complete-application)).

---

## Installation

### Step 1 — Install required software

Install Git and Python 3.10+ (3.11 recommended). Install Docker Desktop only if you will use Compose.

### Step 2 — Clone the repository

Open PowerShell, Command Prompt, or the VS Code terminal. `cd` to the parent folder where the project should live, then:

```bash
git clone https://github.com/hamzajawad123/binance_data_predictor.git
cd binance_data_predictor
```

You must run later commands from this repository root (`binance_data_predictor/`), unless a step says otherwise.

### Step 3 — Open the project in VS Code

1. Start VS Code.
2. **File → Open Folder…** and select `binance_data_predictor`.
3. Confirm the explorer shows `frontend/`, `backend/`, and `README.md`.
4. **Terminal → New Terminal** (or `` Ctrl+` ``).

If the VS Code `code` CLI is installed:

```bash
code .
```

### Step 4 — Create a virtual environment and install dependencies

From the repository root:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

| File | Use |
| --- | --- |
| `requirements.txt` | Local API, Feast, MLflow, SHAP, pytest |
| `requirements-cloud.txt` | Streamlit Cloud / lighter dashboard-only install |

`packages.txt` lists `libgomp1` (used by Docker/Streamlit Cloud native deps). `pyproject.toml` names the package `crypto-vol-forecast` and points pytest at `tests/`.

### Step 5 — Configure environment variables

See [Environment configuration](#environment-configuration).

There is **no** database server to install for the default parquet path. Feast’s SQLite files are created under `data/` when you materialize Feast.

---

## Environment configuration

1. Copy the example file **in the repository root** (same folder as `README.md`):

Windows:

```bash
copy .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

2. Edit `.env` if you need non-default paths or Docker Hub username.

3. **Never commit `.env`.** It is listed in `.gitignore`.

Variables from [`.env.example`](.env.example):

| Variable | Role |
| --- | --- |
| `SYMBOLS` | Comma-separated pairs (default `BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT`) |
| `INTERVAL` | Kline interval (default `1h`) |
| `LOOKBACK_DAYS` | Default collect lookback (default `540`) |
| `BINANCE_BASE_URL` | Kline API base (default `https://api.binance.com`) |
| `RAW_PATH` | Raw OHLCV parquet (default `data/raw.parquet`) |
| `FEATURES_PATH` | Feature parquet (default `data/features.parquet`) |
| `MODELS_DIR` | Serving artifacts directory (default `models`) |
| `MLFLOW_TRACKING_URI` | Default `sqlite:///mlflow.db` |
| `MLFLOW_EXPERIMENT` | Default `crypto-vol-24h` |
| `MODEL_NAME` | Default `crypto_vol_24h` |
| `HOLDOUT_DAYS` | Train/eval holdout (default `60`) |
| `VOL_ALERT_PERCENTILE` | Alert line percentile (default `90`) |
| `DOCKER_HUB_USERNAME` | Compose image prefix (placeholder `yourusername`) |
| `IMAGE_NAME` | Example `crypto-ops-fastapi` (workflow image name is `crypto-ops-fastapi:latest`) |
| `API_HOST` | Example `0.0.0.0` |
| `API_PORT` | Compose host port mapping default `8000` |
| `API_URL` | Dashboard → API (default `http://localhost:8000` in `frontend/dashboard.py` if unset) |

Optional, **not** in `.env.example`, but read by `frontend/dashboard.py`:

| Variable | Role |
| --- | --- |
| `VOL_SERVE` | `inprocess` / `local` / `cloud` forces in-process serving; `api` / `http` forces HTTP |

If `data/raw.parquet` or `data/features.parquet` is missing, `backend/src/__init__.py` falls back to `cloud/raw.parquet` and `cloud/features.parquet`. If `models/model.joblib` is missing, it falls back to `cloud/model.joblib`.

---

## Running the API

The API **is** the backend: `backend/app/main.py` (FastAPI app `Crypto Volatility Forecast API`, version `0.4.0`).

**Terminal 1** — repository root, venv activated, `requirements.txt` installed, `.env` present:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Docker image CMD uses `--host 0.0.0.0 --port 8000`.

| Check | URL |
| --- | --- |
| Health | http://localhost:8000/health |
| Interactive docs (FastAPI default) | http://localhost:8000/docs |
| OpenAPI JSON (FastAPI default) | http://localhost:8000/openapi.json |

Expected health field when running: `"status": "ok"` (see `GET /health` in `backend/app/main.py`).

---

## Running the frontend

**Terminal 2** — repository root, venv activated:

```bash
streamlit run frontend/dashboard.py
```

Docker Compose publishes the UI on **8501**.

The dashboard calls `API_URL` (default `http://localhost:8000`). If `/health` is unreachable (or `VOL_SERVE` is `inprocess` / `local` / `cloud`), it imports FastAPI handlers in-process.

---

## Run the complete application

Keep both processes running.

### Terminal 1 — API

```bash
uvicorn backend.app.main:app --reload --port 8000
```

### Terminal 2 — dashboard

```bash
streamlit run frontend/dashboard.py
```

### Browser

- Dashboard: URL printed by Streamlit (Compose uses http://localhost:8501 )
- API health: http://localhost:8000/health

You can run **only** Terminal 2 if `cloud/` (or local `data/` + `models/`) is present; the UI will serve in-process.

---

## How to stop

- API and Streamlit: focus that terminal and press **Ctrl+C**.
- Docker Compose:

```bash
docker compose down
```

---

## Optional: fetch data and train locally

Full history parquets are **gitignored**. After clone you already have `cloud/`. Use these only if you want a longer local dataset or to retrain.

### Collect candles

From the repository root:

```bash
python -m backend.src.data_collection
```

Incremental:

```bash
python -m backend.src.data_collection --incremental
```

Other flags (from `backend/src/data_collection.py`): `--full-history`, `--days`, `--symbols`, `--output`.

### Features

EDA notebook: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb). Then:

```bash
python -m backend.src.eda_feature_eng
```

### Feast (optional)

```bash
python -m backend.feature_repo.features
```

This materializes the SQLite online store. If it fails (reported as fragile on native Windows in prior project notes), the API still reads parquet.

### Train / evaluate / explain

Local train module (LightGBM + XGBoost, MLflow, writes `models/model.joblib`):

```bash
python -m backend.src.train
python -m backend.src.evaluate
python -m backend.src.explain
```

Colab-oriented notebook: [notebooks/02_train.ipynb](notebooks/02_train.ipynb).

Walk-forward script:

```bash
python -m backend.src.walkforward
```

MLflow UI (tracking URI from `.env.example`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Refresh the committed cloud bundle

```bash
python -m backend.src.export_cloud --days 90
```

### DVC

`dvc.yaml` stages: `collect` → `features` → `train`. Outputs are gitignored; hashes are in `dvc.lock`. The `dvc` CLI is not in `requirements.txt`. If you install DVC yourself, use its docs for `dvc repro` / `dvc pull`. Do not treat `dvc init` as required for a dashboard-only clone.

---

## Docker

From the repository root (`.env` should exist; Compose uses `env_file: .env`):

```bash
docker compose up --build
```

| Service | Published port (Compose) |
| --- | --- |
| `api` (uvicorn) | `${API_PORT:-8000}` → http://localhost:8000 |
| `streamlit` | `8501` → http://localhost:8501 |

The Streamlit service sets `API_URL=http://api:8000`.

Image name in Compose: `${DOCKER_HUB_USERNAME:-yourusername}/crypto-ops-fastapi:latest`.

`Dockerfile` is `python:3.11-slim`, installs `libgomp1`, copies `backend/`, `frontend/`, `cloud/`, exposes **8000**, CMD uvicorn on `0.0.0.0:8000`.

Local build/push (replace the username with your Docker Hub user):

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/crypto-ops-fastapi:latest .
docker push YOUR_DOCKERHUB_USERNAME/crypto-ops-fastapi:latest
```

---

## GitHub Actions

Workflow file: [`.github/workflows/mlops_pipeline.yml`](.github/workflows/mlops_pipeline.yml)

| Trigger | Job |
| --- | --- |
| Push / pull request (not the 6h cron) | `tests`: `pip install -r requirements.txt` then `pytest -q` |
| Cron `0 */6 * * *` (UTC) or **Run workflow** | `refresh_cloud`: incremental Binance fetch from `cloud/`, rebuild features, `export_cloud --days 90`, commit `cloud/` |
| Push to `main` (after tests) | `docker`: build and push `DOCKERHUB_USERNAME/crypto-ops-fastapi:latest` if secrets exist |

Repository secrets used by the workflow:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Runs: [https://github.com/hamzajawad123/binance_data_predictor/actions](https://github.com/hamzajawad123/binance_data_predictor/actions)

---

## API endpoints

Implemented in `backend/app/main.py`. Symbols must be one of `SYMBOLS`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Status, symbols, whether the model and parquets exist |
| GET | `/model-info` | Serving artifact fingerprint and interval metadata |
| GET | `/features?symbol=BTCUSDT` | Latest feature snapshot (Feast or parquet) |
| GET | `/candles?symbol=BTCUSDT&limit=168` | Hourly OHLC for charts (`limit` 24–2000, default 168) |
| POST | `/predict` | 24h vol forecast for one symbol |
| GET | `/predict/all` | Forecasts for all configured symbols |
| GET | `/score?symbol=BTCUSDT&limit=168` | Predicted vs settled actuals and residual band |

POST `/predict` body:

```json
{"symbol": "ETHUSDT"}
```

FastAPI interactive docs: http://localhost:8000/docs (when the API is running locally).

There is no authentication middleware on these routes in `backend/app/main.py` (CORS is `allow_origins=["*"]`).

---

## Project structure

```text
binance_data_predictor/
├── .github/workflows/mlops_pipeline.yml
├── .streamlit/config.toml
├── backend/
│   ├── app/main.py              # FastAPI
│   ├── feature_repo/            # Feast store + feature view
│   └── src/                     # collect, features, train, eval, SHAP, export
├── cloud/                       # 90-day demo bundle + model.joblib
├── data/                        # raw/features parquets (binaries gitignored)
├── docs/
│   ├── SHAP.md
│   └── shap_bar.png
├── frontend/dashboard.py        # Streamlit UI
├── models/                      # local serving bundle (*.joblib gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_train.ipynb
├── tests/
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── packages.txt
├── pyproject.toml
├── requirements.txt
├── requirements-cloud.txt
├── runtime.txt
├── .env.example
└── README.md
```

`docs/PLAN.md` and `docs/EVALUATION.md` are listed in `.gitignore` and are **not** part of the GitHub tree.

---

## Folder and file guide

| Path | Purpose |
| --- | --- |
| `frontend/dashboard.py` | Streamlit dashboard |
| `backend/app/main.py` | FastAPI routes |
| `backend/src/data_collection.py` | Binance hourly download |
| `backend/src/eda_feature_eng.py` | Feature parquet |
| `backend/src/train.py` | Local LightGBM/XGBoost train + MLflow |
| `backend/src/evaluate.py` | Holdout metrics |
| `backend/src/walkforward.py` | Walk-forward vs baselines |
| `backend/src/explain.py` | TreeSHAP |
| `backend/src/export_cloud.py` | Write `cloud/` |
| `backend/src/shadow.py` | Freshness, residual band, scoring helpers |
| `backend/feature_repo/` | Feast entity/view + SQLite online store config |
| `cloud/` | Committed demo data + model |
| `tests/` | pytest (`test_api.py`, `test_pipeline.py`, `test_validate.py`) |
| `notebooks/` | Colab EDA and training notebooks |
| `.env.example` | Environment template |
| `Dockerfile` / `docker-compose.yml` | Container build and two-service Compose |

---

## Common commands

| Command | What it does |
| --- | --- |
| `pip install -r requirements.txt` | Full local dependencies |
| `pip install -r requirements-cloud.txt` | Dashboard / cloud-style dependencies |
| `python -m backend.src.data_collection` | Download hourly klines |
| `python -m backend.src.data_collection --incremental` | Fetch only new hours |
| `python -m backend.src.eda_feature_eng` | Build feature parquet |
| `python -m backend.feature_repo.features` | Feast materialize |
| `python -m backend.src.train` | Train and write `models/model.joblib` |
| `python -m backend.src.evaluate` | Evaluate saved model |
| `python -m backend.src.explain` | TreeSHAP |
| `python -m backend.src.walkforward` | Walk-forward evaluation |
| `python -m backend.src.export_cloud --days 90` | Refresh `cloud/` |
| `uvicorn backend.app.main:app --reload --port 8000` | Start API |
| `streamlit run frontend/dashboard.py` | Start dashboard |
| `mlflow ui --backend-store-uri sqlite:///mlflow.db` | MLflow UI |
| `pytest -q` | Tests (`pyproject.toml` `addopts = -q`) |
| `docker compose up --build` | API + Streamlit containers |

---

## Troubleshooting

**Dashboard loads but numbers look old**  
The committed `cloud/` snapshot is not a live tick feed. GitHub Actions refreshes `cloud/` on a 6-hour UTC cron when that job is enabled. Streamlit Community Cloud also sleeps when idle if you use that host.

**`Could not load a forecast` / missing parquet**  
Need `cloud/` in the clone, or run collection + feature engineering, or copy parquets into `data/`. Health field `features_exist` reflects `FEATURES_PATH`.

**API returns 503 about missing parquet**  
`GET /features` and related routes require a feature file at `FEATURES_PATH` (or the cloud fallback path if that is what `__init__.py` resolved).

**Port 8000 already in use**  
Stop the other process or change `--port` and set `API_URL` on the dashboard to match.

**Feast / Windows**  
The API tries Feast then falls back to parquet (`get_feature_snapshot` in `backend/app/main.py`). You can skip Feast for local dashboard use.

**pytest / pip fails on Feast or LightGBM**  
Use Python 3.11 to match CI. Full tests need `requirements.txt` (includes `feast`, `mlflow`, `shap`).

**Docker Hub job skipped**  
The workflow skips image push if `DOCKERHUB_USERNAME` or `DOCKERHUB_TOKEN` is empty.

**Node 20 deprecation annotation on Actions**  
Warning from `actions/checkout@v4` / `actions/setup-python@v5`. It is not a test failure by itself.

**Dependency install errors (general)**  
Upgrade pip (`python -m pip install --upgrade pip`) and retry inside an activated `.venv`.

---

## Useful links

| Resource | URL |
| --- | --- |
| GitHub repository | [https://github.com/hamzajawad123/binance_data_predictor](https://github.com/hamzajawad123/binance_data_predictor) |
| Issue tracker | [https://github.com/hamzajawad123/binance_data_predictor/issues](https://github.com/hamzajawad123/binance_data_predictor/issues) |
| Actions | [https://github.com/hamzajawad123/binance_data_predictor/actions](https://github.com/hamzajawad123/binance_data_predictor/actions) |
| TreeSHAP notes | [docs/SHAP.md](docs/SHAP.md) |
| Cloud bundle notes | [cloud/README.md](cloud/README.md) |
| Env template | [.env.example](.env.example) |
| Streamlit Cloud | [https://share.streamlit.io](https://share.streamlit.io) (app file `frontend/dashboard.py`, Python `3.11`, prefer `requirements-cloud.txt`) |
| Binance klines (public) | Default base `https://api.binance.com` |

---

## Contributing

No `CONTRIBUTING.md` is in this repository. Suggested workflow:

1. Fork or create a branch from `main`.
2. Use a virtual environment and `pip install -r requirements.txt`.
3. Run `pytest -q`.
4. Commit (do not add `.env`, `data/*.parquet`, or `models/*.joblib`).
5. Push and open a pull request.

---

## Security

Never commit API keys, passwords, tokens, or `.env` to GitHub.

This project’s default Binance path uses the **public** klines URL from `.env.example` (no key variable). Docker Hub tokens belong in GitHub **repository secrets**, not in the repo.
