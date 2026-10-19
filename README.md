# Heart Disease MLOps

Reproducible binary classification using the processed Cleveland subset of the UCI
Heart Disease dataset. The API and evaluation artifacts are educational examples,
not clinical decision support; the model must not be used to diagnose or treat patients.

## Setup

Use Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For stable numerical-library startup in PowerShell, set these once per terminal
session before tests or training:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
```

The source, citation, DOI, license, and target transformation are documented in
[data/SOURCE.md](data/SOURCE.md). Download the raw Cleveland file explicitly with:

```powershell
python -m heart_disease.data
```

This writes `data/raw/processed.cleveland.data` and
`data/processed/cleveland_cleaned.csv`. The cleaned CSV retains missing predictor
values for in-pipeline imputation and does not include the original ordinal target.

## EDA and training

Run the full reproducible workflow:

```powershell
python -m heart_disease.train --cv-folds 5
```

It inspects data types, missing values, and class distribution; writes histograms,
a correlation heatmap, class-balance chart, held-out ROC and confusion matrix, and
cross-validation comparison plots under `reports/`. Logistic Regression and Random
Forest are evaluated and tuned with stratified cross-validation. Each baseline and
tuned run logs parameters, metrics, a model, and a metrics artifact to local MLflow
tracking (`mlflow.db` with artifacts under `mlartifacts/`). The model with the best tuned mean cross-validation ROC-AUC is
selected. A stratified held-out test set is used for final evaluation; the selected
pipeline is then refit on all labeled rows and saved to
`models/heart_disease_pipeline.joblib`.

To view the runs, start `mlflow ui --backend-store-uri sqlite:///mlflow.db` and open the URL
printed by MLflow. Override the tracking destination with `MLFLOW_TRACKING_URI`.

## Tests and lint

```powershell
ruff check .
pytest -q
```

GitHub Actions runs linting, unit/API tests, and the complete training workflow on
pushes and pull requests. Training artifacts are uploaded with the workflow run.

## Serving

Train first, then start the API:

```powershell
uvicorn heart_disease.api:app --host 127.0.0.1 --port 8000
```

`GET /health` reports readiness, `POST /predict` accepts the 13 feature fields and
returns a binary prediction with its estimated disease probability, and `GET /metrics`
exposes Prometheus request counts and latency. An example payload and interactive
schema are available at `/docs`.

## Docker and Kubernetes

Build and exercise the container locally (Docker must be running):

```powershell
docker build -t heart-disease-api:local .
docker run --rm -p 8001:8000 heart-disease-api:local
Invoke-RestMethod http://127.0.0.1:8001/health
```

The image was built and exercised locally. To reproduce the local kind deployment,
install Docker Desktop, kubectl, and kind, then run:

```powershell
kind create cluster --name heart-disease
kind load docker-image heart-disease-api:local --name heart-disease
kubectl --context kind-heart-disease apply -f deployment/kubernetes.yaml
kubectl --context kind-heart-disease rollout status deployment/heart-disease-api
kubectl --context kind-heart-disease port-forward service/heart-disease-api 8002:8000
```

The successful container and kind checks, model response, request logs, image digest,
and screenshots are recorded in [reports/local_api_evidence.md](reports/local_api_evidence.md)
and the `screenshots/` folder. For a public cloud, publish the image to the chosen
registry and replace the manifest image name. No cloud account or credentials were
provided, so public-cloud deployment remains outstanding. Do not commit credentials;
configure registry access and cluster secrets through the provider's secret manager.

## Report and demonstration

The report in [reports/report.md](reports/report.md) records the method, architecture,
evaluation protocol, and deployment limitations. Its numerical results should be
reproducibly generated from `reports/metrics.json` after training. Install the optional
report dependency with `python -m pip install -e ".[report]"`, then run
`python reports/build_report.py` to create the ten-page `reports/report.docx` with
hard page boundaries, the architecture diagram, and generated figures. A short
demonstration can show data download, training/MLflow, API health and prediction, then
metrics; this repository includes no recording file because video capture requires a
local recorder. The current local deployment can be recorded with Windows Game Bar
(`Win+Alt+R`) while showing the evidence commands and `/docs` at
`http://127.0.0.1:8002/docs`.