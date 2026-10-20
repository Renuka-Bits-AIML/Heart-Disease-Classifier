# Deployment Evidence

Captured 2026-10-20 using a synthetic sample payload and the trained pipeline at
`models/heart_disease_pipeline.joblib`. No real patient data was used.

| Layer | Check | Result |
|---|---|
| Docker | Image | `heart-disease-api:local`, SHA-256 `955aeeaac152b9353d568005fc399d845ec56943d6101ccbdffaeefe597f604e`, 691,290,174 bytes |
| Container | Runtime | Running as `heart-disease-api-test`, host port `8001` to container port `8000` |
| Container | `GET /health` | HTTP 200, `status=ok`, `model_loaded=true` |
| Container | `POST /predict` | HTTP 200, prediction `0`, disease probability `0.1498261687` |
| Container | `GET /metrics` | HTTP 200; prediction counter and latency histogram present |
| Kubernetes | Cluster/context | kind v1.37.0, context `kind-heart-disease` |
| Kubernetes | Rollout | Deployment succeeded, 2/2 replicas ready, 0 restarts |
| Kubernetes | Service | `heart-disease-api` ClusterIP `10.96.23.185`, port 8000 |
| Kubernetes | Forwarded API | Host port `8002`; health 200, prediction 0, metrics 200 |
| Kubernetes | Request logs | Pod logs show `api_request` lines with method, path, status, and latency |

The image was rebuilt on 2026-10-20 after verification found that the request-log
lines were missing from container and pod output, because uvicorn does not configure
the application logger. A stream handler was added, the image was rebuilt, the
container was replaced, and the kind deployment was restarted on the new image.

## Screenshots

Captured directly from a browser:

| File | Shows |
|---|---|
| `screenshots/kubernetes_swagger_predict.png` | Full Swagger `POST /predict` on the Kubernetes service: request, curl, HTTP 200 response |
| `screenshots/kubernetes_swagger_predict_summary.png` | Cropped version of the above, used in the report |
| `screenshots/container_swagger_predict.png` | Same call against the Docker container on port 8001 |
| `screenshots/container_swagger_predict_summary.png` | Cropped version of the above |
| `screenshots/kubernetes_swagger_overview.png`, `container_swagger_overview.png` | Swagger endpoint list (`GET /health`, `POST /predict`) |
| `screenshots/kubernetes_health.png`, `container_health.png` | `/health` JSON response |
| `screenshots/kubernetes_api_docs.png`, `container_api_docs.png` | Earlier Swagger page captures |

Rendered as images from the real command or HTTP output, not window captures:

| File | Source |
|---|---|
| `screenshots/docker_image_and_container.png` | `docker images` and `docker ps` |
| `screenshots/kubernetes_deployment_status.png` | `kubectl get nodes` and `get deployment,pods,service` |
| `screenshots/kubernetes_pod_logs.png` | `kubectl logs` filtered to `api_request` lines |
| `screenshots/kubernetes_metrics.png`, `container_metrics.png` | `GET /metrics` filtered to `heart_disease_*` series |
| `screenshots/mlflow_runs_table.png` | Active runs read from the MLflow SQLite tracking database |

The MLflow web interface itself could not be captured legibly, because the available
browser window was too small and headless rendering of the page did not complete. To
capture it, run `mlflow ui --backend-store-uri sqlite:///mlflow.db`, open
`http://127.0.0.1:5000/#/experiments/1/runs`, and take a normal screenshot.

This demonstrates a local container and local Kubernetes deployment. It does not
establish public-cloud deployment, TLS, authentication, or production monitoring.