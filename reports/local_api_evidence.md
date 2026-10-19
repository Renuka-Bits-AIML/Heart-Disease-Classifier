# Deployment Evidence

Captured 2026-10-19 using a synthetic sample payload and the trained pipeline at
`models/heart_disease_pipeline.joblib`. No real patient data was used.

| Layer | Check | Result |
|---|---|
| Docker | Image | `heart-disease-api:local`, SHA-256 `0a5909e44f0e130e01aef1639cce5362e789ec28170f7df6a2e08c6f2e753243`, 691,274,093 bytes |
| Container | Runtime | Running as `heart-disease-api-test`, host port `8001` to container port `8000` |
| Container | `GET /health` | HTTP 200, `status=ok`, `model_loaded=true` |
| Container | `POST /predict` | HTTP 200, prediction `0`, disease probability `0.1498261687` |
| Container | `GET /metrics` | HTTP 200; prediction counter and latency histogram present |
| Kubernetes | Cluster/context | kind v1.37.0, context `kind-heart-disease` |
| Kubernetes | Rollout | Deployment succeeded, 2/2 replicas ready, 0 restarts |
| Kubernetes | Service | `heart-disease-api` ClusterIP `10.96.23.185`, port 8000 |
| Kubernetes | Forwarded API | Host port `8002`; health 200, prediction 0, metrics 200 |
| Kubernetes | Request logs | Pod logs show health, prediction, and metrics requests with HTTP 200 |

Screenshots are in `screenshots/container_api_docs.png`,
`screenshots/kubernetes_api_docs.png`, `screenshots/kubernetes_health.png`, and
`screenshots/kubernetes_metrics.png`.
This demonstrates a local container and local Kubernetes deployment. It does not
establish public-cloud deployment, TLS, authentication, or production monitoring.