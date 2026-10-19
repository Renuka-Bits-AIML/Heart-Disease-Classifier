from fastapi.testclient import TestClient
from test_pipeline import sample_features

from heart_disease.api import create_app
from heart_disease.pipeline import model_candidates


def fitted_model():
    features = sample_features()
    target = [0, 1] * 12
    return model_candidates()["logistic_regression"].fit(features, target)


def test_health_and_prediction_contract(caplog) -> None:
    app = create_app()
    app.state.model = fitted_model()
    client = TestClient(app)
    patient = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }

    with caplog.at_level("INFO", logger="heart_disease.api"):
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert "api_request method=GET path=/health status_code=200" in caplog.text
    response = client.post("/predict", json=patient)
    assert response.status_code == 200
    assert response.json()["prediction"] in {0, 1}
    assert 0 <= response.json()["disease_probability"] <= 1
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "heart_disease_predictions_total" in metrics.text


def test_predict_rejects_invalid_features() -> None:
    app = create_app()
    app.state.model = fitted_model()
    response = TestClient(app).post("/predict", json={"age": 140})

    assert response.status_code == 422