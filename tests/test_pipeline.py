import numpy as np
import pandas as pd

from heart_disease.data import save_cleaned_dataset
from heart_disease.pipeline import FEATURES, model_candidates


def sample_features(rows: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {
            "age": rng.integers(35, 75, rows),
            "sex": rng.integers(0, 2, rows),
            "cp": rng.integers(1, 5, rows),
            "trestbps": rng.integers(100, 180, rows),
            "chol": rng.integers(150, 350, rows),
            "fbs": rng.integers(0, 2, rows),
            "restecg": rng.integers(0, 3, rows),
            "thalach": rng.integers(90, 190, rows),
            "exang": rng.integers(0, 2, rows),
            "oldpeak": rng.random(rows) * 4,
            "slope": rng.integers(1, 4, rows),
            "ca": rng.integers(0, 4, rows),
            "thal": rng.choice([3, 6, 7], rows),
        }
    )
    return frame[FEATURES]


def test_all_pipelines_fit_and_return_binary_probabilities() -> None:
    features = sample_features()
    target = pd.Series([0, 1] * 12)

    for model in model_candidates().values():
        model.fit(features, target)
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        assert set(predictions).issubset({0, 1})
        assert probabilities.shape == (len(features), 2)
        assert np.isfinite(probabilities).all()


def test_pipeline_imputes_missing_features_and_ignores_unseen_categories() -> None:
    features = sample_features()
    target = pd.Series([0, 1] * 12)
    features.loc[0, "chol"] = np.nan
    features.loc[1, "thal"] = np.nan
    model = model_candidates()["logistic_regression"].fit(features, target)
    unseen = sample_features(1)
    unseen.loc[0, "cp"] = 99

    assert model.predict(unseen).shape == (1,)


def test_cleaned_dataset_export_binarizes_target_and_preserves_missing_features(tmp_path) -> None:
    source = tmp_path / "cleveland.data"
    source.write_text(
        "63,1,1,145,233,1,2,150,0,2.3,3,?,6,0\n"
        "45,0,4,120,198,0,0,180,0,0.0,1,0,?,2\n",
        encoding="utf-8",
    )
    destination = tmp_path / "processed" / "cleveland_cleaned.csv"

    saved_path = save_cleaned_dataset(source, destination)
    cleaned = pd.read_csv(saved_path)

    assert cleaned["target"].tolist() == [0, 1]
    assert cleaned["ca"].isna().sum() == 1
    assert cleaned["thal"].isna().sum() == 1
    assert "target_original" not in cleaned.columns