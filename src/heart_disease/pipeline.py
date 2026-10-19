"""Leakage-safe preprocessing and classification pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)]
    )


def model_candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(random_state=42, n_jobs=1)
        ),
    }


def parameter_grids() -> dict[str, dict[str, list]]:
    return {
        "logistic_regression": {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        "random_forest": {
            "classifier__n_estimators": [100, 250],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
    }