"""Cross-validate, tune, evaluate, and log heart disease classifiers."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from heart_disease.data import ROOT, load_dataset
from heart_disease.eda import generate_eda
from heart_disease.pipeline import model_candidates, parameter_grids

matplotlib.use("Agg")

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"


def summarize_scores(scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean(scores[f"test_{metric}"])),
            "std": float(np.std(scores[f"test_{metric}"])),
        }
        for metric in SCORING
    }


def log_model_run(
    name: str,
    stage: str,
    estimator,
    metrics: dict[str, dict[str, float]],
    params: dict | None = None,
) -> None:
    with mlflow.start_run(run_name=f"{name}-{stage}"):
        mlflow.log_params({"model": name, "stage": stage})
        if params:
            mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(
            {metric: values["mean"] for metric, values in metrics.items()}
        )
        mlflow.log_dict(metrics, "cross_validation_metrics.json")
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "pipeline.joblib"
            joblib.dump(estimator, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")


def evaluate_holdout(estimator, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = estimator.predict(x_test)
    probabilities = estimator.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def save_evaluation_plots(
    estimator, x_test: pd.DataFrame, y_test: pd.Series, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = estimator.predict_proba(x_test)[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, label="Selected model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Held-out ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, estimator.predict(x_test))
    plt.title("Held-out confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def train(cv_folds: int = 5, tracking_uri: str | None = None) -> dict:
    features, target = load_dataset()
    generate_eda()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    default_tracking_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or default_tracking_uri
    )
    mlflow.set_experiment("uci-heart-disease")

    candidates = model_candidates()
    grids = parameter_grids()
    comparison: dict[str, dict] = {}
    tuned_estimators = {}

    for name, estimator in candidates.items():
        cv_scores = cross_validate(
            estimator, x_train, y_train, cv=cv, scoring=SCORING, n_jobs=1
        )
        baseline_metrics = summarize_scores(cv_scores)
        fitted_baseline = clone(estimator).fit(x_train, y_train)
        log_model_run(name, "baseline", fitted_baseline, baseline_metrics)
        comparison[name] = {"baseline": baseline_metrics}

        search = GridSearchCV(
            estimator,
            grids[name],
            scoring=SCORING,
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        ).fit(x_train, y_train)
        best_index = search.best_index_
        tuned_metrics = {
            metric: {
                "mean": float(search.cv_results_[f"mean_test_{metric}"][best_index]),
                "std": float(search.cv_results_[f"std_test_{metric}"][best_index]),
            }
            for metric in SCORING
        }
        log_model_run(
            name,
            "tuned",
            search.best_estimator_,
            tuned_metrics,
            search.best_params_,
        )
        comparison[name]["tuned"] = tuned_metrics
        comparison[name]["best_params"] = search.best_params_
        tuned_estimators[name] = search.best_estimator_

    selected_name = max(
        tuned_estimators,
        key=lambda name: comparison[name]["tuned"]["roc_auc"]["mean"],
    )
    selected_for_evaluation = tuned_estimators[selected_name]
    holdout_metrics = evaluate_holdout(selected_for_evaluation, x_test, y_test)
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_plots(selected_for_evaluation, x_test, y_test, figures_dir)

    comparison_rows = [
        {
            "model": name,
            "stage": stage,
            **{metric: values["mean"] for metric, values in result.items()},
            **{f"{metric}_std": values["std"] for metric, values in result.items()},
        }
        for name, model_results in comparison.items()
        for stage in ("baseline", "tuned")
        for result in [model_results[stage]]
    ]
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(reports_dir / "model_comparison.csv", index=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=comparison_frame, x="model", y="roc_auc", hue="stage")
    plt.ylim(0, 1)
    plt.title("Cross-validation ROC-AUC")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison.png", dpi=160)
    plt.close()

    final_model = clone(selected_for_evaluation).fit(features, target)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    results = {
        "selected_model": selected_name,
        "selection_metric": "mean cross-validation ROC-AUC",
        "cross_validation": comparison,
        "holdout_metrics": holdout_metrics,
        "holdout_size": len(y_test),
        "training_rows": len(features),
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with mlflow.start_run(run_name=f"{selected_name}-final-evaluation"):
        mlflow.log_params({"model": selected_name, "stage": "final_evaluation"})
        mlflow.log_metrics({f"holdout_{key}": value for key, value in holdout_metrics.items()})
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="selected_pipeline")
        mlflow.log_artifact(str(reports_dir / "metrics.json"), artifact_path="evaluation")
        for plot_path in figures_dir.glob("*.png"):
            mlflow.log_artifact(str(plot_path), artifact_path="figures")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    print(json.dumps(train(args.cv_folds, args.tracking_uri), indent=2))


if __name__ == "__main__":
    main()