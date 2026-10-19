"""Dataset profiling and exploratory visualizations."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from heart_disease.data import ROOT, load_dataset
from heart_disease.pipeline import NUMERIC_FEATURES

matplotlib.use("Agg")


def generate_eda(output_dir: Path = ROOT / "reports" / "figures") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features, target = load_dataset()
    frame = features.assign(target=target)

    profile = {
        "rows": int(frame.shape[0]),
        "features": int(features.shape[1]),
        "missing_values": {name: int(count) for name, count in frame.isna().sum().items()},
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "target_distribution": {
            str(label): int(count) for label, count in target.value_counts().sort_index().items()
        },
    }
    (output_dir.parent / "data_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )

    frame[NUMERIC_FEATURES].hist(figsize=(12, 8), bins=16, edgecolor="white")
    plt.tight_layout()
    plt.savefig(output_dir / "numeric_histograms.png", dpi=160)
    plt.close("all")

    plt.figure(figsize=(11, 8))
    sns.heatmap(frame.corr(numeric_only=True), cmap="vlag", center=0, annot=False)
    plt.title("Cleveland feature and target correlations")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close("all")

    counts = target.value_counts().reindex([0, 1], fill_value=0)
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=["No disease", "Disease"],
        y=counts.values,
        hue=["No disease", "Disease"],
        legend=False,
    )
    plt.ylabel("Patient count")
    plt.title("Binary target distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=160)
    plt.close("all")
    return profile


if __name__ == "__main__":
    print(json.dumps(generate_eda(), indent=2))