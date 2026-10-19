"""UCI Heart Disease data acquisition and validation."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_DATA_PATH = ROOT / "data" / "processed" / "cleveland_cleaned.csv"
UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target_original",
]


def download_dataset(destination: Path = RAW_DATA_PATH, force: bool = False) -> Path:
    """Download only the processed Cleveland file from UCI's official archive."""
    destination = Path(destination)
    if destination.exists() and not force:
        return destination

    with urlopen(UCI_DOWNLOAD_URL, timeout=30) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = next(
            (name for name in archive.namelist() if Path(name).name == "processed.cleveland.data"),
            None,
        )
        if member is None:
            raise ValueError("UCI archive did not contain processed.cleveland.data")
        contents = archive.read(member)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(contents)
    return destination


def load_dataset(path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load Cleveland data, convert '?' to missing values, and binarize diagnosis."""
    path = Path(path)
    if not path.exists():
        download_dataset(path)

    frame = pd.read_csv(path, header=None, names=COLUMNS, na_values=["?"])
    if frame.shape[1] != len(COLUMNS) or frame.empty:
        raise ValueError(f"Unexpected UCI data shape: {frame.shape}")

    for column in COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["target_original"].isna().any():
        raise ValueError("The target contains missing or invalid values")

    labels = (frame.pop("target_original") > 0).astype("int8").rename("target")
    return frame, labels


def save_cleaned_dataset(
    source_path: Path = RAW_DATA_PATH,
    destination: Path = CLEAN_DATA_PATH,
) -> Path:
    """Export named numeric features and a binary target, retaining missing predictors."""
    features, target = load_dataset(source_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.assign(target=target).to_csv(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download the archive again")
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--cleaned-output", type=Path, default=CLEAN_DATA_PATH)
    args = parser.parse_args()
    destination = download_dataset(args.output, force=args.force)
    print(f"Downloaded UCI Cleveland data to {destination}")
    cleaned_destination = save_cleaned_dataset(destination, args.cleaned_output)
    print(f"Saved cleaned dataset to {cleaned_destination}")


if __name__ == "__main__":
    main()