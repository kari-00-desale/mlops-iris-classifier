"""Convert Experiment 4's feature CSV into a timestamped Feast Parquet source."""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = r"C:\Users\HP\mlops-iris-classifier\data\processed\iris_features.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "iris_features.parquet"
REQUIRED_COLUMNS = {
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "species",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
    "petal_length_bin",
}


def prepare_feature_source(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Add stable entity IDs and UTC event timestamps, then write Parquet."""
    df = pd.read_csv(input_path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {sorted(missing_columns)}")
    if df.empty:
        raise ValueError(f"Input contains no feature rows: {input_path}")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(f"Input contains null feature values: {input_path}")

    df.insert(0, "sample_id", range(len(df)))
    end_time = pd.Timestamp.now(tz="UTC").floor("s")
    df["event_timestamp"] = pd.date_range(
        end=end_time,
        periods=len(df),
        freq="min",
    )
    df["created_timestamp"] = df["event_timestamp"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    prepare_feature_source(arguments.input, arguments.output)
