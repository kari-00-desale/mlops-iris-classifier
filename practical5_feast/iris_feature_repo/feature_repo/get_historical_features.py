"""Retrieve point-in-time-correct historical features for model training."""

from pathlib import Path

import pandas as pd
from feast import FeatureStore

from features import IRIS_FEATURE_REFERENCES

REPO_PATH = Path(__file__).resolve().parent
SOURCE_PATH = REPO_PATH / "data" / "iris_features.parquet"


def get_historical_features() -> pd.DataFrame:
    """Retrieve every source row and verify complete, point-in-time feature values."""
    source_df = pd.read_parquet(SOURCE_PATH)
    entity_df = source_df[["sample_id", "event_timestamp"]].copy()
    store = FeatureStore(repo_path=str(REPO_PATH))
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=IRIS_FEATURE_REFERENCES,
    ).to_df()

    expected_columns = [reference.split(":", maxsplit=1)[1] for reference in IRIS_FEATURE_REFERENCES]
    missing_columns = set(expected_columns) - set(training_df.columns)
    if missing_columns:
        raise RuntimeError(f"Historical response is missing feature columns: {sorted(missing_columns)}")
    if len(training_df) != len(source_df):
        raise RuntimeError(
            f"Expected {len(source_df)} historical rows, received {len(training_df)}."
        )
    if training_df[expected_columns].isna().any().any():
        raise RuntimeError("Historical response contains null feature values.")
    return training_df


if __name__ == "__main__":
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Feature source not found; run prepare_feature_source.py first: {SOURCE_PATH}")

    features_df = get_historical_features()
    print("FEAST HISTORICAL FEATURE RETRIEVAL TEST")
    print(features_df.head().to_string(index=False))
    print(f"\nNumber of rows retrieved: {len(features_df)}")
    print("All registered feature columns are populated.")
