"""Retrieve the latest registered Iris features for sample IDs."""

import argparse
from pathlib import Path

import pandas as pd
from feast import FeatureStore

from features import IRIS_FEATURE_REFERENCES

REPO_PATH = Path(__file__).resolve().parent
SOURCE_PATH = REPO_PATH / "data" / "iris_features.parquet"


def get_online_features(sample_ids: list[int]) -> pd.DataFrame:
    """Return online feature values and fail if any requested values are absent."""
    if not sample_ids:
        raise ValueError("Provide at least one sample ID.")

    store = FeatureStore(repo_path=str(REPO_PATH))
    result = store.get_online_features(
        features=IRIS_FEATURE_REFERENCES,
        entity_rows=[{"sample_id": sample_id} for sample_id in sample_ids],
    )
    feature_df = pd.DataFrame(result.to_dict())
    feature_columns = [reference.split(":", maxsplit=1)[1] for reference in IRIS_FEATURE_REFERENCES]
    missing_columns = set(feature_columns) - set(feature_df.columns)
    if missing_columns:
        raise RuntimeError(f"Online response is missing feature columns: {sorted(missing_columns)}")
    if feature_df[feature_columns].isna().any().any():
        raise RuntimeError("Online response contains null feature values.")
    return feature_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ids", nargs="+", type=int, default=[0, 1, 2])
    arguments = parser.parse_args()

    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Feature source not found; run prepare_feature_source.py first: {SOURCE_PATH}")

    features_df = get_online_features(arguments.sample_ids)
    print("FEAST ONLINE FEATURE RETRIEVAL TEST")
    print(features_df.to_string(index=False))
