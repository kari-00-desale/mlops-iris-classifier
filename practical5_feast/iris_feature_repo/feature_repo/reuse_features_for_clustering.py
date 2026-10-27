"""Cluster Iris samples using the registered FeatureService without re-engineering."""

from pathlib import Path

import pandas as pd
from feast import FeatureStore
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from features import iris_feature_service

REPO_PATH = Path(__file__).resolve().parent
SOURCE_PATH = REPO_PATH / "data" / "iris_features.parquet"
SERVICE_NAME = iris_feature_service.name


def cluster_samples() -> pd.Series:
    """Fetch the shared online feature service and cluster all available samples."""
    source_df = pd.read_parquet(SOURCE_PATH)
    entity_rows = [{"sample_id": int(sample_id)} for sample_id in source_df["sample_id"]]
    store = FeatureStore(repo_path=str(REPO_PATH))
    service = store.get_feature_service(SERVICE_NAME)
    feature_df = pd.DataFrame(
        store.get_online_features(
            features=service,
            entity_rows=entity_rows,
        ).to_dict()
    )

    if feature_df.isna().any().any():
        raise RuntimeError("FeatureService returned null values; materialize the online store first.")
    feature_df = pd.get_dummies(
        feature_df.drop(columns=["sample_id"]),
        columns=["petal_length_bin"],
        dtype=float,
    )
    if len(feature_df) < 3:
        raise ValueError("Clustering requires at least three samples.")

    scaled_features = StandardScaler().fit_transform(feature_df)
    labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled_features)
    cluster_counts = pd.Series(labels, name="cluster").value_counts().sort_index()
    if len(cluster_counts) != 3:
        raise RuntimeError(f"Expected three non-empty clusters; got {cluster_counts.to_dict()}.")
    return cluster_counts


if __name__ == "__main__":
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Feature source not found; run prepare_feature_source.py first: {SOURCE_PATH}")

    print(f"Clustering with shared Feast FeatureService: {SERVICE_NAME}")
    print(cluster_samples().to_string())
