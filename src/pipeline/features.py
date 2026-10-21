
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("features")


def engineer_features(input_path, output_path):
    df = pd.read_csv(input_path)

    df["sepal_area"] = (
        df["sepal length (cm)"] * df["sepal width (cm)"]
    )

    df["petal_area"] = (
        df["petal length (cm)"] * df["petal width (cm)"]
    )

    df["sepal_to_petal_length_ratio"] = (
        df["sepal length (cm)"]
        / df["petal length (cm)"].replace(0, float("nan"))
    )

    df["petal_length_bin"] = pd.cut(
        df["petal length (cm)"],
        bins=[0, 2, 4.5, 7],
        labels=["short", "medium", "long"]
    )

    Path(output_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(output_path, index=False)

    logger.info(
        "Engineered features; saved %d columns",
        df.shape[1]
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/iris_preprocessed.csv"
    )

    parser.add_argument(
        "--output",
        default="data/processed/iris_features.csv"
    )

    args = parser.parse_args()
    engineer_features(args.input, args.output)