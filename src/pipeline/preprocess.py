
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("preprocess")


def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    initial_rows = len(df)
    df = df.drop_duplicates()

    logger.info(
        "Dropped %d duplicate rows",
        initial_rows - len(df)
    )

    numeric_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=["species"])
    df = df.drop(columns=["collected_at"], errors="ignore")

    Path(output_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(output_path, index=False)

    logger.info(
        "Preprocessed %d rows -> %s",
        len(df), output_path
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/raw/iris_raw.csv"
    )

    parser.add_argument(
        "--output",
        default="data/processed/iris_preprocessed.csv"
    )

    args = parser.parse_args()
    preprocess(args.input, args.output)