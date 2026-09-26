from pathlib import Path

import pandas as pd


BASE_COLUMNS = {
    "id",
    "segment_id",
    "review",
    "aspect",
    "opinion",
    "sentiment",
}

VALID_SENTIMENTS = {"pos", "neg", "neu"}
UNLABELED_SENTIMENT = "-"


def load_muse_dataset(
    file_path: str | Path,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load a MuSe-CarASTE CSV file.

    The full annotation file uses cp1252 encoding, while the
    train/development files can be read using UTF-8.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path, encoding=encoding)


def validate_muse_columns(df: pd.DataFrame) -> None:
    """Validate that the required MuSe-CarASTE columns exist."""

    missing = BASE_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing MuSe-CarASTE columns: {sorted(missing)}"
        )


def get_labeled_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows containing a valid sentiment annotation.

    '-' means no aspect-level sentiment annotation and is NOT
    treated as Neutral.
    """
    validate_muse_columns(df)

    return df[df["sentiment"].isin(VALID_SENTIMENTS)].copy()


def get_unlabeled_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose sentiment annotation is '-'. """

    validate_muse_columns(df)

    return df[df["sentiment"] == UNLABELED_SENTIMENT].copy()


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return basic dataset statistics."""

    validate_muse_columns(df)

    labeled = get_labeled_rows(df)

    return {
        "total_rows": len(df),
        "labeled_rows": len(labeled),
        "unlabeled_rows": len(df) - len(labeled),
        "unique_reviews": df["id"].nunique(),
        "unique_segments": df["segment_id"].nunique(),
        "sentiment_counts": (
            labeled["sentiment"].value_counts().to_dict()
        ),
        "unique_aspects": labeled["aspect"].nunique(),
    }


if __name__ == "__main__":
    dataset_path = Path(
        "data/raw/muse_caraste/"
        "FinalFullAnnotationsAllLabelsCompleted.csv"
    )

    df = load_muse_dataset(
        dataset_path,
        encoding="cp1252",
    )

    summary = dataset_summary(df)

    print("MuSe-CarASTE Dataset Summary")
    print("-" * 35)

    for key, value in summary.items():
        print(f"{key}: {value}")