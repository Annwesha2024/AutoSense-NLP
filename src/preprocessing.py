from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"review_id", "vehicle", "review"}


def load_reviews(file_path: str | Path) -> pd.DataFrame:
    """
    Load an automotive review CSV file.

    Parameters
    ----------
    file_path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded review dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


def clean_review_text(text: str) -> str:
    """
    Perform basic text cleaning while preserving useful NLP information.

    Cleaning performed:
    - Convert non-string values to empty strings.
    - Remove leading/trailing whitespace.
    - Collapse repeated whitespace.

    We deliberately do not:
    - remove stopwords
    - remove punctuation
    - lowercase everything

    Those operations can destroy useful information for transformer models
    and aspect detection.
    """
    if not isinstance(text, str):
        return ""

    text = " ".join(text.split())

    return text.strip()


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate an automotive review DataFrame.

    Processing:
    1. Copy the input DataFrame.
    2. Remove rows with missing/empty reviews.
    3. Clean review text.
    4. Remove duplicate reviews.
    5. Reset the DataFrame index.
    """
    processed = df.copy()

    # Remove genuinely missing review values.
    processed = processed.dropna(subset=["review"])

    # Clean the review text.
    processed["review"] = processed["review"].apply(clean_review_text)

    # Remove reviews that became empty after cleaning.
    processed = processed[processed["review"] != ""]

    # Remove exact duplicate review texts.
    processed = processed.drop_duplicates(
        subset=["review"],
        keep="first",
    )

    # Make the resulting DataFrame easier to work with.
    processed = processed.reset_index(drop=True)

    return processed


def save_processed_reviews(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save a processed DataFrame to CSV.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_path = Path(
        "data/raw/synthetic/automotive_reviews.csv"
    )

    output_path = Path(
        "data/processed/synthetic_reviews_cleaned.csv"
    )

    reviews = load_reviews(input_path)

    print(f"Loaded reviews: {len(reviews)}")

    processed_reviews = preprocess_reviews(reviews)

    print(f"Processed reviews: {len(processed_reviews)}")
    print(
        f"Removed reviews: "
        f"{len(reviews) - len(processed_reviews)}"
    )

    save_processed_reviews(
        processed_reviews,
        output_path,
    )

    print(f"Saved processed dataset to: {output_path}")