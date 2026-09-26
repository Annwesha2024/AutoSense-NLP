import pandas as pd
import pytest

from src.preprocessing import (
    clean_review_text,
    load_reviews,
    preprocess_reviews,
)


def test_clean_review_text_collapses_whitespace():
    text = "  The   engine   is   smooth.  "

    result = clean_review_text(text)

    assert result == "The engine is smooth."


def test_clean_review_text_handles_non_string():
    assert clean_review_text(None) == ""
    assert clean_review_text(123) == ""


def test_load_reviews_rejects_missing_columns(tmp_path):
    invalid_file = tmp_path / "invalid.csv"

    df = pd.DataFrame(
        {
            "review_id": [1],
            "review": ["The engine is smooth."],
        }
    )

    df.to_csv(invalid_file, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_reviews(invalid_file)


def test_preprocess_removes_missing_reviews():
    df = pd.DataFrame(
        {
            "review_id": [1, 2, 3],
            "vehicle": ["A", "B", "C"],
            "review": [
                "The engine is smooth.",
                None,
                "The seats are comfortable.",
            ],
        }
    )

    result = preprocess_reviews(df)

    assert len(result) == 2
    assert result["review"].isna().sum() == 0


def test_preprocess_removes_empty_reviews():
    df = pd.DataFrame(
        {
            "review_id": [1, 2],
            "vehicle": ["A", "B"],
            "review": [
                "   ",
                "The engine is powerful.",
            ],
        }
    )

    result = preprocess_reviews(df)

    assert len(result) == 1
    assert result.iloc[0]["review"] == "The engine is powerful."


def test_preprocess_removes_duplicate_reviews():
    df = pd.DataFrame(
        {
            "review_id": [1, 2, 3],
            "vehicle": ["A", "B", "C"],
            "review": [
                "The engine is smooth.",
                "The engine is smooth.",
                "The seats are comfortable.",
            ],
        }
    )

    result = preprocess_reviews(df)

    assert len(result) == 2
    assert result["review"].is_unique


def test_preprocess_preserves_valid_reviews():
    df = pd.DataFrame(
        {
            "review_id": [1, 2],
            "vehicle": ["Toyota", "Honda"],
            "review": [
                "The engine is smooth.",
                "The seats are comfortable.",
            ],
        }
    )

    result = preprocess_reviews(df)

    assert len(result) == 2
    assert result.iloc[0]["vehicle"] == "Toyota"
    assert result.iloc[1]["vehicle"] == "Honda"