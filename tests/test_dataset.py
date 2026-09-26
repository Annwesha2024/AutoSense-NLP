import pandas as pd
import pytest

from src.dataset import (
    dataset_summary,
    get_labeled_rows,
    get_unlabeled_rows,
    validate_muse_columns,
)


def make_test_dataset():
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 3],
            "segment_id": [1, 2, 1, 1],
            "review": [
                "The engine is excellent.",
                "The seats are uncomfortable.",
                "The price is reasonable.",
                "No annotation here.",
            ],
            "aspect": [
                "engine",
                "seats",
                "price",
                "-",
            ],
            "opinion": [
                "excellent",
                "uncomfortable",
                "reasonable",
                "-",
            ],
            "sentiment": [
                "pos",
                "neg",
                "neu",
                "-",
            ],
        }
    )


def test_validate_muse_columns():
    df = make_test_dataset()

    validate_muse_columns(df)


def test_validate_muse_columns_rejects_invalid_dataset():
    df = pd.DataFrame(
        {
            "review": ["test"],
            "sentiment": ["pos"],
        }
    )

    with pytest.raises(ValueError):
        validate_muse_columns(df)


def test_get_labeled_rows():
    df = make_test_dataset()

    result = get_labeled_rows(df)

    assert len(result) == 3
    assert set(result["sentiment"]) == {"pos", "neg", "neu"}


def test_unlabeled_is_not_neutral():
    df = make_test_dataset()

    result = get_labeled_rows(df)

    assert "-" not in result["sentiment"].values


def test_get_unlabeled_rows():
    df = make_test_dataset()

    result = get_unlabeled_rows(df)

    assert len(result) == 1
    assert result.iloc[0]["sentiment"] == "-"


def test_dataset_summary():
    df = make_test_dataset()

    summary = dataset_summary(df)

    assert summary["total_rows"] == 4
    assert summary["labeled_rows"] == 3
    assert summary["unlabeled_rows"] == 1
    assert summary["unique_aspects"] == 3
    