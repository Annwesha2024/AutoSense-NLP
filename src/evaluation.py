from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from src.sentiment import SentimentAnalyzer
except ModuleNotFoundError:
    from sentiment import SentimentAnalyzer


DATASET_PATH = Path(
    "data/raw/muse_caraste/"
    "FinalFullAnnotationsAllLabelsCompleted.csv"
)


LABEL_MAP = {
    "pos": "POSITIVE",
    "neg": "NEGATIVE",
    "neu": "NEUTRAL",
}


def load_evaluation_data(
    file_path: str | Path,
) -> pd.DataFrame:

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        encoding="cp1252",
    )

    # Only genuine sentiment annotations.
    df = df[
        df["sentiment"].isin(
            LABEL_MAP.keys()
        )
    ].copy()

    # Opinion text must be available.
    df = df[
        df["opinion"].notna()
        & (df["opinion"].astype(str).str.strip() != "")
        & (df["opinion"].astype(str) != "-")
    ].copy()

    df["gold_label"] = (
        df["sentiment"].map(LABEL_MAP)
    )

    return df.reset_index(drop=True)


def evaluate_sentiment_model(
    df: pd.DataFrame,
    analyzer: SentimentAnalyzer,
    batch_size: int = 32,
) -> tuple[list[str], list[str]]:

    texts = df["opinion"].astype(str).tolist()
    gold_labels = df["gold_label"].tolist()

    predictions = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):

        batch = texts[
            start:start + batch_size
        ]

        results = analyzer.predict_batch(
            batch
        )

        predictions.extend(
            result["sentiment"]
            for result in results
        )

        processed = min(
            start + batch_size,
            len(texts),
        )

        if (
            processed % 500 == 0
            or processed == len(texts)
        ):
            print(
                f"Processed "
                f"{processed}/{len(texts)}"
            )

    return gold_labels, predictions


def print_evaluation(
    y_true: list[str],
    y_pred: list[str],
) -> None:

    labels = [
        "NEGATIVE",
        "NEUTRAL",
        "POSITIVE",
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 70)
    print("AUTOSENSE-NLP SENTIMENT MODEL EVALUATION")
    print("=" * 70)

    print()
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall   : {recall:.4f}")
    print(f"Weighted F1       : {f1:.4f}")

    print()
    print("Classification Report")
    print("-" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    print("Confusion Matrix")
    print("-" * 70)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual {label}"
            for label in labels
        ],
        columns=[
            f"Predicted {label}"
            for label in labels
        ],
    )

    print(matrix_df)


def save_predictions(
    df: pd.DataFrame,
    y_pred: list[str],
) -> None:

    output_path = Path(
        "outputs/sentiment_evaluation_predictions.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = df[
        [
            "id",
            "segment_id",
            "review",
            "aspect",
            "opinion",
            "sentiment",
            "gold_label",
        ]
    ].copy()

    result["predicted_label"] = y_pred

    result["correct"] = (
        result["gold_label"]
        == result["predicted_label"]
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Predictions saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":

    print(
        "Loading MuSe-CarASTE evaluation data..."
    )

    evaluation_df = load_evaluation_data(
        DATASET_PATH
    )

    print(
        f"Evaluation samples: "
        f"{len(evaluation_df)}"
    )

    print()
    print("Gold label distribution:")
    print(
        evaluation_df["gold_label"]
        .value_counts()
    )

    print()
    print("Loading transformer model...")

    analyzer = SentimentAnalyzer()

    print()
    print("Running evaluation...")

    y_true, y_pred = evaluate_sentiment_model(
        evaluation_df,
        analyzer,
    )

    print_evaluation(
        y_true,
        y_pred,
    )

    save_predictions(
        evaluation_df,
        y_pred,
    )