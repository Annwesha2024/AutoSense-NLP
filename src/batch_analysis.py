from __future__ import annotations

from pathlib import Path

import pandas as pd

from aspect_sentiment import analyze_review
from sentiment import SentimentAnalyzer


def analyze_csv(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if "review" not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'review' column."
        )

    analyzer = SentimentAnalyzer()

    results = []

    for _, row in df.iterrows():

        review = str(row["review"])

        analysis = analyze_review(
            review,
            analyzer,
        )

        aspects = analysis["aspects"]

        if aspects:

            for item in aspects:

                results.append(
                    {
                        "review_id": row.get(
                            "review_id",
                            len(results) + 1,
                        ),
                        "vehicle": row.get(
                            "vehicle",
                            "",
                        ),
                        "review": review,
                        "aspect": item["aspect"],
                        "sentiment": item["sentiment"],
                        "confidence": round(
                            item["confidence"],
                            4,
                        ),
                        "matched_text": item["text"],
                        "overall_sentiment":
                            analysis["overall_sentiment"],
                    }
                )

        else:

            results.append(
                {
                    "review_id": row.get(
                        "review_id",
                        len(results) + 1,
                    ),
                    "vehicle": row.get(
                        "vehicle",
                        "",
                    ),
                    "review": review,
                    "aspect": "None",
                    "sentiment": "NEUTRAL",
                    "confidence": 1.0,
                    "matched_text": "",
                    "overall_sentiment":
                        analysis["overall_sentiment"],
                }
            )

    result_df = pd.DataFrame(results)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    return result_df


if __name__ == "__main__":

    input_file = (
        "data/raw/synthetic/"
        "automotive_reviews.csv"
    )

    output_file = (
        "outputs/"
        "automotive_sentiment_results.csv"
    )

    print("Running batch automotive sentiment analysis...")
    print()

    result = analyze_csv(
        input_file,
        output_file,
    )

    print(
        f"Input reviews analyzed: "
        f"{result['review'].nunique()}"
    )

    print(
        f"Aspect-level records generated: "
        f"{len(result)}"
    )

    print(
        f"Output saved to: "
        f"{output_file}"
    )

    print()
    print("Sentiment distribution:")
    print(result["sentiment"].value_counts())

    print()
    print("Aspect distribution:")
    print(result["aspect"].value_counts().head(15))