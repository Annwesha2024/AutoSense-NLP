from __future__ import annotations

import re
from typing import Dict, List

try:
    from src.aspect_extraction import extract_aspects
    from src.sentiment import SentimentAnalyzer
except ModuleNotFoundError:
    from aspect_extraction import extract_aspects
    from sentiment import SentimentAnalyzer


CLAUSE_PATTERN = (
    r"\s*(?:but|however|although|though|while|yet|and)\s*"
)

SENTIMENT_PRIORITY = {
    "POSITIVE": 1,
    "NEUTRAL": 2,
    "NEGATIVE": 3,
}


def split_into_clauses(text: str) -> List[str]:
    """
    Split a review into simple clauses around common conjunctions.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    clauses = re.split(CLAUSE_PATTERN, text, flags=re.IGNORECASE)

    return [
        clause.strip()
        for clause in clauses
        if clause.strip()
    ]


def analyze_review(
    text: str,
    sentiment_analyzer: SentimentAnalyzer,
) -> Dict:
    """
    Perform aspect-level sentiment analysis on one review.
    """

    if not isinstance(text, str) or not text.strip():
        return {
            "review": text,
            "overall_sentiment": "NEUTRAL",
            "aspects": [],
        }

    clauses = split_into_clauses(text)

    aspect_results = []

    for clause in clauses:
        aspects = extract_aspects(clause)

        if not aspects:
            continue

        sentiment_result = sentiment_analyzer.predict(clause)

        for aspect in aspects:
            aspect_results.append(
                {
                    "aspect": aspect,
                    "sentiment": sentiment_result["sentiment"],
                    "confidence": sentiment_result["confidence"],
                    "text": clause,
                }
            )

    overall_sentiment = determine_overall_sentiment(
        aspect_results
    )

    return {
        "review": text,
        "overall_sentiment": overall_sentiment,
        "aspects": aspect_results,
    }


def determine_overall_sentiment(
    aspect_results: List[Dict],
) -> str:
    """
    Determine review-level sentiment from aspect sentiments.
    """

    if not aspect_results:
        return "NEUTRAL"

    sentiments = {
        result["sentiment"]
        for result in aspect_results
    }

    if "POSITIVE" in sentiments and "NEGATIVE" in sentiments:
        return "MIXED"

    if "NEGATIVE" in sentiments:
        return "NEGATIVE"

    if "POSITIVE" in sentiments:
        return "POSITIVE"

    return "NEUTRAL"


def print_analysis(result: Dict) -> None:
    """
    Pretty-print an analysis result.
    """

    print("\n" + "=" * 70)
    print("AUTOMOTIVE REVIEW ANALYSIS")
    print("=" * 70)

    print(f"\nReview:\n{result['review']}")

    print(f"\nOverall Sentiment: {result['overall_sentiment']}")

    print("\nAspect-Level Results:")

    if not result["aspects"]:
        print("  No automotive aspects detected.")
        return

    for item in result["aspects"]:
        print(
            f"  {item['aspect']:15} → "
            f"{item['sentiment']:8} "
            f"({item['confidence']:.3f})"
        )
        print(f"                    Text: {item['text']}")


if __name__ == "__main__":

    analyzer = SentimentAnalyzer()

    examples = [
        "The battery range is excellent but charging takes too long.",
        "The seats are comfortable and the interior looks beautiful.",
        "The engine is powerful but the ride is uncomfortable.",
        "The vehicle has four doors and five seats.",
        "The price is expensive and the service is terrible.",
    ]

    for review in examples:
        result = analyze_review(review, analyzer)
        print_analysis(result)