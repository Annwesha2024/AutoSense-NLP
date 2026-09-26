from src.aspect_sentiment import (
    determine_overall_sentiment,
    split_into_clauses,
)


def test_clause_splitting():
    text = "The battery is excellent but charging is slow."

    clauses = split_into_clauses(text)

    assert len(clauses) == 2
    assert "battery" in clauses[0].lower()
    assert "charging" in clauses[1].lower()


def test_clause_splitting_empty():
    assert split_into_clauses("") == []
    assert split_into_clauses(None) == []


def test_positive_overall_sentiment():
    results = [
        {"aspect": "Battery", "sentiment": "POSITIVE"},
        {"aspect": "Comfort", "sentiment": "POSITIVE"},
    ]

    assert determine_overall_sentiment(results) == "POSITIVE"


def test_negative_overall_sentiment():
    results = [
        {"aspect": "Charging", "sentiment": "NEGATIVE"},
        {"aspect": "Price", "sentiment": "NEGATIVE"},
    ]

    assert determine_overall_sentiment(results) == "NEGATIVE"


def test_mixed_overall_sentiment():
    results = [
        {"aspect": "Battery", "sentiment": "POSITIVE"},
        {"aspect": "Charging", "sentiment": "NEGATIVE"},
    ]

    assert determine_overall_sentiment(results) == "MIXED"


def test_neutral_overall_sentiment():
    results = [
        {"aspect": "Vehicle", "sentiment": "NEUTRAL"},
    ]

    assert determine_overall_sentiment(results) == "NEUTRAL"


def test_empty_results_are_neutral():
    assert determine_overall_sentiment([]) == "NEUTRAL"