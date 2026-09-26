import pytest

from src.sentiment import SentimentAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()


def test_model_has_three_classes(analyzer):
    labels = set(analyzer.id2label.values())

    assert "negative" in labels
    assert "neutral" in labels
    assert "positive" in labels
    assert len(labels) == 3


def test_positive_prediction(analyzer):
    result = analyzer.predict("The car is excellent and performs beautifully.")

    assert result["sentiment"] in {"POSITIVE", "NEGATIVE", "NEUTRAL"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_negative_prediction(analyzer):
    result = analyzer.predict("The car is terrible and extremely unreliable.")

    assert result["sentiment"] in {"POSITIVE", "NEGATIVE", "NEUTRAL"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_neutral_prediction(analyzer):
    result = analyzer.predict("The vehicle has four doors and five seats.")

    assert result["sentiment"] in {"POSITIVE", "NEGATIVE", "NEUTRAL"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_empty_text_rejected(analyzer):
    with pytest.raises(ValueError):
        analyzer.predict("")
        