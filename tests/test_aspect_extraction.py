from src.aspect_extraction import (
    extract_aspects,
    extract_aspect_matches,
)


def test_battery_and_charging():
    text = "The battery range is excellent but charging takes too long."

    aspects = extract_aspects(text)

    assert "Battery" in aspects
    assert "Charging" in aspects


def test_engine_and_comfort():
    text = "The engine is powerful and the seats are comfortable."

    aspects = extract_aspects(text)

    assert "Engine" in aspects
    assert "Comfort" in aspects
    assert "Performance" in aspects
    assert "Interior" not in aspects


def test_mileage():
    text = "The mileage is impressive."

    aspects = extract_aspects(text)

    assert "Mileage" in aspects


def test_price_and_infotainment():
    text = "The infotainment system is good but the price is expensive."

    aspects = extract_aspects(text)

    assert "Infotainment" in aspects
    assert "Price" in aspects


def test_empty_text():
    assert extract_aspects("") == []
    assert extract_aspects(None) == []


def test_word_boundary():
    text = "The car is fantastic."

    aspects = extract_aspects(text)

    assert "Vehicle" in aspects


def test_exact_matches_are_returned():
    text = "The battery range is excellent and charging is fast."

    matches = extract_aspect_matches(text)

    assert "Battery" in matches
    assert "Charging" in matches
    assert "battery range" in matches["Battery"]
    assert "charging" in matches["Charging"]