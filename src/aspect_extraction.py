from __future__ import annotations

import re
from typing import Dict, List


ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "Vehicle": [
        "vehicle",
        "car",
        "automobile",
        "suv",
        "sedan",
        "hatchback",
    ],

    "Engine": [
        "engine",
        "motor",
        "cylinder",
        "turbo",
        "turbocharged",
    ],

    "Battery": [
        "battery",
        "battery pack",
        "battery life",
        "battery capacity",
        "battery range",
        "range",
    ],

    "Mileage": [
        "mileage",
        "fuel economy",
        "fuel efficiency",
        "mpg",
        "kmpl",
        "fuel consumption",
    ],

    "Safety": [
        "safety",
        "airbag",
        "brake",
        "brakes",
        "abs",
        "collision",
        "blind spot",
        "crash",
        "traction control",
    ],

    "Comfort": [
        "comfort",
        "comfortable",
        "uncomfortable",
        "ride comfort",
        "seat comfort",
        "seats",
        "seat",
    ],

    "Service": [
        "service",
        "servicing",
        "maintenance",
        "dealer",
        "dealership",
        "repair",
        "workshop",
    ],

    "Infotainment": [
        "infotainment",
        "navigation",
        "bluetooth",
        "audio",
        "stereo",
        "speaker",
        "speakers",
        "screen",
        "touchscreen",
    ],

    "Price": [
        "price",
        "cost",
        "expensive",
        "affordable",
        "value",
        "value for money",
        "worth the money",
    ],

    "Design": [
        "design",
        "styling",
        "style",
        "looks",
        "appearance",
    ],

    "Performance": [
        "performance",
        "power",
        "powerful",
        "acceleration",
        "speed",
        "handling",
    ],

    "Reliability": [
        "reliable",
        "reliability",
        "dependable",
        "durability",
        "durable",
        "problem",
        "problems",
        "fault",
        "faults",
    ],

    "Charging": [
        "charging",
        "charge",
        "charger",
        "charging time",
        "fast charging",
        "charging station",
    ],

    "Interior": [
        "interior",
        "dashboard",
        "cabin",
        "legroom",
        "headroom",
        "boot",
        "trunk",
    ],

    "Exterior": [
        "exterior",
        "body",
        "paint",
        "wheels",
        "wheel",
        "door",
        "doors",
        "headlights",
        "tail lights",
    ],
}


def _contains_phrase(text: str, phrase: str) -> bool:
    """
    Match a complete word or phrase rather than a substring.
    """

    pattern = (
        r"(?<!\w)"
        + re.escape(phrase.lower())
        + r"(?!\w)"
    )

    return re.search(pattern, text.lower()) is not None


def extract_aspects(text: str) -> List[str]:
    """
    Detect standardized automotive aspect categories.

    Each category is returned at most once.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    detected = []

    for category, keywords in ASPECT_KEYWORDS.items():

        if any(
            _contains_phrase(text, keyword)
            for keyword in keywords
        ):
            detected.append(category)

    return detected


def extract_aspect_matches(
    text: str,
) -> Dict[str, List[str]]:
    """
    Return the exact keywords matched for every
    detected aspect category.

    Useful for debugging and explainability.
    """

    if not isinstance(text, str) or not text.strip():
        return {}

    matches = {}

    for category, keywords in ASPECT_KEYWORDS.items():

        found = [
            keyword
            for keyword in keywords
            if _contains_phrase(text, keyword)
        ]

        if found:
            matches[category] = found

    return matches


if __name__ == "__main__":

    examples = [
        "The battery range is excellent but charging takes too long.",
        "The engine is powerful and the seats are comfortable.",
        "The infotainment screen is slow and the price is expensive.",
        "The mileage is excellent and the vehicle is reliable.",
        "The interior plastics feel cheap.",
        "The infotainment system is responsive and easy to use.",
    ]

    for review in examples:

        print(f"\nReview: {review}")
        print("Aspects:", extract_aspects(review))
        print(
            "Matches:",
            extract_aspect_matches(review),
        )