from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABEL_MAP = {
    "negative": "NEGATIVE",
    "neutral": "NEUTRAL",
    "positive": "POSITIVE",
}


class SentimentAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

        # Read labels directly from the model configuration.
        self.id2label = {
            int(key): value.lower()
            for key, value in self.model.config.id2label.items()
        }

    def predict(self, text: str) -> Dict[str, float | str]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())
        raw_label = self.id2label[predicted_id]

        sentiment = LABEL_MAP.get(raw_label, raw_label.upper())
        confidence = float(probabilities[predicted_id].item())

        return {
            "sentiment": sentiment,
            "confidence": confidence,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, float | str]]:
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    examples = [
        "The battery range is excellent.",
        "The charging time is terrible.",
        "The car has four doors.",
        "The seats are comfortable.",
        "The engine is powerful but the ride is uncomfortable.",
    ]

    for text in examples:
        result = analyzer.predict(text)

        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")