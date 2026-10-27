from collections import Counter
import math


class MultimodalEmotionFusion:
    """Late fusion of available text, voice, and face emotion signals."""

    DEFAULT_WEIGHTS = {
        "text": 0.40,
        "voice": 0.30,
        "face": 0.30,
    }

    def __init__(self, weights=None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()

    def _validate_weights(self):
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Los pesos multimodales no pueden ser negativos")
        if sum(self.weights.values()) <= 0:
            raise ValueError("La suma de pesos multimodales debe ser mayor que cero")

    def fuse(self, signals):
        """Return the best-supported emotion; UNKNOWN means no usable evidence."""
        scores = Counter()
        active_weight = 0.0

        for modality, signal in (signals or {}).items():
            if modality not in self.weights or not signal:
                continue
            if signal.get("available", True) is False:
                continue

            emotion = str(signal.get("emotion", "")).strip().upper()
            if not emotion or emotion == "UNKNOWN":
                continue

            try:
                confidence = float(signal.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(confidence):
                continue

            confidence = max(0.0, min(1.0, confidence))
            if confidence <= 0.0:
                continue

            weight = float(self.weights[modality])
            if weight <= 0.0:
                continue
            scores[emotion] += weight * confidence
            active_weight += weight

        if not scores or active_weight <= 0.0:
            return {"emotion": "UNKNOWN", "confidence": 0.0, "scores": {}}

        emotion, winning_score = scores.most_common(1)[0]
        total_score = sum(scores.values())
        confidence = winning_score / total_score if total_score else 0.0

        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "scores": {key: round(value, 6) for key, value in scores.items()},
        }
