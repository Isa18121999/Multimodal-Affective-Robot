from collections import Counter


class MultimodalEmotionFusion:
    """Fusiona señales emocionales de texto, voz y rostro.

    Cada modalidad puede entregar una etiqueta y una confianza entre 0 y 1.
    Una modalidad con ``available=False`` no participa en la fusión. Cuando
    ninguna señal válida está disponible, el resultado es ``UNKNOWN`` en lugar
    de inventar una emoción por defecto.
    """

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
        """Devuelve la emoción con mayor puntuación ponderada.

        ``signals`` puede tener la forma:
        {
            "text": {"emotion": "SAD", "confidence": 0.9, "available": True},
            "voice": {"emotion": "SAD", "confidence": 0.7, "available": False},
            "face": {"emotion": "CALM", "confidence": 0.6, "available": True},
        }

        ``available=False`` significa que la modalidad no produjo una señal
        utilizable y, por tanto, no debe aportar puntuación.
        """
        scores = Counter()
        active_weight = 0.0

        for modality, signal in signals.items():
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

            confidence = max(0.0, min(1.0, confidence))
            if confidence <= 0.0:
                continue

            weight = float(self.weights[modality])
            scores[emotion] += weight * confidence
            active_weight += weight

        if not scores or active_weight <= 0.0:
            return {"emotion": "UNKNOWN", "confidence": 0.0, "scores": {}}

        emotion, score = scores.most_common(1)[0]
        total = sum(scores.values())
        confidence = score / total if total else 0.0

        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "scores": dict(scores),
        }
