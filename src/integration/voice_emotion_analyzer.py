from __future__ import annotations


class VoiceEmotionAnalyzer:
    """Punto de integración para reconocimiento emocional de voz.

    El proyecto no debe inventar una emoción a partir del audio sin un modelo
    de emoción de voz entrenado y validado. Por eso la implementación base
    devuelve una señal de confianza cero. Un modelo real puede reemplazar esta
    clase sin modificar el controlador ni el motor de fusión multimodal.
    """

    def analyze(self, audio) -> dict:
        return {
            "emotion": "CALM",
            "confidence": 0.0,
            "source": "voice",
            "available": False,
        }
