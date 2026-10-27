from src.emotion.chuwi_emotional_adaptation import ChuwiEmotionalAdaptation
from src.emotion.emotion_database import EmotionDatabase
from src.emotion.multimodal_emotion_fusion import MultimodalEmotionFusion


class ChuwiEmotionalManager:
    """Coordina detección emocional, fusión multimodal y estrategias."""

    EMOTION_DB_MAP = {
        "ANXIOUS": "anxiety",
    }

    EXTERNAL_EMOTION_MAP = {
        "feliz": "HAPPY",
        "happy": "HAPPY",
        "alegre": "HAPPY",
        "triste": "SAD",
        "sad": "SAD",
        "enojado": "FRUSTRATED",
        "angry": "FRUSTRATED",
        "molesto": "FRUSTRATED",
        "asustado": "FEAR",
        "fear": "FEAR",
        "miedo": "FEAR",
        "nervioso": "ANXIOUS",
        "ansioso": "ANXIOUS",
        "anxious": "ANXIOUS",
        "neutral": "CALM",
        "calm": "CALM",
        "sorprendido": "CALM",
        "surprised": "CALM",
    }

    def __init__(self, fusion=None):
        self.adapter = ChuwiEmotionalAdaptation()
        self.database = EmotionDatabase()
        self.fusion = fusion or MultimodalEmotionFusion()

    @classmethod
    def normalize_emotion(cls, emotion):
        """Normaliza etiquetas externas sin convertir señales ausentes en calma."""
        normalized = str(emotion or "").strip().casefold()
        if not normalized or normalized == "unknown":
            return "UNKNOWN"
        return cls.EXTERNAL_EMOTION_MAP.get(normalized, normalized.upper())

    def analyze_interaction(self, message, modalities=None):
        """Analiza texto y fusiona señales opcionales de voz y rostro."""
        message = message or ""
        signals = {}
        if message.strip():
            text_emotion = self.adapter.detect_emotion(message)
            signals["text"] = {
                "emotion": self.normalize_emotion(text_emotion),
                "confidence": 0.6,
                "available": True,
            }

        for modality, signal in (modalities or {}).items():
            if not signal:
                continue
            signals[modality] = {
                **signal,
                "emotion": self.normalize_emotion(signal.get("emotion")),
            }

        fused = self.fusion.fuse(signals)
        emotion = fused["emotion"]
        database_emotion = self.EMOTION_DB_MAP.get(emotion, emotion.lower())
        strategy = self.database.get_strategy(database_emotion)

        return {
            "emotion": emotion,
            "strategy": strategy,
            "confidence": fused["confidence"],
            "scores": fused["scores"],
        }
