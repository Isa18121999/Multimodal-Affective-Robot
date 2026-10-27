from src.emotion.chuwi_emotional_adaptation import ChuwiEmotionalAdaptation
from src.emotion.chuwi_emotional_manager import ChuwiEmotionalManager
from src.emotion.multimodal_emotion_fusion import MultimodalEmotionFusion


def test_empty_text_is_unknown_not_calm():
    adaptation = ChuwiEmotionalAdaptation()
    assert adaptation.detect_emotion("") == "UNKNOWN"
    assert adaptation.detect_emotion("   ") == "UNKNOWN"


def test_empty_interaction_without_modalities_returns_unknown():
    manager = ChuwiEmotionalManager()
    result = manager.analyze_interaction("")
    assert result["emotion"] == "UNKNOWN"
    assert result["confidence"] == 0.0
    assert result["scores"] == {}


def test_surprise_does_not_normalize_to_calm():
    assert ChuwiEmotionalManager.normalize_emotion("sorprendido") == "SURPRISE"
    assert ChuwiEmotionalManager.normalize_emotion("unknown") == "UNKNOWN"


def test_text_signal_still_detects_obvious_keyword():
    manager = ChuwiEmotionalManager()
    result = manager.analyze_interaction("Tengo mucho miedo")
    assert result["emotion"] == "FEAR"
    assert result["confidence"] == 1.0
