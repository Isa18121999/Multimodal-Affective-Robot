from src.emotion.multimodal_emotion_fusion import MultimodalEmotionFusion


def test_ignores_unavailable_and_unknown_modalities():
    fusion = MultimodalEmotionFusion()
    result = fusion.fuse({
        "text": {"emotion": "SAD", "confidence": 0.6, "available": True},
        "voice": {"emotion": "CALM", "confidence": 0.9, "available": False},
        "face": {"emotion": "UNKNOWN", "confidence": 0.0, "available": False},
    })
    assert result["emotion"] == "SAD"
    assert result["confidence"] == 1.0
    assert result["scores"] == {"SAD": 0.24}


def test_returns_unknown_when_no_valid_signal_exists():
    fusion = MultimodalEmotionFusion()
    result = fusion.fuse({
        "text": {"emotion": "CALM", "confidence": 0.0},
        "voice": {"emotion": "UNKNOWN", "confidence": 0.0, "available": False},
    })
    assert result == {"emotion": "UNKNOWN", "confidence": 0.0, "scores": {}}


def test_clamps_confidence_and_respects_modality_weights():
    fusion = MultimodalEmotionFusion()
    result = fusion.fuse({
        "text": {"emotion": "HAPPY", "confidence": 1.5},
        "face": {"emotion": "SAD", "confidence": -0.1},
    })
    assert result["emotion"] == "HAPPY"
    assert result["scores"] == {"HAPPY": 0.4}
