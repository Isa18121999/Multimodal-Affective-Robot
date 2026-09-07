from src.integration.voice_emotion_analyzer import VoiceEmotionAnalyzer


class FakeAudio:
    def get_raw_data(self, convert_rate=16000, convert_width=2):
        return b"\x00\x00" * 16000


def run_tests():
    analyzer = VoiceEmotionAnalyzer()

    assert analyzer.analyze(None) is not None
    empty = analyzer.analyze(None)
    assert empty["confidence"] == 0.0
    assert empty["available"] is False

    signal = analyzer.analyze(FakeAudio())
    assert signal["confidence"] == 0.0
    assert signal["source"] == "voice"
    assert signal["available"] is False
    print("OK - Punto de integración de emoción vocal")
    print("OK - Sin modelo vocal, la fusión no inventa emociones")


if __name__ == "__main__":
    run_tests()
