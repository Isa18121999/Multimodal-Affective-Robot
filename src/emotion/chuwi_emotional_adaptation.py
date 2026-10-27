from enum import Enum


class EmotionalState(Enum):
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    FEAR = "fear"
    FRUSTRATED = "frustrated"


class ChuwiEmotionalAdaptation:
    """Rule-based text emotion detector. Empty text is UNKNOWN, not CALM."""

    KEYWORDS = {
        EmotionalState.FEAR: ("miedo", "asustado", "temor"),
        EmotionalState.SAD: ("triste", "llorar", "solo", "extraño", "extrano"),
        EmotionalState.ANXIOUS: ("nervioso", "ansioso", "preocupado"),
        EmotionalState.FRUSTRATED: ("enojado", "molesto", "frustrado"),
        EmotionalState.HAPPY: ("feliz", "genial", "bien"),
    }

    def analyze_text(self, message):
        text = (message or "").casefold().strip()
        if not text:
            return None

        # Preserve the current rule-based precedence while avoiding a false calm
        # classification when there is no user statement to analyze.
        for state, words in self.KEYWORDS.items():
            if any(word in text for word in words):
                return state
        return EmotionalState.CALM

    def detect_emotion(self, message):
        """Return the canonical state name or UNKNOWN for empty input."""
        state = self.analyze_text(message)
        return state.name if state is not None else "UNKNOWN"

    def select_strategy(self, emotion):
        strategies = {
            EmotionalState.FEAR: "acompañamiento_y_calma",
            EmotionalState.SAD: "escucha_activa",
            EmotionalState.ANXIOUS: "relajacion_guiada",
            EmotionalState.FRUSTRATED: "motivacion_y_juego",
            EmotionalState.HAPPY: "aprendizaje_y_interaccion",
            EmotionalState.CALM: "conversacion_general",
        }
        return strategies[emotion]
