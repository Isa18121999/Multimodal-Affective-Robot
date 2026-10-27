from enum import Enum


class EmotionalState(Enum):
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    FEAR = "fear"
    FRUSTRATED = "frustrated"


class ChuwiEmotionalAdaptation:
    """Algoritmo base de adaptación emocional para pacientes pediátricos."""

    def analyze_text(self, message):
        text = (message or "").lower().strip()
        if not text:
            return None

        if any(word in text for word in ["miedo", "asustado", "temor"]):
            return EmotionalState.FEAR
        if any(word in text for word in ["triste", "llorar", "solo", "extraño", "extrano"]):
            return EmotionalState.SAD
        if any(word in text for word in ["nervioso", "ansioso", "preocupado"]):
            return EmotionalState.ANXIOUS
        if any(word in text for word in ["enojado", "molesto", "frustrado"]):
            return EmotionalState.FRUSTRATED
        if any(word in text for word in ["feliz", "genial", "bien"]):
            return EmotionalState.HAPPY
        return EmotionalState.CALM

    def detect_emotion(self, message):
        """Devuelve un estado o UNKNOWN si el mensaje no aporta evidencia textual."""
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
