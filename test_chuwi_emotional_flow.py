from src.memory.memory_manager import ChuwiMemory
from src.emotion.chuwi_emotional_manager import ChuwiEmotionalManager
from src.emotion.multimodal_emotion_fusion import MultimodalEmotionFusion
from src.core.chuwi_ai_engine import ChuwiAIEngine
from app.support_protocol import plan_for_emotion


def run_tests():
    print("=== PRUEBAS CHUWI EMOTIONAL AI ===")

    memory = ChuwiMemory()
    emotional = ChuwiEmotionalManager()
    ai = ChuwiAIEngine(emotional_manager=emotional)

    memory.save_user_name("Paciente")
    user = memory.get_user()
    assert user["name"] == "Paciente"
    print("OK - Memoria del paciente")

    cases = [
        ("Tengo miedo de mi operación", "FEAR", "acompañamiento y calma"),
        ("Extraño mi casa y mi familia", "SAD", "escucha activa y apoyo"),
        ("Estoy muy nervioso por estar en el hospital", "ANXIOUS", "respiración y conversación tranquila"),
        ("Estoy molesto porque no puedo hacerlo", "FRUSTRATED", "motivación y distracción"),
        ("Estoy feliz, todo está genial", "HAPPY", "refuerzo emocional"),
    ]

    for message, expected_emotion, expected_strategy in cases:
        result = emotional.analyze_interaction(message)
        assert result["emotion"] == expected_emotion
        assert result["strategy"] == expected_strategy
        print(f"OK - {expected_emotion}: {expected_strategy}")

    multimodal = emotional.analyze_interaction(
        "Estoy tranquilo",
        modalities={"face": {"emotion": "triste", "confidence": 1.0}},
    )
    assert multimodal["emotion"] == "SAD"
    assert multimodal["scores"]["SAD"] > multimodal["scores"]["CALM"]
    print("OK - Fusión texto + rostro")

    assert plan_for_emotion("FEAR").key == "respiracion"
    assert plan_for_emotion("SAD").key == "visualizacion"
    assert plan_for_emotion("ANXIOUS").key == "respiracion"
    assert plan_for_emotion("FRUSTRATED").key == "anclaje"
    assert plan_for_emotion("HAPPY").key == "acompañamiento"
    print("OK - Mapeo de emociones internas a protocolos")

    response, strategy = ai.generate_response("Tengo miedo de mi operación", user)
    assert response is not None
    assert strategy == "acompañamiento y calma"
    print("OK - Respuesta adaptativa con estrategia emocional")

    print("=== TODAS LAS PRUEBAS COMPLETADAS ===")


if __name__ == "__main__":
    run_tests()
