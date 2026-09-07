"""Guías breves de regulación emocional para Chuwi.

No son diagnóstico ni tratamiento psicológico. Están pensadas como apoyo
inmediato y opcional mientras un adulto acompaña al niño.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SupportPlan:
    key: str
    name: str
    message: str


CALM_CHECK_IN = SupportPlan(
    "acompañamiento",
    "acompañamiento",
    "Estoy aquí contigo. ¿Quieres contarme cómo te sientes o prefieres respirar un momento?",
)

PLANS = {
    "ansiedad": SupportPlan(
        "respiracion",
        "respiración lenta",
        "Parece que puedes estar sintiendo algo intenso. Si quieres, hagamos tres respiraciones lentas: "
        "pon una mano en tu barriga, toma aire suave por la nariz contando uno, dos, tres; "
        "y suéltalo despacito contando uno, dos, tres, cuatro. Yo cuento contigo.",
    ),
    "tristeza": SupportPlan(
        "visualizacion",
        "lugar tranquilo",
        "Está bien sentirse triste. Si quieres, imagina un lugar donde te sientas seguro: quizá tu cama, "
        "un parque o estar con alguien que quieres. ¿Qué cosa bonita habría allí? También podemos avisar a un adulto que te acompañe.",
    ),
    "enojo": SupportPlan(
        "anclaje",
        "anclaje corporal",
        "Veo que puede haber mucho enojo. No estás en problemas por sentirlo. Probemos sentir los pies firmes en el suelo, "
        "apretar las manos suavemente por tres segundos y soltarlas. Después podemos decirle a un adulto qué pasó.",
    ),
}

RISK_PATTERN = re.compile(
    r"\b(me quiero morir|quiero morir|no quiero vivir|me quiero matar|matarme|suicid\w*|autolesion\w*|"
    r"hacerme daño|hacerme dano|me hacen daño|me hacen dano|me pegan|me golpean|me lastiman|abuso)\b",
    re.IGNORECASE,
)

RISK_MESSAGE = (
    "Gracias por decírmelo. No tienes que pasar por esto a solas. Voy a avisar a un adulto de confianza "
    "que esté contigo ahora. Si hay peligro inmediato, llamen a emergencias locales."
)


def is_risk_disclosure(text: str) -> bool:
    return bool(RISK_PATTERN.search(text or ""))


EMOTION_TO_PLAN = {
    "FEAR": "ansiedad",
    "ANXIOUS": "ansiedad",
    "SAD": "tristeza",
    "FRUSTRATED": "enojo",
    "HAPPY": "acompañamiento",
    "CALM": "acompañamiento",
}


def plan_for_emotion(emotion: str, allowed: list[str] | None = None) -> SupportPlan:
    """Convierte etiquetas internas y externas en un plan seguro de apoyo."""
    normalized = str(emotion or "CALM").strip().casefold()
    plan_key = EMOTION_TO_PLAN.get(normalized.upper())

    if plan_key is None:
        if any(word in normalized for word in ("miedo", "ansiedad", "ansioso", "nervioso", "asustado", "preocup")):
            plan_key = "ansiedad"
        elif any(word in normalized for word in ("triste", "llanto", "deprim", "pena", "extrañ", "extran")):
            plan_key = "tristeza"
        elif any(word in normalized for word in ("enojo", "enfad", "rabia", "molesto", "furia")):
            plan_key = "enojo"
        else:
            plan_key = "acompañamiento"

    plan = PLANS.get(plan_key, CALM_CHECK_IN)
    return plan if allowed is None or plan.key in allowed else CALM_CHECK_IN


def personalize_message(plan: SupportPlan, profile: dict | None) -> str:
    """Adapta el tono sin incluir diagnóstico, tratamiento ni promesas."""
    if not profile:
        return plan.message
    alias = profile.get("alias", "")
    age_group = profile.get("rango_edad", "6-12")
    interests = profile.get("intereses", [])
    greeting = f"{alias}, " if alias else ""
    if age_group == "3-5" and plan.key == "respiracion":
        return (
            f"{greeting}vamos poquito a poquito. Pon tu mano en la pancita: olemos una flor contando uno, dos; "
            "y soplamos una vela contando uno, dos, tres. Hagámoslo tres veces juntos."
        )
    if plan.key == "visualizacion" and interests:
        return (
            f"{greeting}si quieres, imagina un lugar tranquilo con {interests[0]}. "
            "¿Qué detalle bonito habría allí? Podemos respirar despacio y también avisar a un adulto que te acompañe."
        )
    return f"{greeting}{plan.message}"
