from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://chuwibot:chuwibot_secret@localhost:5432/chuwibot_db"
    SECRET_KEY: str = "cambiar-en-produccion"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    ENVIRONMENT: str = "development"
    INITIAL_ADMIN_PASSWORD: str = ""

    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


def validate_deployment_settings() -> None:
    """Fail early rather than exposing known credentials in production."""
    if settings.ENVIRONMENT.lower() != "production":
        return
    if settings.SECRET_KEY in {"", "cambiar-en-produccion", "cambiar-en-produccion-abc123"}:
        raise RuntimeError("SECRET_KEY debe configurarse con un valor seguro en producción")
    if not settings.INITIAL_ADMIN_PASSWORD:
        raise RuntimeError("INITIAL_ADMIN_PASSWORD es obligatorio en producción")
