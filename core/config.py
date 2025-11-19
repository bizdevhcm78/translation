from pydantic_settings import BaseSettings
from typing import Dict


class Settings(BaseSettings):
    """Application settings"""

    APP_NAME: str = "Translation API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ML Models
    DEVICE: str = "auto"  # auto, cuda, cpu
    MAX_TEXT_LENGTH: int = 5000
    MAX_BATCH_SIZE: int = 100

    # Supported translations
    SUPPORTED_TRANSLATIONS: Dict[str, str] = {
        "en-vi": "Helsinki-NLP/opus-mt-en-vi",
        "vi-en": "Helsinki-NLP/opus-mt-vi-en",
        "ja-en": "Helsinki-NLP/opus-mt-ja-en",
        "en-ja": "Helsinki-NLP/opus-mt-en-jap",
        "ja-vi": "Helsinki-NLP/opus-mt-ja-vi",
    }

    # CORS
    CORS_ORIGINS: list = ["*"]

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
