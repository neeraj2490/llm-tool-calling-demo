"""Application configuration loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings for Groq and OpenWeatherMap."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    groq_api_key: str = Field(..., min_length=1, description="Groq API key")
    openweather_api_key: str = Field(
        ...,
        min_length=1,
        description="OpenWeatherMap API key",
    )


def get_settings() -> Settings:
    """Return a validated Settings instance."""
    return Settings()
