"""Application entry point for the LLM weather tool-calling demo."""

from __future__ import annotations

import logging
import sys

from groq import Groq

from config import get_settings
from services.agent import WeatherAgent

USER_PROMPT = (
    "Pick a random, interesting city anywhere in the world "
    "and tell me its current weather."
)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a weather lookup tool. "
    "When the user asks about weather, choose an appropriate city and call "
    "get_current_weather before answering. Summarize the results clearly."
)


def configure_logging() -> None:
    """Configure structured console logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    try:
        settings = get_settings()
    except Exception:
        logger.exception(
            "Failed to load configuration. Ensure GROQ_API_KEY and "
            "OPENWEATHER_API_KEY are set in the environment or .env file."
        )
        sys.exit(1)

    client = Groq(api_key=settings.groq_api_key)
    agent = WeatherAgent(
        client,
        openweather_api_key=settings.openweather_api_key,
    )

    logger.info("Starting weather tool-calling demo")
    logger.info("User prompt: %s", USER_PROMPT)

    try:
        response = agent.run(USER_PROMPT, system_prompt=SYSTEM_PROMPT)
    except Exception:
        logger.exception("Agent execution failed")
        sys.exit(1)

    logger.info("Final response:\n%s", response)
    print("\n--- Assistant ---")
    print(response)


if __name__ == "__main__":
    main()
