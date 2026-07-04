"""OpenWeatherMap client and Groq tool schema for current weather lookups."""

from __future__ import annotations

import logging
from typing import Any, Final

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException, Timeout

logger = logging.getLogger(__name__)

OPENWEATHER_BASE_URL: Final[str] = "https://api.openweathermap.org/data/2.5/weather"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0


class WeatherApiError(Exception):
    """Base exception for OpenWeatherMap API failures."""


class WeatherTimeoutError(WeatherApiError):
    """Raised when the OpenWeatherMap API does not respond in time."""


class CityNotFoundError(WeatherApiError):
    """Raised when the requested city cannot be found."""


class InvalidApiKeyError(WeatherApiError):
    """Raised when the OpenWeatherMap API key is missing or invalid."""


GET_CURRENT_WEATHER_TOOL: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": (
            "Get the current weather conditions for a city anywhere in the world, "
            "including temperature, humidity, wind, and a short description."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": (
                        "City name, optionally with country code "
                        "(e.g. 'London,UK' or 'Tokyo')."
                    ),
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


def get_current_weather(city: str, *, api_key: str) -> dict[str, Any]:
    """
    Fetch current weather for a city from OpenWeatherMap.

    Args:
        city: City name to look up.
        api_key: Valid OpenWeatherMap API key.

    Returns:
        Normalized weather payload suitable for tool responses.

    Raises:
        WeatherTimeoutError: On request timeout.
        CityNotFoundError: When the city is not found (HTTP 404).
        InvalidApiKeyError: When the API key is invalid (HTTP 401).
        WeatherApiError: For other API or transport failures.
    """
    if not city.strip():
        raise WeatherApiError("City name must not be empty.")

    params = {
        "q": city.strip(),
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(
            OPENWEATHER_BASE_URL,
            params=params,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except Timeout as exc:
        logger.exception("OpenWeatherMap request timed out for city=%r", city)
        raise WeatherTimeoutError(
            f"OpenWeatherMap request timed out after {DEFAULT_TIMEOUT_SECONDS}s."
        ) from exc
    except RequestsConnectionError as exc:
        logger.exception("OpenWeatherMap connection failed for city=%r", city)
        raise WeatherApiError("Unable to connect to OpenWeatherMap.") from exc
    except RequestException as exc:
        logger.exception("OpenWeatherMap request failed for city=%r", city)
        raise WeatherApiError("OpenWeatherMap request failed.") from exc

    if response.status_code == 404:
        logger.warning("City not found: %r", city)
        raise CityNotFoundError(f"City not found: {city!r}")

    if response.status_code == 401:
        logger.error("Invalid OpenWeatherMap API key.")
        raise InvalidApiKeyError("Invalid OpenWeatherMap API key.")

    if response.status_code == 403:
        logger.error("OpenWeatherMap API key forbidden or subscription inactive.")
        raise InvalidApiKeyError("OpenWeatherMap API key is not authorized.")

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.exception(
            "OpenWeatherMap returned HTTP %s for city=%r",
            response.status_code,
            city,
        )
        raise WeatherApiError(
            f"OpenWeatherMap returned HTTP {response.status_code}."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        logger.exception("OpenWeatherMap returned invalid JSON for city=%r", city)
        raise WeatherApiError("OpenWeatherMap returned invalid JSON.") from exc

    return _normalize_weather_payload(payload)


def _normalize_weather_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Map raw OpenWeatherMap JSON into a compact, LLM-friendly structure."""
    weather_items = payload.get("weather") or [{}]
    main = payload.get("main") or {}
    wind = payload.get("wind") or {}
    sys_info = payload.get("sys") or {}

    return {
        "city": payload.get("name"),
        "country": sys_info.get("country"),
        "description": weather_items[0].get("description"),
        "temperature_celsius": main.get("temp"),
        "feels_like_celsius": main.get("feels_like"),
        "humidity_percent": main.get("humidity"),
        "wind_speed_mps": wind.get("speed"),
        "cloudiness_percent": (payload.get("clouds") or {}).get("all"),
    }
