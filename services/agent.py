"""Groq-backed agent that orchestrates local tool-calling for weather queries."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Final

from groq import Groq
from groq.types.chat import ChatCompletionMessageParam

from tools.weather import (
    GET_CURRENT_WEATHER_TOOL,
    WeatherApiError,
    get_current_weather,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "llama-3.1-8b-instant"
MAX_TOOL_ITERATIONS: Final[int] = 5

ToolHandler = Callable[..., dict[str, Any]]


class WeatherAgent:
    """Runs a Groq chat loop with local weather tool execution."""

    def __init__(
        self,
        client: Groq,
        *,
        openweather_api_key: str,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._client = client
        self._openweather_api_key = openweather_api_key
        self._model = model
        self._tool_registry: dict[str, ToolHandler] = {
            "get_current_weather": self._handle_get_current_weather,
        }
        self._tools: list[dict[str, Any]] = [GET_CURRENT_WEATHER_TOOL]

    def _handle_get_current_weather(self, *, city: str) -> dict[str, Any]:
        return get_current_weather(city, api_key=self._openweather_api_key)

    def run(self, user_prompt: str, *, system_prompt: str | None = None) -> str:
        """
        Execute the tool-calling loop until the model returns a final answer.

        Args:
            user_prompt: End-user request.
            system_prompt: Optional system instructions for the model.

        Returns:
            Final natural-language response from the model.

        Raises:
            RuntimeError: If the loop exceeds the maximum iteration limit.
        """
        messages: list[ChatCompletionMessageParam] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
            logger.info("Groq completion request (iteration %d)", iteration)

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=self._tools,
                tool_choice="auto",
                temperature=1.0,
            )

            assistant_message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            logger.debug(
                "Groq response finish_reason=%r tool_calls=%s",
                finish_reason,
                bool(assistant_message.tool_calls),
            )

            if not assistant_message.tool_calls:
                content = assistant_message.content
                if not content:
                    raise RuntimeError("Model returned an empty response.")
                return content.strip()

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in assistant_message.tool_calls
                    ],
                }
            )

            for tool_call in assistant_message.tool_calls:
                tool_result = self._execute_tool_call(tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(tool_result),
                    }
                )

        raise RuntimeError(
            f"Tool-calling loop exceeded {MAX_TOOL_ITERATIONS} iterations."
        )

    def _execute_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Resolve and invoke a tool from the local registry."""
        function_name = tool_call.function.name
        handler = self._tool_registry.get(function_name)

        if handler is None:
            logger.error("Unknown tool requested by model: %s", function_name)
            return {
                "error": True,
                "message": f"Unknown tool: {function_name}",
            }

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            logger.exception(
                "Invalid JSON arguments for tool %s: %r",
                function_name,
                tool_call.function.arguments,
            )
            return {
                "error": True,
                "message": f"Invalid tool arguments: {exc}",
            }

        if not isinstance(arguments, dict):
            return {
                "error": True,
                "message": "Tool arguments must be a JSON object.",
            }

        logger.info("Executing tool %s with args=%s", function_name, arguments)

        try:
            result = handler(**arguments)
            logger.info("Tool %s completed successfully", function_name)
            return {"error": False, "data": result}
        except WeatherApiError as exc:
            logger.warning("Tool %s failed: %s", function_name, exc)
            return {"error": True, "message": str(exc)}
        except TypeError as exc:
            logger.exception("Tool %s received invalid arguments", function_name)
            return {"error": True, "message": f"Invalid arguments: {exc}"}
        except Exception:
            logger.exception("Unexpected error executing tool %s", function_name)
            return {
                "error": True,
                "message": "An unexpected error occurred while executing the tool.",
            }
