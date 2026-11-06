import inspect
import json
import re
import textwrap
from collections.abc import Callable
from typing import Any

from tinyagent.schema import tool_to_schema
from tinyagent.types import Response


class Tools:
    """Tool registry for text-based ReAct agents."""

    def __init__(self, requires_approval: list[str] | None = None) -> None:
        self.registry: dict[str, dict[str, Any]] = {}
        self.requires_approval = list(requires_approval or [])

    def parse(self, response: Response) -> Response:
        """Parse a JSON action, recording malformed model output for recovery."""
        text = response.content or ""
        if not re.search(r'"tool"\s*:', text):
            return response

        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                action, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(action, dict) and "tool" in action:
                response.tool_calls = [action]
                return response

        response.parse_error = "Invalid JSON action. Reply with a valid ACTION blob."
        return response

    def observation(self, result: str) -> tuple[str, str]:
        """Return a text-mode observation as a user message."""
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """Return whether the action ends the run; normalize final answers."""
        action = response.tool_call
        if not action:
            return True
        if action.get("tool") != "final_answer":
            return False
        # Normalize in place because TinyAgent returns response.content directly.
        answer = action.get("kwargs", "")
        if isinstance(answer, dict):
            answer = answer.get("answer") or next(
                (value for value in answer.values() if isinstance(value, str)), ""
            )
        response.content = str(answer)
        return True

    def execute(self, response: Response) -> str:
        """Execute a parsed tool action or return a recoverable error message."""
        action = response.tool_call or {}
        name = action.get("tool", "")
        kwargs = action.get("kwargs", {})
        if not isinstance(kwargs, dict):
            return f"Error: 'kwargs' must be an object, got {type(kwargs).__name__}."
        if name not in self.registry:
            available = ", ".join(sorted(self.registry)) or "(none)"
            return f"Tool '{name}' not found. Available tools: {available}."

        if name in self.requires_approval:
            answer = input(f"Allow {name}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                return f"Tool '{name}' was denied by the user."
        try:
            return str(self.registry[name]["function"](**kwargs))
        except TypeError as error:
            return f"Error calling '{name}': {error}"
        except Exception as error:
            return f"Tool '{name}' failed: {type(error).__name__}: {error}"

    def add_tool(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a callable; its signature and docstring describe the tool."""
        tool_name = name or func.__name__
        docstring = inspect.getdoc(func) or ""
        tool_description = description or (
            docstring.splitlines()[0] if docstring else ""
        )
        self.registry[tool_name] = {
            "function": func,
            "description": tool_description,
        }

    @property
    def descriptions(self) -> str:
        """Get each registered tool's signature and short description."""
        lines = []
        for name, tool in self.registry.items():
            signature = inspect.signature(tool["function"])
            lines.append(f"`{name}{signature}`: {tool['description']}")
        return "\n".join(lines)

    @property
    def schemas(self) -> None:
        """Text-based tools do not send function schemas to the model."""
        return None

    @property
    def prompt(self) -> str:
        return textwrap.dedent(
            f"""
            # Tools

            If needed, use only these tools:
            {self.descriptions}

            Return an action as valid JSON:
            {{"tool": "name", "kwargs": {{"param": "value"}}}}
            """
        ).strip()


class NativeTools(Tools):
    """Tool registry using native function calling."""

    @property
    def schemas(self) -> list[dict]:
        return [
            tool_to_schema(tool["function"], name=name, description=tool["description"])
            for name, tool in self.registry.items()
        ]

    @property
    def prompt(self) -> str:
        return ""

    def parse(self, response: Response) -> Response:
        """Normalize every native call while retaining its provider call ID."""
        actions = []
        for call in response.tool_calls:
            if not isinstance(call, dict):
                actions.append(
                    {
                        "tool": "",
                        "kwargs": {},
                        "error": "Invalid native tool call: expected an object.",
                    }
                )
                continue
            function = call.get("function", {})
            if not isinstance(function, dict):
                function = {}
            try:
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except (KeyError, json.JSONDecodeError, ValueError, TypeError) as error:
                actions.append(
                    {
                        "id": call.get("id"),
                        "tool": function.get("name", ""),
                        "kwargs": {},
                        "error": f"Invalid native tool arguments: {error}",
                    }
                )
                continue
            actions.append(
                {
                    "id": call.get("id"),
                    "tool": function.get("name", ""),
                    "kwargs": arguments,
                }
            )
        response.tool_calls = actions
        return response

    def observation(self, result: str, tool_call_id: str | None = None) -> dict:
        """Build a provider-compliant native tool result message."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id or "",
            "content": str(result),
        }

    def is_done(self, response: Response) -> bool:
        return not response.tool_calls
