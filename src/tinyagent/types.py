from dataclasses import dataclass
from typing import Any


@dataclass(init=False)
class Response:
    """A normalized response from an LLM call."""

    content: str
    reasoning: str | None
    tool_calls: list[dict]
    metadata: dict | None
    parse_error: str | None

    def __init__(
        self,
        content: str | None = "",
        reasoning: str | None = None,
        tool_calls: list[dict] | None = None,
        metadata: dict | None = None,
        parse_error: str | None = None,
        *,
        tool_call: dict | None = None,
    ) -> None:
        self.content = content or ""
        self.reasoning = reasoning
        self.tool_calls = list(tool_calls or [])
        if tool_call is not None and not self.tool_calls:
            self.tool_calls = [tool_call]
        self.metadata = metadata
        self.parse_error = parse_error

    @property
    def tool_call(self) -> dict | None:
        """Return the first tool call for single-call convenience."""
        return self.tool_calls[0] if self.tool_calls else None

    @tool_call.setter
    def tool_call(self, value: dict | None) -> None:
        self.tool_calls = [value] if value is not None else []


@dataclass
class Step:
    """A single step in an agent's trajectory."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None
    stopped: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "answer": self.answer,
            "metadata": self.metadata,
            "stopped": self.stopped,
        }
