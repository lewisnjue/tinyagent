from tinyagent.http import LLMError, post_json
from tinyagent.types import Response


class LLM:
    """Minimal OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "no_key",
        think: bool | str | None = False,
        temperature: float | None = None,
        timeout: float = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.think = think
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, messages: list[dict], tools: list | None = None) -> Response:
        """Generate a response from the configured chat-completions endpoint."""
        body = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if tools:
            body["tools"] = tools
        if self.think is not None:
            # Some OpenAI-compatible endpoints reject reasoning_effort.
            body["reasoning_effort"] = (
                "medium"
                if self.think is True
                else "none"
                if self.think is False
                else self.think
            )

        data = post_json(
            f"{self.base_url}/chat/completions",
            body,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            self.timeout,
        )
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(
                "The chat-completions response has no assistant message."
            ) from error

        usage = data.get("usage") or {}
        return Response(
            content=message.get("content") or "",
            reasoning=message.get("reasoning") or message.get("reasoning_content"),
            tool_calls=message.get("tool_calls") or [],
            metadata={
                "model": data.get("model", self.model),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )
