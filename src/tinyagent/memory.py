import json
import math
from typing import TYPE_CHECKING

from tinyagent.embeddings import EmbeddingModel
from tinyagent.types import Response

if TYPE_CHECKING:
    from tinyagent.llm import LLM


def is_task_message(message: dict) -> bool:
    content = message.get("content") or ""
    return message.get("role") == "user" and not content.startswith("OBSERVATION:")


class Memory:
    """Simple conversation memory with an optional system prompt."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.system: dict | None = None

    def add(
        self,
        role: str,
        content: str | None,
        tool_call: dict | None = None,
        tool_calls: list[dict] | None = None,
        **kwargs,
    ) -> None:
        message = {"role": role, "content": content or ""}
        calls = tool_calls or ([tool_call] if tool_call else [])
        if calls:
            message["tool_calls"] = [self._provider_tool_call(call) for call in calls]
        message.update(kwargs)
        self.add_message(message)

    @staticmethod
    def _provider_tool_call(call: dict) -> dict:
        if not isinstance(call, dict):
            return {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": "{}"},
            }
        source_function = call.get("function")
        if isinstance(source_function, dict):
            function = dict(source_function)
        else:
            function = {
                "name": call.get("tool", ""),
                "arguments": call.get("kwargs", {}),
            }
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            function["arguments"] = json.dumps(arguments)
        return {**call, "type": call.get("type", "function"), "function": function}

    def add_message(self, message: dict) -> None:
        stored = dict(message)
        if stored.get("role") == "system":
            self.system = stored
        else:
            self.messages.append(stored)

    def get_messages(self) -> list[dict]:
        """Return a shallow copy of the message list."""
        return ([dict(self.system)] if self.system else []) + [
            dict(message) for message in self.messages
        ]

    def end_turn(self) -> None:
        """Hook called after an agent run finishes."""

    def reset(self) -> None:
        """Clear conversation messages while retaining the system prompt."""
        self.messages.clear()


class TrimmingMemory(Memory):
    """Memory retaining the most recent complete user-task turns."""

    def __init__(self, max_turns: int = 2) -> None:
        super().__init__()
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns

    def add_message(self, message: dict) -> None:
        super().add_message(message)
        task_indexes = [
            index for index, item in enumerate(self.messages) if is_task_message(item)
        ]
        if len(task_indexes) > self.max_turns:
            del self.messages[: task_indexes[-self.max_turns]]


class SummarizationMemory(Memory):
    """Compress completed conversation turns while preserving system instructions."""

    def __init__(self, llm: "LLM") -> None:
        super().__init__()
        self.llm = llm
        self.summary = ""

    def get_messages(self) -> list[dict]:
        messages = super().get_messages()
        if self.summary:
            system_index = 1 if self.system else 0
            messages.insert(
                system_index,
                {"role": "system", "content": f"Conversation summary: {self.summary}"},
            )
        return messages

    def end_turn(self) -> None:
        if not self.messages:
            return
        conversation = []
        for message in self.messages:
            content = message.get("content")
            calls = message.get("tool_calls")
            if content:
                conversation.append(f"{message['role']}: {content}")
            if calls:
                compact = [
                    {
                        "name": call.get("function", {}).get("name", call.get("tool")),
                        "arguments": call.get("function", {}).get(
                            "arguments", call.get("kwargs")
                        ),
                    }
                    for call in calls
                ]
                conversation.append(f"tool calls: {compact}")
        if not conversation:
            self.messages.clear()
            return
        conversation_text = "\n".join(conversation)
        prompt = (
            "Update the conversation summary with these messages. Preserve useful "
            "facts and unresolved user preferences. Output only the summary.\n\n"
            f"Previous summary:\n{self.summary}\n\n"
            f"Conversation:\n{conversation_text}"
        )
        response: Response = self.llm.generate([{"role": "user", "content": prompt}])
        self.summary = response.content
        self.messages.clear()

    def reset(self) -> None:
        super().reset()
        self.summary = ""


class RAGMemory(Memory):
    """Conversation memory that retrieves relevant documents for task messages."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        documents: list[str],
        top_k: int = 3,
    ) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.embedding_model = embedding_model
        self.documents = list(documents)
        self.top_k = top_k
        self.embeddings = (
            embedding_model.embed(self.documents) if self.documents else []
        )

    def get_messages(self) -> list[dict]:
        messages = super().get_messages()
        for index in range(len(messages) - 1, -1, -1):
            if is_task_message(messages[index]):
                task = messages[index].get("content") or ""
                context = "\n".join(self.search(task))
                if context:
                    messages[index]["content"] = (
                        f"Relevant context:\n{context}\n\nTask:\n{task}"
                    )
                break
        return messages

    def search(self, query: str) -> list[str]:
        """Return the most similar documents for a query."""
        if not self.documents:
            return []
        query_embedding = self.embedding_model.embed(query)
        scores = [
            self._cosine(query_embedding, embedding) for embedding in self.embeddings
        ]
        ranked = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )
        return [self.documents[index] for index in ranked[: self.top_k]]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity, returning zero for zero-length vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
        return dot / norm if norm else 0.0
