from conftest import FakeLLM

from tinyagent.memory import (
    RAGMemory,
    SummarizationMemory,
    TrimmingMemory,
)
from tinyagent.types import Response


def test_trimming_keeps_whole_current_turn_and_tool_pair():
    memory = TrimmingMemory(max_turns=1)
    memory.add("system", "instructions")
    memory.add("user", "old task")
    memory.add(
        "assistant",
        "",
        tool_calls=[
            {
                "id": "old-call",
                "function": {"name": "lookup", "arguments": {}},
            }
        ],
    )
    memory.add_message(
        {"role": "tool", "tool_call_id": "old-call", "content": "old result"}
    )
    memory.add("user", "current task")
    memory.add("assistant", "done")
    messages = memory.get_messages()
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[1]["content"] == "current task"


def test_summarization_keeps_system_prompt():
    memory = SummarizationMemory(FakeLLM([Response(content="User studies memory.")]))
    memory.add("system", "Original instructions")
    memory.add("user", "I am studying memory.")
    memory.add("assistant", "Memory stores context.")
    memory.end_turn()
    messages = memory.get_messages()
    assert messages[0] == {"role": "system", "content": "Original instructions"}
    assert messages[1]["content"] == "Conversation summary: User studies memory."


class FakeEmbeddings:
    def __init__(self):
        self.inputs = []

    def embed(self, text):
        self.inputs.append(text)
        if isinstance(text, list):
            return [[1.0, 0.0], [0.0, 1.0]]
        return [1.0, 0.0]


def test_rag_batches_documents_and_augments_latest_task_only():
    model = FakeEmbeddings()
    memory = RAGMemory(model, ["relevant", "unrelated"], top_k=1)
    memory.add("user", "question")
    memory.add("user", "OBSERVATION: result")
    messages = memory.get_messages()
    assert model.inputs[0] == ["relevant", "unrelated"]
    assert model.inputs[1] == "question"
    assert "Relevant context:\nrelevant" in messages[0]["content"]
    assert memory.messages[0]["content"] == "question"
    assert messages[1]["content"] == "OBSERVATION: result"
    assert RAGMemory._cosine([0.0], [0.0]) == 0.0
