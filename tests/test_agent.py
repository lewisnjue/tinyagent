from conftest import FakeLLM

from tinyagent.agent import TinyAgent
from tinyagent.tools import NativeTools
from tinyagent.types import Response


def test_one_step_plain_answer():
    agent = TinyAgent.native(FakeLLM([Response(content="hello")]))
    assert agent.run("say hello") == "hello"
    assert agent.trajectory.runs[0]["stopped"] is None


def test_native_and_text_protocols_return_the_same_answer():
    text = TinyAgent.text(
        FakeLLM(
            [
                Response(
                    content='ACTION:\n{"tool": "final_answer", "kwargs": "same answer"}'
                )
            ]
        )
    )
    native = TinyAgent.native(FakeLLM([Response(content="same answer")]))
    assert text.run("task") == native.run("task") == "same answer"


def test_text_tool_then_final_answer():
    final_action = '{"tool": "final_answer", "kwargs": "It stores context."}'
    llm = FakeLLM(
        [
            Response(
                content="THOUGHT: Look it up.\nACTION:\n"
                '{"tool": "lookup", "kwargs": {"topic": "memory"}}'
            ),
            Response(content=f"ACTION:\n{final_action}"),
        ]
    )
    agent = TinyAgent.text(llm)
    agent.tools.add_tool(lambda topic: f"Found {topic}", name="lookup")
    assert agent.run("what is memory?") == "It stores context."
    assert "OBSERVATION: Found memory" in str(llm.requests[1]["messages"])


def test_native_round_trip_preserves_call_id():
    llm = FakeLLM(
        [
            Response(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"topic": "memory"},
                        },
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"topic": "planning"},
                        },
                    },
                ],
            ),
            Response(content="Found memory."),
        ]
    )
    tools = NativeTools()
    tools.add_tool(lambda topic: f"Found {topic}", name="lookup")
    agent = TinyAgent.native(llm, tools=tools)
    assert agent.run("look up memory") == "Found memory."

    followup = llm.requests[1]["messages"]
    assistant_call = next(message for message in followup if message.get("tool_calls"))
    assert assistant_call["tool_calls"][0]["function"]["arguments"] == (
        '{"topic": "memory"}'
    )
    tool_results = [message for message in followup if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_results] == [
        "call-1",
        "call-2",
    ]
    assert [message["content"] for message in tool_results] == [
        "Found memory",
        "Found planning",
    ]


def test_invalid_native_arguments_become_tool_observation():
    llm = FakeLLM(
        [
            Response(
                tool_calls=[
                    {
                        "id": "bad-call",
                        "function": {"name": "lookup", "arguments": "{bad json"},
                    }
                ]
            ),
            Response(content="Recovered."),
        ]
    )
    tools = NativeTools()
    tools.add_tool(lambda: "unused", name="lookup")
    agent = TinyAgent.native(llm, tools=tools)
    assert agent.run("recover from the invalid call") == "Recovered."
    tool_result = next(
        message for message in llm.requests[1]["messages"] if message["role"] == "tool"
    )
    assert tool_result["tool_call_id"] == "bad-call"
    assert "Invalid native tool arguments" in tool_result["content"]


def test_max_steps_is_recorded_without_magic_answer():
    llm = FakeLLM(
        [
            Response(
                tool_calls=[
                    {
                        "id": "call-1",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ]
            )
        ]
    )
    tools = NativeTools()
    tools.add_tool(lambda: "result", name="lookup")
    agent = TinyAgent.native(llm, tools=tools, max_steps=1)
    assert agent.run("lookup") == ""
    assert agent.trajectory.runs[0]["stopped"] == "max_steps"
