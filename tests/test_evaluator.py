from conftest import FakeLLM

from tinyagent.agent import TinyAgent
from tinyagent.evaluator import Benchmark, Evaluator, format_report
from tinyagent.scorers import exact_match_scorer
from tinyagent.types import Response


def test_errors_are_isolated_and_tokens_are_aggregated():
    agents = iter(
        [
            TinyAgent.native(
                FakeLLM(
                    [
                        Response(
                            content="J",
                            metadata={"prompt_tokens": 4, "completion_tokens": 2},
                        )
                    ]
                )
            ),
            TinyAgent.native(FakeLLM([RuntimeError("endpoint down")])),
        ]
    )
    benchmark = Benchmark(
        "choices",
        [{"task": "first", "expected": "J"}, {"task": "second", "expected": "A"}],
        exact_match_scorer,
    )
    result = Evaluator(lambda: next(agents)).run(benchmark, verbose=False)
    assert result["pass_rate"] == 0.5
    assert result["errors"] == 1
    assert result["total_tokens"] == {
        "prompt": 4,
        "completion": 2,
    }
    assert result["results"][1]["status"] == "error"
    assert "endpoint down" in format_report(result)


def test_max_steps_is_not_scored_as_an_answer():
    from tinyagent.tools import NativeTools

    tools = NativeTools()
    tools.add_tool(lambda: "result", name="lookup")
    agent = TinyAgent.native(
        FakeLLM(
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
        ),
        tools=tools,
        max_steps=1,
    )
    result = Evaluator(lambda: agent).run(
        Benchmark("limited", [{"task": "task", "expected": "A"}], exact_match_scorer),
        verbose=False,
    )
    assert result["results"][0]["status"] == "max_steps"
    assert result["pass_rate"] == 0.0
    assert result["max_steps_reached"] == 1
