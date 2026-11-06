from tinyagent.planning import ReAct
from tinyagent.types import Response


def test_react_parse_extracts_thought_and_action():
    response = Response(
        content='THOUGHT: Search now.\nACTION:\n{"tool": "lookup", "kwargs": {}}'
    )
    parsed = ReAct().parse(response)
    assert parsed.reasoning == "Search now."
    assert parsed.content == '{"tool": "lookup", "kwargs": {}}'


def test_missing_action_preserves_plain_answer_and_none_content():
    assert ReAct().parse(Response(content="Here is the answer.")).content == (
        "Here is the answer."
    )
    assert ReAct().parse(Response(content=None)).content == ""
