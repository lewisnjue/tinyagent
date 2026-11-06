from typing import Literal

from tinyagent.tools import NativeTools, Tools
from tinyagent.types import Response


def test_text_actions_and_invalid_json_are_recoverable():
    tools = Tools()
    parsed = tools.parse(Response(content='{"tool" : "lookup", "kwargs": {}}'))
    assert parsed.tool_call == {"tool": "lookup", "kwargs": {}}

    invalid = tools.parse(Response(content='{"tool": "lookup",'))
    assert invalid.parse_error


def test_final_answer_normalizes_dict_and_string():
    tools = Tools()
    answer = Response(tool_call={"tool": "final_answer", "kwargs": {"answer": "done"}})
    assert tools.is_done(answer)
    assert answer.content == "done"

    answer = Response(tool_call={"tool": "final_answer", "kwargs": "finished"})
    assert tools.is_done(answer)
    assert answer.content == "finished"


def test_unknown_tool_lists_registered_names_and_bad_args_are_observations():
    def lookup(topic: str) -> str:
        return topic

    tools = Tools()
    tools.add_tool(lookup)
    assert "lookup" in tools.execute(
        Response(tool_call={"tool": "missing", "kwargs": {}})
    )
    assert "Error calling 'lookup'" in tools.execute(
        Response(tool_call={"tool": "lookup", "kwargs": {"wrong": 1}})
    )
    assert "must be an object" in tools.execute(
        Response(tool_call={"tool": "lookup", "kwargs": []})
    )


def test_approval_can_be_denied(monkeypatch):
    def delete_record(record_id: int) -> str:
        return str(record_id)

    monkeypatch.setattr("builtins.input", lambda _: "no")
    tools = Tools(requires_approval=["delete_record"])
    tools.add_tool(delete_record)
    assert "denied" in tools.execute(
        Response(tool_call={"tool": "delete_record", "kwargs": {"record_id": 1}})
    )


def test_native_schema_uses_registered_name_and_type_hints():
    def search(tags: list[str], category: str | None, level: Literal["a", "b"]):
        """Search records."""

    tools = NativeTools()
    tools.add_tool(search, name="record_search")
    schema = tools.schemas[0]["function"]
    assert schema["name"] == "record_search"
    assert schema["parameters"]["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert schema["parameters"]["properties"]["category"] == {"type": "string"}
    assert schema["parameters"]["properties"]["level"]["enum"] == ["a", "b"]


def test_empty_native_registry_has_no_schemas():
    assert NativeTools().schemas == []
