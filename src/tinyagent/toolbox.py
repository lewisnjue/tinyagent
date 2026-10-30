from dataclasses import dataclass
from collections.abc import Callable
import inspect
@dataclass
class Response:
    """Strucred response from LLM calls."""

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None


# Convert specific types to string descriptions
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool_to_schema(function: Callable) -> dict:
    """Convert a Python function to an OpenAI-style tool schema."""
    signature = inspect.signature(function)

    # Extract meatadata
    properties, required = {}, []
    for name, parameter in signature.parameters.items():
        properties[name] = {"type": TYPE_MAP.get(parameter.annotation, "string")}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    # Fill schema
    schema = {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": inspect.getdoc(function),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    return schema



@dataclass
class Step:
    """A single step in an agent's trajectory."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None
