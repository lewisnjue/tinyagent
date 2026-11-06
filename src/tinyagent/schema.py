import inspect
import types
from collections.abc import Callable
from typing import Literal, Union, get_args, get_origin, get_type_hints


def _json_type(annotation: object) -> dict:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = list(args)
        value_type = type(values[0]) if values else str
        json_type = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }.get(value_type, "string")
        return {"type": json_type, "enum": values}

    if origin in (list, tuple, set):
        item_type = args[0] if args else str
        return {"type": "array", "items": _json_type(item_type)}

    if origin in (types.UnionType, Union):
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])

    return {
        "type": {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            dict: "object",
        }.get(annotation, "string")
    }


def tool_to_schema(
    function: Callable, name: str | None = None, description: str | None = None
) -> dict:
    """Convert a Python function to an OpenAI-style tool schema."""
    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function)
    except (NameError, TypeError):
        hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []
    for parameter_name, parameter in signature.parameters.items():
        annotation = hints.get(parameter_name, parameter.annotation)
        properties[parameter_name] = _json_type(annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter_name)

    return {
        "type": "function",
        "function": {
            "name": name or function.__name__,
            "description": description or inspect.getdoc(function) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
