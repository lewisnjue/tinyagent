from collections.abc import Callable
import inspect
from typing import Any
from .llm import Response
import json


class Tools:
    """Tool registry for the Agent."""

    def __init__(self, requires_approval: list[str] = []):
        """Initialize and select tools that require approval before execution."""
        self.registry = {}
        self.requires_approval = requires_approval

    def parse(self, response: Response) -> Response:
        """Parse a JSON tool call from text."""
        text = response.content

        if '"tool":' in text or '"tool:"' in text:
            start, end = text.find("{"), text.rfind("}") + 1
            tool_call = json.loads(text[start:end])

            # Add the parsed tool call to the response
            return Response(
                content=response.content,
                reasoning=response.reasoning,
                tool_call=tool_call,
            )

        return response

    def observation(self, result: str) -> tuple[str, str]:
        """Return the observation as a user."""
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """The `TinyAgent`'s stopping mechanism."""
        if not response.tool_call:
            return True
        if response.tool_call["tool"] == "final_answer":
            response.content = response.tool_call.get("kwargs", "")
            return True
        return False

    def execute(self, response: Response) -> Any:
        """Run a registered tool.

        Arguments:
            tool_call: A parsed tool call dict with "tool" and "kwargs" keys.
        """
        tool_call = response.tool_call
        name, kwargs = tool_call["tool"], tool_call.get("kwargs", {})

        # Human-in-the-loop: ask before running dangerous tools
        if name in self.registry and name in self.requires_approval:
            response = input(f"Allow {name}? [y/N] ").strip().lower()
            if response not in ("y", "yes"):
                return f"Tool '{name}' was denied by the user."

        # Handle registered tools
        if name in self.registry:
            tool_func = self.registry[name]["function"]
            return tool_func(**kwargs)

        return f"Tool '{name}' not found."

    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a tool that the Agent can use.

        Arguments:
            name: The name of the tool.
            func: The function implementing the tool.
            description: A description of the tool.
        """
        self.registry[name] = {"function": func, "description": description}

    @property
    def descriptions(self) -> str:
        """Get descriptions of all registered tools."""
        return "\n".join(
            f"`{tool}`: {self.registry[tool]['description']}" for tool in self.registry
        )

    @property
    def prompt(self) -> str:
        return f"""
        # Tools
            
        If needed, you can only use the following tools to assist you 
        in completing tasks:
        
        {self.descriptions}
        
        To use a tool, respond with JSON: 
        {{"tool": "name", "kwargs": {{"param": "value"}}}}
        """

    @property
    def schemas(self) -> None:
        """Used only for native tool-calling."""
        return None


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


class NativeTools(Tools):
    """Tool registry using native function calling."""

    @property
    def schemas(self) -> list[dict]:
        """Return tool functions for native function calling."""
        return [tool_to_schema(tool["function"]) for tool in self.registry.values()]

    @property
    def prompt(self) -> str:
        """Empty because we don't need a prompt for native tool calling"""
        return ""

    def parse(self, response: Response) -> Response:
        """Parse a tool call."""
        # If there's no tool call, return the response as is
        if not response.tool_call:
            return response

        # Extract the tool name and arguments from the tool call
        args = response.tool_call["function"]["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        tool_call = {
            "tool": response.tool_call["function"]["name"],
            "kwargs": args,
        }

        # Add the parsed tool call to the response
        return Response(
            content=response.content,
            reasoning=response.reasoning,
            tool_call=tool_call,
        )

    def observation(self, result: str) -> tuple[str, str]:
        """Native tool results use the 'tool' role."""
        return "tool", str(result)

    def is_done(self, response: Response) -> bool:
        """No tool call means the `TinyAgent` is done."""
        return not response.tool_call
