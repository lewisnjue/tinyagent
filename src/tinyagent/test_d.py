
from .llm import Response
from .tools import NativeTools, Tools


def multiply(a: str, b: str) -> float:
    return float(a) * float(b)


registry = {"multiply": multiply}
tools = Tools()
tools.add_tool(
    name="multiply",
    func=multiply,
    description="Multiplies two numbers: multiply(a: str, b: str)",
)

print(tools.prompt)
tool_schema = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    },
}


tools = Tools()
tools.add_tool("multiply", multiply, "Multiplies two numbers: multiply(a: str, b: str)")

# Fake LLM response
response = Response(content='{"tool": "multiply", "kwargs": {"a": "3.1", "b": "6.5"} }')

# Parse into a tool call
response = tools.parse(response)

# Execute the tool
tools.execute(response)
# Gemma 3 12B (no native thinking or tool calling)
llm = LLM(model="gemma3:12b")

# Register tool
tools = Tools()
tools.add_tool(
    name="multiply",
    func=multiply,
    description="Multiplies two numbers: multiply(a: str, b: str)",
)

# Memory
memory = Memory()

# Initialize Agent
agent = TinyAgent(llm=llm, memory=memory, tools=tools)
agent.run("What is 5.1 times 7.3?")
# Tools
tools = NativeTools()
tools.add_tool("multiply", multiply)

# Memory
memory = Memory()

# Initialize Agent
agent = TinyAgent(llm=llm, memory=memory, tools=tools)
agent.run("What is 5.1 times 7.3?")
