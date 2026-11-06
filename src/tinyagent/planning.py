import re
import textwrap

from tinyagent.types import Response


class ReAct:
    """Prompt and parser for text-based Reason-and-Act interactions."""

    @property
    def prompt(self) -> str:
        return textwrap.dedent(
            """
            # ReAct (Reason and Act)

            Complete exactly one step per reply using this format:
            THOUGHT: Briefly explain the next step.
            ACTION:
            {"tool": "tool_name", "kwargs": {"param": "value"}}

            The agent will return tool observations in a later message. Do not
            invent observations. Use only valid JSON with no trailing commas.

            Example tool step:
            THOUGHT: I should find the requested fact.
            ACTION:
            {"tool": "lookup", "kwargs": {"topic": "memory"}}

            After receiving the observation, finish with:
            THOUGHT: I have enough information.
            ACTION:
            {"tool": "final_answer", "kwargs": "Memory keeps useful context."}

            Use final_answer when all subtasks are complete. If you cannot follow
            this format, reply with your best plain-text answer.
            """
        ).strip()

    def parse(self, response: Response) -> Response:
        """Extract THOUGHT/ACTION, treating unformatted text as a final answer."""
        text = response.content or ""
        thought = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|OBSERVATION:|$)", text, re.S)
        action = re.search(r"ACTION:\s*(.+?)(?=THOUGHT:|OBSERVATION:|$)", text, re.S)
        response.reasoning = thought.group(1).strip() if thought else ""
        if action:
            response.content = action.group(1).strip()
        else:
            response.content = text.strip()
        return response


class NativeReAct(ReAct):
    """Planner for providers using native tool calling."""

    @property
    def prompt(self) -> str:
        return ""

    def parse(self, response: Response) -> Response:
        return response
