from tinyagent.llm import LLM
from tinyagent.memory import Memory
from tinyagent.tools import NativeTools
from tinyagent.agent import TinyAgent
from tinyagent.planning import NativeReAct
import re

llm = LLM("gemma:4e4b", think=True)


def create_agent():
    """Create a new instance of TinyAgent"""
    return TinyAgent(
        llm=llm,
        memory=Memory(),
        tools=NativeTools(),
        planner=NativeReAct(),
    )


def exact_match_scorer(prediction: str, example: dict) -> bool:
    """Return True if the answer matches the prediction, False otherwise"""
    match = re.search(r"\b([A-J])\b", prediction.upper())
    return match.group(1) == example["expected"]


def programmatic_scorer(prediction: str, example: dict) -> bool:
    """Check a prediction against its related check."""
    return example["check"](prediction)
