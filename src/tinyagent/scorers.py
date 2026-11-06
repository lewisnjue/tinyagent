import re
from collections.abc import Callable
from typing import TypeAlias

Scorer: TypeAlias = Callable[[str, dict], bool | float]


def extract_choice(text: str) -> str | None:
    """Extract a multiple-choice answer without matching lowercase articles."""
    text = text.strip()
    match = re.fullmatch(r"\(?([A-J])\)?[.)]?", text)
    if match:
        return match.group(1)
    match = re.search(r"answer\s*(?:is|:)?\s*\(?([A-J])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    letters = re.findall(r"\b([A-J])\b", text)
    return letters[-1] if letters else None


def exact_match_scorer(prediction: str, example: dict) -> bool:
    """Compare the extracted answer choice to the expected choice."""
    expected = example["expected"]
    return extract_choice(prediction) == str(expected).upper()


def programmatic_scorer(prediction: str, example: dict) -> bool | float:
    """Run the example's callable check; evaluator captures raised errors."""
    return example["check"](prediction)
