import pytest

from tinyagent.scorers import exact_match_scorer, extract_choice, programmatic_scorer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("J", "J"),
        ("(J)", "J"),
        ("The answer is J.", "J"),
        ("I think the answer is J", "J"),
        ("a: J", "J"),
        ("", None),
    ],
)
def test_extract_choice(text, expected):
    assert extract_choice(text) == expected


def test_exact_match_and_programmatic_scorer():
    assert exact_match_scorer("Answer: J", {"expected": "J"})
    assert not exact_match_scorer("", {"expected": "J"})
    assert programmatic_scorer("abc", {"check": lambda text: len(text) == 3})
