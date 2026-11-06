import os

import pytest

from tinyagent import LLM, TinyAgent


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("TINYAGENT_INTEGRATION") != "1",
    reason="set TINYAGENT_INTEGRATION=1 to contact local Ollama",
)
def test_ollama_smoke():
    model = os.getenv("TINYAGENT_MODEL", "gemma4:e4b")
    answer = TinyAgent.native(LLM(model=model)).run("Reply with the single word OK.")
    assert "OK" in answer.upper()
