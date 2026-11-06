"""Run a small retrieval-augmented agent; first install embeddinggemma in Ollama."""

from tinyagent import LLM, EmbeddingModel, RAGMemory, TinyAgent

documents = [
    "Planning selects the next step an agent should take.",
    "Memory preserves conversation context between model calls.",
    "Tools let an agent invoke ordinary Python functions.",
    "A trajectory records the thoughts, actions, and observations of a run.",
    "Evaluation compares agent outputs with expected results or checks.",
]

agent = TinyAgent.native(
    LLM(model="gemma4:e4b"),
    memory=RAGMemory(EmbeddingModel("embeddinggemma"), documents),
)
print(agent.run("How can I inspect what an agent did during a task?"))
