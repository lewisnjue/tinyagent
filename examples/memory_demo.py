"""Compare the four memory strategies (LLM-backed ones contact the endpoint)."""

from tinyagent import (
    LLM,
    EmbeddingModel,
    Memory,
    RAGMemory,
    SummarizationMemory,
    TrimmingMemory,
)

llm = LLM(model="gemma4:e4b")
documents = [
    "Memory stores useful context across agent steps.",
    "Trimming memory keeps only a bounded number of recent task turns.",
    "Summarization memory compresses completed conversations.",
    "RAG retrieves documents relevant to the current task.",
]

memories = {
    "plain": Memory(),
    "trimming": TrimmingMemory(max_turns=2),
    "summarization": SummarizationMemory(llm),
    "RAG": RAGMemory(EmbeddingModel("embeddinggemma"), documents),
}
for name, memory in memories.items():
    memory.add("system", "You are a helpful assistant.")
    memory.add("user", "What does memory contribute to an agent?")
    if name == "summarization":
        memory.end_turn()
    print(f"{name}: {memory.get_messages()}")
