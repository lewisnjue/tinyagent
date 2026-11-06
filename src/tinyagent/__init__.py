"""A small, dependency-free framework for learning how LLM agents work."""

from tinyagent.agent import TinyAgent
from tinyagent.embeddings import EmbeddingModel
from tinyagent.evaluator import Benchmark, Evaluator
from tinyagent.llm import LLM
from tinyagent.memory import Memory, RAGMemory, SummarizationMemory, TrimmingMemory
from tinyagent.planning import NativeReAct, ReAct
from tinyagent.tools import NativeTools, Tools
from tinyagent.trajectory import Trajectory

__version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "EmbeddingModel",
    "Evaluator",
    "LLM",
    "Memory",
    "NativeReAct",
    "NativeTools",
    "RAGMemory",
    "ReAct",
    "SummarizationMemory",
    "TinyAgent",
    "Tools",
    "Trajectory",
    "TrimmingMemory",
]
