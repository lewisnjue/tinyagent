"""Three small instruction-following checks; these are harness smoke tests."""

from benchmarks.common import run_benchmark
from tinyagent.evaluator import Benchmark
from tinyagent.scorers import programmatic_scorer

ifeval = Benchmark(
    name="IFeval",
    examples=[
        {
            "task": (
                "Write me a funny song with less than 10 sentences for a proposal "
                "to build a new playground at my local elementary school."
            ),
            # Approximation: punctuation characters are counted, not sentences.
            "check": lambda text: sum(1 for char in text if char in ".!?") < 10,
        },
        {
            "task": (
                "Write an ad copy for a new product, a digital photo frame that "
                "connects to your social media accounts and displays your photos. "
                "Respond with at most 150 words."
            ),
            "check": lambda text: len(text.split()) <= 150,
        },
        {
            "task": (
                "I am planning a trip to Japan, and I would like you to write an "
                "itinerary for my journey in a Shakespearean style. You are not "
                "allowed to use any commas in your response."
            ),
            "check": lambda text: "," not in text,
        },
    ],
    scorer=programmatic_scorer,
)


if __name__ == "__main__":
    run_benchmark(ifeval)
