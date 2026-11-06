import argparse
import json
from pathlib import Path

from tinyagent.agent import TinyAgent
from tinyagent.evaluator import Benchmark, Evaluator, format_report, save
from tinyagent.llm import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tinyagent smoke benchmark.")
    parser.add_argument("--model", default="gemma4:e4b", help="Chat model name.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument("--mode", choices=("native", "text"), default="native")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, help="Limit the number of examples.")
    parser.add_argument(
        "--data",
        type=Path,
        help="Optional JSONL file with task and expected/check fields.",
    )
    parser.add_argument("--output", type=Path, help="Path for the JSON report.")
    return parser.parse_args()


def create_agent(args: argparse.Namespace) -> TinyAgent:
    llm = LLM(model=args.model, base_url=args.base_url, think=args.think)
    if args.mode == "native":
        return TinyAgent.native(llm)
    return TinyAgent.text(llm)


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error
            if not isinstance(example, dict) or not isinstance(
                example.get("task"), str
            ):
                raise ValueError(f"Line {line_number} of {path} needs a string task.")
            examples.append(example)
    return examples


def run_benchmark(benchmark: Benchmark) -> None:
    args = parse_args()
    if args.data:
        benchmark.examples = load_jsonl(args.data)
    result = Evaluator(lambda: create_agent(args)).run(benchmark, limit=args.limit)
    print(format_report(result))
    output = (
        args.output
        or Path("results") / f"{benchmark.name.lower().replace(' ', '')}.json"
    )
    save(result, output)
    print(f"Saved results to {output}")
