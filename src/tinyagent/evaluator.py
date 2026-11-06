import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

Scorer: TypeAlias = Callable[[str, dict], bool | float]


@dataclass
class Benchmark:
    name: str
    examples: list[dict]
    scorer: Scorer


class Evaluator:
    """Run a fresh agent for each benchmark example and aggregate results."""

    def __init__(self, create_agent: Callable):
        self.create_agent = create_agent

    def run(
        self,
        benchmark: Benchmark,
        limit: int | None = None,
        verbose: bool = True,
    ) -> dict:
        examples = (
            benchmark.examples[:limit] if limit is not None else benchmark.examples
        )
        results = []
        token_totals = {"prompt": 0, "completion": 0}
        for index, example in enumerate(examples, start=1):
            started = time.perf_counter()
            result = {
                "task": example.get("task", ""),
                "expected": example.get("expected"),
                "prediction": "",
                "score": 0.0,
                "passed": False,
                "status": "ok",
                "steps": 0,
                "tokens": {"prompt": 0, "completion": 0},
                "seconds": 0.0,
                "error": None,
            }
            try:
                agent = self.create_agent()
                result["prediction"] = agent.run(result["task"]) or ""
                trajectory_run = agent.trajectory.runs[-1]
                steps = trajectory_run["steps"]
                result["steps"] = len(steps)
                for step in steps:
                    metadata = step.metadata or {}
                    result["tokens"]["prompt"] += metadata.get("prompt_tokens", 0) or 0
                    result["tokens"]["completion"] += (
                        metadata.get("completion_tokens", 0) or 0
                    )
                stopped = trajectory_run.get("stopped")
                if stopped == "max_steps":
                    result["status"] = "max_steps"
                else:
                    raw_score = benchmark.scorer(result["prediction"], example)
                    result["score"] = float(raw_score)
                    result["passed"] = result["score"] >= 1.0
            except Exception as error:
                result["status"] = "error"
                result["error"] = f"{type(error).__name__}: {error}"
            result["seconds"] = round(time.perf_counter() - started, 3)
            for key in token_totals:
                token_totals[key] += result["tokens"][key]
            results.append(result)
            if verbose:
                marker = "✓" if result["passed"] else "✗"
                task = " ".join(result["task"].split())
                print(
                    f"{marker} {index}/{len(examples)} {task[:72]}  "
                    f"{result['seconds']:.1f}s [{result['status']}]"
                )

        scores = [item["score"] for item in results]
        return {
            "name": benchmark.name,
            "n": len(results),
            "pass_rate": sum(scores) / len(scores) if scores else 0.0,
            "errors": sum(item["status"] == "error" for item in results),
            "max_steps_reached": sum(item["status"] == "max_steps" for item in results),
            "total_tokens": token_totals,
            "avg_steps": (
                sum(item["steps"] for item in results) / len(results)
                if results
                else 0.0
            ),
            "results": results,
        }


def format_report(result: dict) -> str:
    """Format an evaluator result as a compact human-readable report."""
    lines = [
        f"{result['name']}: {result['pass_rate']:.1%} ({result['n']} examples)",
        f"Errors: {result['errors']} | max steps: {result['max_steps_reached']} "
        f"| average steps: {result['avg_steps']:.2f}",
        "Tokens: "
        f"{result['total_tokens']['prompt']} prompt / "
        f"{result['total_tokens']['completion']} completion",
    ]
    for item in result["results"]:
        lines.append(
            f"- {item['status']}: {item['score']:.2f} — "
            f"{' '.join(item['task'].split())[:100]}"
        )
        if item["error"]:
            lines.append(f"  {item['error']}")
    return "\n".join(lines)


def save(result: dict, path: str | Path) -> None:
    """Write an evaluation result as indented JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
