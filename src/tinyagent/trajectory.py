import json
from pathlib import Path

from tinyagent.types import Response, Step


class Trajectory:
    """Records agent execution as a sequence of runs."""

    def __init__(self) -> None:
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        self.runs.append({"query": query, "steps": [], "stopped": None})

    def add(
        self,
        response: Response,
        observation: str | None = None,
        stopped: str | None = None,
    ) -> None:
        step = Step(
            thought=response.reasoning or "",
            action=response.tool_call if observation is not None else None,
            observation=observation,
            answer=response.content if observation is None else None,
            metadata=response.metadata,
            stopped=stopped,
        )
        self.runs[-1]["steps"].append(step)
        if stopped:
            self.runs[-1]["stopped"] = stopped

    @property
    def usage(self) -> dict[str, int]:
        totals = {"prompt_tokens": 0, "completion_tokens": 0}
        for run in self.runs:
            for step in run["steps"]:
                metadata = step.metadata or {}
                for key in totals:
                    totals[key] += metadata.get(key, 0) or 0
        return totals

    def to_dict(self) -> dict:
        return {
            "runs": [
                {
                    **run,
                    "steps": [step.to_dict() for step in run["steps"]],
                }
                for run in self.runs
            ],
            "usage": self.usage,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def __str__(self) -> str:
        lines = []
        for index, run in enumerate(self.runs, start=1):
            lines.append(f"Run {index}: {run['query']}")
            for step_number, step in enumerate(run["steps"], start=1):
                if step.action:
                    lines.append(f"  {step_number}. ACTION {step.action}")
                if step.observation is not None:
                    lines.append(f"     OBSERVATION {step.observation}")
                if step.answer is not None:
                    lines.append(f"     ANSWER {step.answer}")
                if step.stopped:
                    lines.append(f"     STOPPED {step.stopped}")
        return "\n".join(lines)
