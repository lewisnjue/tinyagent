"""A small study companion demonstrating tinyagent's main components.

Run with native function calling (default):
    uv run python examples/study_buddy.py

Run with the text-based ReAct protocol instead:
    uv run python examples/study_buddy.py --mode text

Ollama must be serving the OpenAI-compatible API and have the selected model
available. Notes are stored in ~/.tinyagent/study_buddy_notes.json by default.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from tinyagent.agent import TinyAgent
from tinyagent.llm import LLM
from tinyagent.memory import Memory
from tinyagent.planning import NativeReAct, ReAct
from tinyagent.tools import NativeTools, Tools


class StudyNotes:
    """A tiny JSON-backed store exposed to the agent through tools."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {self.path}")
            self.notes: list[dict[str, str]] = data
        else:
            self.notes = []

    def remember_note(self, topic: str, note: str) -> str:
        """Save a concise learning note under its topic for future sessions."""
        entry = {
            "topic": topic.strip(),
            "note": note.strip(),
            "saved_at": datetime.now().astimezone().isoformat(timespec="minutes"),
        }
        self.notes.append(entry)
        self.path.write_text(
            json.dumps(self.notes, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return f"Saved a note about {entry['topic']} ({len(self.notes)} notes total)."

    def search_notes(self, query: str) -> str:
        """Search saved study notes by matching their topic or note text."""
        terms = query.casefold().split()
        matches = [
            entry
            for entry in self.notes
            if all(
                term in f"{entry['topic']} {entry['note']}".casefold()
                for term in terms
            )
        ]
        if not matches:
            return f"No saved notes matched {query!r}."

        return "\n".join(
            f"- {entry['topic']}: {entry['note']} (saved {entry['saved_at']})"
            for entry in matches[-5:]
        )


def plan_study_session(topic: str, minutes: int) -> str:
    """Create a short, timed study agenda for a topic."""
    if minutes < 5:
        return "Please choose at least 5 minutes for a useful study session."

    review = max(1, minutes // 5)
    practice = max(1, minutes * 3 // 5)
    reflect = minutes - review - practice
    return (
        f"{minutes}-minute session for {topic}:\n"
        f"1. Review prior notes ({review} min)\n"
        f"2. Work through an example or practice problem ({practice} min)\n"
        f"3. Write down one takeaway and one question ({reflect} min)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tinyagent study buddy.")
    parser.add_argument(
        "--mode",
        choices=("native", "text"),
        default="native",
        help="Use native function calling or the text-based ReAct protocol.",
    )
    parser.add_argument("--model", default="gemma4:e4b", help="Ollama model name.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="Ollama OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--notes-file",
        type=Path,
        default=Path.home() / ".tinyagent" / "study_buddy_notes.json",
        help="Path for persistent study notes.",
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="Task for the agent; defaults to a short Python study-session request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notes = StudyNotes(args.notes_file)

    if args.mode == "native":
        tools = NativeTools()
        planner = NativeReAct(max_steps=8)
    else:
        tools = Tools()
        planner = ReAct(max_steps=8)

    tools.add_tool("remember_note", notes.remember_note)
    tools.add_tool("search_notes", notes.search_notes)
    tools.add_tool("plan_study_session", plan_study_session)

    llm = LLM(model=args.model, base_url=args.base_url)
    agent = TinyAgent(llm=llm, memory=Memory(), tools=tools, planner=planner)
    task = " ".join(args.task) or (
        "I am learning Python functions. Make me a 30-minute study plan, "
        "save that I want to practice writing small functions, and summarize "
        "the plan and the note you saved."
    )
    print(agent.run(task))


if __name__ == "__main__":
    main()