import argparse
from pathlib import Path

from terminal_tools import Terminal, register_tools

from tinyagent import LLM, NativeTools, TinyAgent, TrimmingMemory
from tinyagent.http import LLMError

HERE = Path(__file__).parent

SYSTEM_PROMPT = """You are a terminal assistant on a Linux machine. Use the tools
to inspect the system and answer the user's question.

Rules:
- Prefer the specific tools for system information, directories, and files.
- Use run_command when the specific tools do not cover the task.
- Never run interactive programs. Use non-interactive forms such as
  `top -bn1`, `journalctl -n 50 --no-pager`, or
  `systemctl status NAME --no-pager`.
- Keep output small with limits such as `| head -n 30` or `-n` options.
- Never use sudo. If root is needed, tell the user the exact command to run.
- If a command is denied or refused, do not retry it; explain and suggest options.
- Base your answer on real command output. Do not invent results.
- Give a short, plain-language answer once you have what you need."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a terminal agent.")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verbose", action="store_true", help="Print thoughts, actions, observations."
    )
    parser.add_argument("--max-steps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = HERE / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    terminal = Terminal(log_path=logs / "commands.jsonl")
    tools = NativeTools()
    register_tools(tools, terminal)

    # None disables urllib's socket timeout while a local model loads or responds.
    llm = LLM(
        model=args.model,
        base_url=args.base_url,
        think=args.think,
        timeout=None,
    )
    agent = TinyAgent.native(
        llm,
        memory=TrimmingMemory(max_turns=3),
        tools=tools,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )
    agent.memory.add("system", SYSTEM_PROMPT)

    print("Terminal agent ready. Commands: /reset  /trace  /exit")
    while True:
        try:
            task = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not task:
            continue
        if task == "/exit":
            break
        if task == "/reset":
            agent.reset()
            print("Conversation cleared.")
            continue
        if task == "/trace":
            print(agent.trajectory)
            continue
        try:
            answer = agent.run(task)
        except LLMError as error:
            print(f"LLM error: {error}")
            continue
        print(f"\nagent> {answer or '(no answer: step limit reached, try /trace)'}")

    agent.trajectory.save(logs / "last_trajectory.json")
    print("\nBye. Trajectory saved to .terminal/logs/last_trajectory.json")


if __name__ == "__main__":
    main()
