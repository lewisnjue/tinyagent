import json
import subprocess
import time
from pathlib import Path

from safety import classify

from tinyagent import NativeTools

MAX_OUTPUT = 4000


def truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    """Keep the head and tail so errors at the end of output survive."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n...[{omitted} characters omitted]...\n{text[-half:]}"


class Terminal:
    def __init__(self, cwd: Path | None = None, log_path: Path | None = None) -> None:
        self.cwd = Path(cwd or Path.home()).resolve()
        self.log_path = log_path

    def _log(self, command: str, verdict: str, outcome: str) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cwd": str(self.cwd),
            "command": command,
            "verdict": verdict,
            "outcome": outcome,
        }
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry) + "\n")

    def _resolve(self, path: str) -> Path:
        return (self.cwd / Path(path).expanduser()).resolve()

    def run_command(self, command: str) -> str:
        """Run a bash command on the user's Linux machine and return its output."""
        verdict = classify(command)
        if verdict == "blocked":
            self._log(command, verdict, "refused")
            return (
                "Refused: this command is blocked for safety (destructive, sudo, "
                "or empty). Suggest the command to the user instead of running it."
            )
        if verdict == "confirm":
            print(f"\n[agent wants to run] {command}")
            if input("Run it? [y/N] ").strip().lower() not in ("y", "yes"):
                self._log(command, verdict, "denied")
                return "The user denied this command. Do not retry it."
        try:
            proc = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=self.cwd,
                capture_output=True,
                text=True,
                errors="replace",
                stdin=subprocess.DEVNULL,
            )
        except OSError as error:
            self._log(command, verdict, f"error: {error}")
            return f"Command failed to start: {error}"
        self._log(command, verdict, f"exit {proc.returncode}")
        return truncate(
            f"exit code: {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )

    def change_directory(self, path: str) -> str:
        """Change the working directory used by later commands."""
        target = self._resolve(path)
        if not target.is_dir():
            return f"Not a directory: {target}"
        self.cwd = target
        return f"Working directory is now {self.cwd}"

    def system_info(self) -> str:
        """Summarize OS, kernel, uptime, memory, and disk usage."""
        script = (
            "grep PRETTY_NAME /etc/os-release; uname -srm; uptime -p; "
            "free -h | head -2; df -h / | tail -1"
        )
        proc = subprocess.run(
            script,
            shell=True,
            executable="/bin/bash",
            cwd=self.cwd,
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
        )
        self._log("system_info", "safe", f"exit {proc.returncode}")
        output = f"{proc.stdout}\n{proc.stderr}".strip()
        return truncate(output) or "(no system information returned)"

    def list_directory(self, path: str = ".") -> str:
        """List files in a directory (first 100 entries)."""
        target = self._resolve(path)
        if not target.is_dir():
            return f"Not a directory: {target}"
        entries = sorted(
            target.iterdir(), key=lambda item: (not item.is_dir(), item.name)
        )
        lines = [
            f"{item.name}/" if item.is_dir() else item.name for item in entries[:100]
        ]
        more = f"\n...and {len(entries) - 100} more" if len(entries) > 100 else ""
        return "\n".join(lines) + more or "(empty)"

    def read_file(self, path: str, max_chars: int = MAX_OUTPUT) -> str:
        """Read a text file and return up to max_chars characters."""
        target = self._resolve(path)
        if not target.is_file():
            return f"Not a file: {target}"
        text = target.read_text(encoding="utf-8", errors="replace")
        return truncate(text, min(max_chars, MAX_OUTPUT))

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a text file after the user confirms."""
        target = self._resolve(path)
        print(f"\n[agent wants to write] {target} ({len(content)} chars)")
        if input("Write it? [y/N] ").strip().lower() not in ("y", "yes"):
            self._log(f"write_file {target}", "confirm", "denied")
            return "The user denied this write."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._log(f"write_file {target}", "confirm", "written")
        return f"Wrote {len(content)} characters to {target}"


def register_tools(tools: NativeTools, terminal: Terminal) -> None:
    for method in (
        terminal.run_command,
        terminal.change_directory,
        terminal.system_info,
        terminal.list_directory,
        terminal.read_file,
        terminal.write_file,
    ):
        tools.add_tool(method)
