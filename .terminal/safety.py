import re
import shlex

SAFE_COMMANDS = {
    "ls",
    "pwd",
    "whoami",
    "id",
    "date",
    "uptime",
    "uname",
    "hostname",
    "df",
    "du",
    "free",
    "lsblk",
    "lscpu",
    "lsusb",
    "lspci",
    "nproc",
    "ps",
    "ip",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "which",
    "echo",
    "lsb_release",
    "printenv",
    "sensors",
}

BLOCKED_PATTERNS = [
    r"\bsudo\b",
    r"\brm\s+(-\w+\s+)*(/|~|\$HOME|\*)(\s|$)",
    r"\brm\b.*--no-preserve-root",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\b.*\bof=/dev/",
    r">\s*/dev/sd",
    r":\(\)\s*\{",
    r"\b(shutdown|reboot|poweroff|halt)\b",
    r"\bchmod\s+-R\s+\S+\s+/(\s|$)",
    r"\b(curl|wget)\b.*\|\s*(ba)?sh",
]

SHELL_META = re.compile(r"[;&|<>`]|\$\(")


def classify(command: str) -> str:
    """Classify a shell command as safe, confirmation-required, or blocked."""
    text = command.strip()
    if not text:
        return "blocked"
    if any(re.search(pattern, text) for pattern in BLOCKED_PATTERNS):
        return "blocked"
    if SHELL_META.search(text):
        return "confirm"
    try:
        parts = shlex.split(text)
    except ValueError:
        return "confirm"
    return "safe" if parts and parts[0] in SAFE_COMMANDS else "confirm"
