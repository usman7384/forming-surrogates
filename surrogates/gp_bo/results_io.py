"""Write a results JSON in the project's results-JSON schema (task, title, command, git, data, metrics, figures, notes, created), with the exact command that produced it."""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "no-git"


def command_line(module: str, argv: list[str] | None) -> str:
    args = sys.argv[1:] if argv is None else argv
    return " ".join(["python -m", module, *map(str, args)])


def write_result(path: str | Path, *, task: str, title: str, command: str, data: dict, metrics: dict,
                 figures: list[str] | None = None, notes: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(task=task, title=title, command=command, git=git_sha(), data=data, metrics=metrics,
               figures=[str(f).replace("\\", "/") for f in (figures or [])], notes=notes, created=dt.datetime.now().isoformat(timespec="seconds"))
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2, default=float)
    return path
