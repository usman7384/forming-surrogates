"""Write a results JSON in the project's results-JSON schema (task, title, command, git, data, metrics, figures, notes, created); every metric goes through here."""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

from common.data import REPO


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "no-git"


def write_result(path, task: str, title: str, command: str, data: dict, metrics: dict,
                 figures: list[str] | None = None, notes: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"task": task, "title": title, "command": command, "git": git_sha(), "data": data,
           "metrics": metrics, "figures": figures or [], "notes": notes,
           "created": dt.datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(doc, indent=1))
    return path
