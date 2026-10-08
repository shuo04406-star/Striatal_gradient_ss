#!/usr/bin/env python3
"""Parse Python heredocs embedded in the retained Bash programs."""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
HEREDOC = re.compile(r"python3\s+-\s+<<'PY'[^\n]*\n(.*?)\nPY\s*$", re.DOTALL | re.MULTILINE)


def main() -> int:
    scripts = sorted((REPO / "src").rglob("*.sh"))
    if not scripts:
        raise RuntimeError("No Bash scripts found")
    for script in scripts:
        matches = HEREDOC.findall(script.read_text(encoding="utf-8"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one Python heredoc in {script}, found {len(matches)}")
        ast.parse(matches[0], filename=f"{script}:embedded-python")
        print(f"PASS\t{script.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
