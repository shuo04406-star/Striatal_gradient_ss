#!/usr/bin/env python3
"""Report whether the release's direct Python dependencies can be imported."""

from __future__ import annotations

import importlib
import sys


MODULES = ("numpy", "pandas", "scipy", "statsmodels", "patsy", "nibabel", "nilearn", "brainspace", "matplotlib", "yaml")


def main() -> int:
    missing: list[str] = []
    for module in MODULES:
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "unknown")
            print(f"PASS\t{module}\t{version}")
        except Exception as exc:  # report the import failure without exposing environment values
            missing.append(module)
            print(f"FAIL\t{module}\t{type(exc).__name__}")
    if missing:
        print("Missing dependencies: " + ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
