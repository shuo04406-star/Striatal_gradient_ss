#!/usr/bin/env python
"""Rebuild the numerical tables and plots in a new results directory."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

sys.dont_write_bytecode = True

import pandas as pd

from followup_audit import run_audit
from followup_common import Inputs, sha256
from followup_phase1 import run_phase1
from followup_phase2 import run_phase2
from followup_phase3_4 import run_phase3, run_phase4

HERE = Path(__file__).resolve().parent


def fresh_directory(path):
    """Reserve a new version if the requested name is already in use."""
    for version in range(1, 1000):
        candidate = path if version == 1 else path.with_name(f"{path.name}_v{version:02d}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"No versioned directory available for {path}")


def plot(rscript, out):
    env = os.environ.copy()
    # The installed Windows cli DLL reads this variable during process cleanup.
    if os.name == "nt" and not env.get("PROCESSOR_ARCHITECTURE"):
        arch = {"win-amd64": "AMD64", "win-arm64": "ARM64", "win32": "x86"}
        env["PROCESSOR_ARCHITECTURE"] = arch[sysconfig.get_platform()]
    command = [rscript, "--vanilla"]
    for fmt in ("png", "pdf"):
        subprocess.run([*command, str(HERE/"plot_followup.R"), str(out), fmt], check=True, env=env)
    subprocess.run([*command, str(HERE/"check_figures.R"), str(out)], check=True, env=env)


def verify_inputs(out):
    sources = pd.read_csv(out/"00_INPUT_PROVENANCE.csv")
    changed = [row.full_path for row in sources.itertuples() if sha256(Path(row.full_path)) != row.sha256]
    if changed:
        raise RuntimeError(f"Read-only source changed during the analysis: {changed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    src = Inputs(args.analysis_root, args.project_root)
    out = fresh_directory(args.output_dir.resolve())
    scores = run_audit(src, out)
    run_phase1(src, out)
    run_phase2(src, out)
    run_phase3(src, out, scores)
    run_phase4(src, out)
    if not args.tables_only:
        plot(args.rscript, out)
    verify_inputs(out)
    print(f"Reproduction completed in {out}", flush=True)


if __name__ == "__main__":
    main()
