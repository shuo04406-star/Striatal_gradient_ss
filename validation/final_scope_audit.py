#!/usr/bin/env python3
"""Final read-only audit for CURRENT64 Steps 6--10 and R figure artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


ROOT_VALUE = os.environ.get("STRIATAL_CURRENT64_ROOT")
if not ROOT_VALUE:
    raise RuntimeError("Set STRIATAL_CURRENT64_ROOT to the private canonical-rerun root")
ROOT = Path(ROOT_VALUE).resolve()
OUT_TSV = ROOT / "16_VALIDATION" / "final_completion_audit.tsv"
OUT_JSON = ROOT / "16_VALIDATION" / "final_completion_audit.json"
NOW = datetime.now().astimezone()
LOG = ROOT / "17_LOGS" / f"final_completion_audit_{NOW.strftime('%Y%m%dT%H%M%S%z')}.log"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


checks: list[dict[str, str]] = []


def check(name: str, condition: bool, detail: str) -> None:
    checks.append({"Check": name, "Status": "PASS" if condition else "FAIL", "Detail": detail})


def required_rows(relative: str, expected: int) -> list[dict[str, str]]:
    path = ROOT / relative
    exists = path.is_file()
    rows = read_tsv(path) if exists else []
    check(relative, exists and len(rows) == expected, f"expected_rows={expected}; observed_rows={len(rows)}")
    return rows


def latest_log(stem: str) -> Path | None:
    candidates = sorted((ROOT / "17_LOGS").glob(f"{stem}_*.log"))
    return candidates[-1] if candidates else None


def parse_figure_log(path: Path) -> tuple[str, dict[Path, str]]:
    script_hash = ""
    outputs: dict[Path, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if fields[0] == "script_sha256" and len(fields) >= 2:
            script_hash = fields[1]
        elif fields[0] == "output" and len(fields) >= 3:
            outputs[Path(fields[1])] = fields[2]
    return script_hash, outputs


def main() -> int:
    if OUT_TSV.exists() or OUT_JSON.exists():
        raise RuntimeError("Default no-overwrite policy: final audit output already exists")

    required_rows("06_REWARD_STAGE/reward_stage_subject_level.tsv", 1280)
    required_rows("06_REWARD_STAGE/reward_stage_descriptives.tsv", 40)
    required_rows("09_INDIVIDUAL_SLOPES/subject_stage_slopes_long.tsv", 128)
    required_rows("09_INDIVIDUAL_SLOPES/subject_stage_slopes_wide.tsv", 64)

    qc = required_rows("09_INDIVIDUAL_SLOPES/slope_QC.tsv", 128)
    check(
        "slope_QC_all_pass",
        all(row["pass_structural_qc"] == "True" and row["is_na"] == "False" and row["is_inf"] == "False" for row in qc),
        "128/128 stage slopes structurally valid",
    )

    mixed = required_rows("07_STAGE_GRADIENT_MODELS/diagnostics/mixed_model_diagnostics.tsv", 6)
    check(
        "mixed_models_converged_with_retained_boundary_warning",
        all(row["converged"] == "True" for row in mixed)
        and all(row["warnings"] == "The MLE may be on the boundary of the parameter space." for row in mixed),
        "6 converged fits; 6 retained boundary warnings",
    )

    gee = required_rows("08_CATEGORICAL_SHAPE/GEE_omnibus_tests.tsv", 1)
    ols = required_rows("08_CATEGORICAL_SHAPE/clustered_OLS_omnibus_tests.tsv", 1)
    check("categorical_GEE_locked", abs(float(gee[0]["p"]) - 0.0017529515982185663) < 1e-15, f"p={gee[0]['p']}")
    check("categorical_clustered_OLS_locked", abs(float(ols[0]["p"]) - 0.0079000467945081303) < 1e-15, f"p={ols[0]['p']}")

    brain = required_rows("10_BRAIN_BEHAVIOR/brain_behavior_results.tsv", 4)
    check("brain_behavior_available_case_N", [int(row["N_total"]) for row in brain] == [57, 58, 60, 59], "N=[57,58,60,59]")

    validation = required_rows("16_VALIDATION/stage_analysis_historical_vs_canonical.tsv", 16)
    allowed = {"MATCH", "NUMERIC_TOLERANCE", "EXPECTED_SIGN_REVERSAL", "MODEL_SPECIFICATION_DIFFERENCE", "OTHER"}
    observed = {row["Interpretation"] for row in validation}
    check("historical_classifications_locked", observed.issubset(allowed), f"observed={sorted(observed)}")

    source_counts = {
        "Fig_stage_gradient_profiles.tsv": 40,
        "Fig_individual_stage_slopes.tsv": 192,
        "Fig_stage_difference_profile.tsv": 20,
        "Fig_brain_behavior.tsv": 874,
        "figure_statistics.tsv": 14,
    }
    for name, expected in source_counts.items():
        required_rows(f"14_R_FIGURES/source_data/{name}", expected)

    stems = {
        "Fig_stage_gradient_profiles": "01_reward_stage_profiles.R",
        "Fig_individual_stage_slopes": "02_subject_slopes.R",
        "Fig_stage_difference_profile": "03_stage_difference_profile.R",
        "Fig_brain_behavior": "04_brain_behavior.R",
    }
    for stem, script_name in stems.items():
        script = ROOT / "14_R_FIGURES" / "scripts" / script_name
        log = latest_log(stem)
        script_hash, logged_outputs = parse_figure_log(log) if log else ("", {})
        expected_outputs = {
            ROOT / "14_R_FIGURES" / "png" / f"{stem}.png",
            ROOT / "14_R_FIGURES" / "pdf" / f"{stem}.pdf",
            ROOT / "14_R_FIGURES" / "svg" / f"{stem}.svg",
        }
        output_ok = expected_outputs == set(logged_outputs)
        output_ok = output_ok and all(path.is_file() and path.stat().st_size > 1000 and sha256(path) == logged_outputs[path] for path in expected_outputs)
        check(
            f"{stem}_hashes",
            bool(log) and script_hash == sha256(script) and output_ok,
            f"latest_log={log.name if log else 'MISSING'}; script_and_three_output_hashes_verified={output_ok}",
        )

    final_tables = sorted((ROOT / "15_FINAL_TABLES").glob("Table_*.tsv"))
    check("final_tables", len(final_tables) == 9 and all(path.stat().st_size > 0 for path in final_tables), f"count={len(final_tables)}")

    stat_log = latest_log("stage_analysis_steps6_to10")
    stat_script = ROOT / "scripts" / "python" / "03_run_stage_analyses.py"
    stat_text = stat_log.read_text(encoding="utf-8") if stat_log else ""
    check(
        "statistics_resume_verification",
        bool(stat_log)
        and f"SHA256={sha256(stat_script)}" in stat_text
        and "CURRENT64 Steps 6-10 statistical stage completed" in stat_text,
        f"latest_log={stat_log.name if stat_log else 'MISSING'}",
    )

    r_session = ROOT / "17_LOGS" / "R_sessionInfo.txt"
    r_text = r_session.read_text(encoding="utf-8") if r_session.is_file() else ""
    check("R_sessionInfo", "R version 4.5.1" in r_text and "ggplot2_3.5.2" in r_text and "svglite_2.2.1" in r_text, "R and plotting packages recorded")

    result_dirs = [ROOT / name for name in ("06_REWARD_STAGE", "07_STAGE_GRADIENT_MODELS", "08_CATEGORICAL_SHAPE", "09_INDIVIDUAL_SLOPES", "10_BRAIN_BEHAVIOR", "14_R_FIGURES", "15_FINAL_TABLES", "16_VALIDATION", "17_LOGS")]
    forbidden = re.compile(r"normative|leave.?one.?out|rms.?deviation|profile.?similarity|abnormal.?burden|bin10.*(?:post|roi|subfamily|decomposition)", re.I)
    forbidden_paths = [str(path.relative_to(ROOT)) for directory in result_dirs for path in directory.rglob("*") if forbidden.search(path.name)]
    check("prohibited_modules_absent", not forbidden_paths, f"matches={forbidden_paths}")

    summary = ROOT / "00_README" / "STAGE_ANALYSIS_RERUN_SUMMARY.md"
    manifest = ROOT / "16_VALIDATION" / "final_completion_manifest.json"
    check("final_summary_and_manifest", summary.is_file() and manifest.is_file(), "summary and completion manifest present")

    overall = all(row["Status"] == "PASS" for row in checks)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Check", "Status", "Detail"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(checks)
    payload = {
        "completed_at": NOW.isoformat(),
        "overall_status": "PASS" if overall else "FAIL",
        "checks_passed": sum(row["Status"] == "PASS" for row in checks),
        "checks_total": len(checks),
        "normative_deviation_run": False,
        "bin10_posthoc_or_decomposition_run": False,
        "audit_tsv_sha256": sha256(OUT_TSV),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log_lines = [
        f"time\t{NOW.isoformat()}",
        f"script\t{Path(__file__).resolve()}",
        f"script_sha256\t{sha256(Path(__file__).resolve())}",
        f"python\t{sys.version.replace(chr(10), ' ')}",
        f"overall_status\t{payload['overall_status']}",
        f"checks\t{payload['checks_passed']}/{payload['checks_total']}",
        f"output\t{OUT_TSV}\t{sha256(OUT_TSV)}",
        f"output\t{OUT_JSON}\t{sha256(OUT_JSON)}",
    ]
    LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"LOG_PATH={LOG}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
