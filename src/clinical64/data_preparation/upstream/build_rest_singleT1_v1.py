#!/usr/bin/env python3
"""Build and audit a rest-only BIDS dataset with manually selected single T1w.

This program is intentionally conservative:
* source trees are read-only;
* files are selected by explicit BIDS filename rules;
* all source checks finish before a staging directory is created;
* the final output directory is published only after validation succeeds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE = Path(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"])
REST_SOURCE = WORKSPACE / "bids_build" / "03_bids_raw_rest_humor_for_fmriprep"
T1_QC_SOURCE = (
    WORKSPACE / "bids_build" / "03_bids_raw_task_humor_reconverted_v5_singleT1_QC"
)
T1_DATA_SOURCE = (
    WORKSPACE / "bids_build" / "03_bids_raw_task_humor_reconverted_v5_singleT1"
)
OUTPUT = WORKSPACE / "bids_build" / "03_bids_raw_rest_singleT1_v1"
LOG_DIR = WORKSPACE / "logs" / "build_rest_singleT1_v1"
COHORT_CONFIG = json.loads(Path(os.environ["STRIATAL_REST_BUILD_CONFIG"]).read_text(encoding="utf-8-sig"))
EXPECTED_SUBJECTS = COHORT_CONFIG["subjects"]
TWO_RUN_SUBJECT = COHORT_CONFIG["two_run_subject"]
T1_VERIFICATION_SUBJECT = COHORT_CONFIG["t1_verification_subject"]
DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class CopyPlan:
    source: Path
    relative_target: Path
    category: str
    modified_json: dict[str, Any] | None = None
    note: str = ""


class BuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def t1_stem(path: Path) -> str:
    if not path.name.endswith(".nii.gz"):
        raise BuildError(f"Not a .nii.gz file: {path}")
    return path.name[:-7]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def subject_dirs(root: Path) -> list[Path]:
    return sorted(
        (entry for entry in root.iterdir() if entry.is_dir() and entry.name.startswith("sub-")),
        key=lambda path: path.name,
    )


def read_selection_manifest() -> dict[str, dict[str, str]]:
    """Read all 70 actual selections, applying the final manual-review overlay."""
    manifest = T1_QC_SOURCE / "t1_selection_summary.csv"
    final_manifest = T1_QC_SOURCE / "final_singleT1_selection_log.csv"
    require(manifest.is_file(), f"Missing QC selection manifest: {manifest}")
    require(final_manifest.is_file(), f"Missing final QC selection manifest: {final_manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 70, f"QC selection manifest has {len(rows)} rows, expected 70")
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        subject = (row.get("subject") or "").strip()
        require(subject not in indexed, f"Duplicate subject in QC selection manifest: {subject}")
        row["_selection_manifest"] = str(manifest)
        indexed[subject] = row
    require(sorted(indexed) == EXPECTED_SUBJECTS, "QC manifest subject IDs are not the configured rest cohort")

    with final_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        final_rows = list(csv.DictReader(stream))
    require(len(final_rows) == 3, f"Final manual-review manifest has {len(final_rows)} rows, expected 3")
    final_subjects: set[str] = set()
    for final_row in final_rows:
        subject = (final_row.get("subject") or "").strip()
        require(subject in indexed, f"Unknown subject in final manual-review manifest: {subject}")
        require(subject not in final_subjects, f"Duplicate final manual-review subject: {subject}")
        require(
            (indexed[subject].get("needs_review") or "").strip().lower() == "yes",
            f"{subject}: final review overlay exists but initial manifest does not mark needs_review=yes",
        )
        kept_t1w = (final_row.get("kept_t1w") or "").strip()
        kept_json = (final_row.get("kept_json") or "").strip()
        require(kept_t1w and kept_json, f"{subject}: final manual-review selection is incomplete")
        indexed[subject]["selected_t1w"] = kept_t1w
        indexed[subject]["selected_json"] = kept_json
        indexed[subject]["needs_review"] = "resolved"
        indexed[subject]["_selection_manifest"] = str(final_manifest)
        final_subjects.add(subject)
    require(
        all((row.get("selected_t1w") or "").strip() and (row.get("selected_json") or "").strip() for row in indexed.values()),
        "At least one subject remains without a final selected T1w pair",
    )
    return indexed


def normalized_intended_target(subject: str, value: str) -> str:
    normalized = value.replace("\\", "/")
    if normalized.startswith("bids::"):
        normalized = normalized[6:]
    normalized = normalized.lstrip("/")
    if normalized.startswith("sub-"):
        return normalized
    return f"{subject}/{normalized}"


def inspect_sources() -> tuple[list[CopyPlan], list[dict[str, Any]], list[dict[str, Any]]]:
    require(REST_SOURCE.is_dir(), f"Missing rest source: {REST_SOURCE}")
    require(T1_QC_SOURCE.is_dir(), f"Missing singleT1 QC directory: {T1_QC_SOURCE}")
    require(T1_DATA_SOURCE.is_dir(), f"Missing resolved singleT1 data directory: {T1_DATA_SOURCE}")
    require(not OUTPUT.exists(), f"Output already exists; refusing to overwrite or merge: {OUTPUT}")

    rest_subjects = subject_dirs(REST_SOURCE)
    t1_subjects = subject_dirs(T1_DATA_SOURCE)
    require([p.name for p in rest_subjects] == EXPECTED_SUBJECTS, "Rest source does not contain exactly the configured rest cohort")
    require([p.name for p in t1_subjects] == EXPECTED_SUBJECTS, "singleT1 data source does not contain exactly the configured rest cohort")

    selection_rows = read_selection_manifest()
    plans: list[CopyPlan] = []
    t1_manifest: list[dict[str, Any]] = []
    rest_manifest_seed: list[dict[str, Any]] = []

    # Top-level ordinary BIDS files are an explicit, non-recursive whitelist.
    for source in sorted(REST_SOURCE.iterdir(), key=lambda path: path.name):
        if source.is_file() and "task-humor" not in source.name.lower():
            plans.append(CopyPlan(source, Path(source.name), "top_level"))

    top_names = {plan.relative_target.name for plan in plans if plan.category == "top_level"}
    require("dataset_description.json" in top_names, "Rest source lacks dataset_description.json")
    require("participants.tsv" in top_names, "Rest source lacks participants.tsv")
    with (REST_SOURCE / "dataset_description.json").open("r", encoding="utf-8-sig") as stream:
        json.load(stream)

    total_rest_nii = 0
    total_rest_json = 0
    special_names = COHORT_CONFIG["selected_t1_filenames"]

    for index, subject in enumerate(EXPECTED_SUBJECTS, start=1):
        rest_dir = REST_SOURCE / subject
        t1_dir = T1_DATA_SOURCE / subject / "anat"
        require(t1_dir.is_dir(), f"Missing anat directory in singleT1 source: {t1_dir}")

        t1_nii = sorted(t1_dir.glob("*_T1w.nii.gz"), key=lambda path: path.name)
        t1_json = sorted(t1_dir.glob("*_T1w.json"), key=lambda path: path.name)
        require(len(t1_nii) == 1, f"{subject}: expected exactly one T1w NIfTI, found {len(t1_nii)}")
        require(len(t1_json) == 1, f"{subject}: expected exactly one T1w JSON, found {len(t1_json)}")
        require(t1_stem(t1_nii[0]) == t1_json[0].stem, f"{subject}: T1w NIfTI/JSON basename mismatch")

        qc_row = selection_rows[subject]
        qc_nii = Path(qc_row["selected_t1w"])
        qc_json = Path(qc_row["selected_json"])
        require(qc_nii.resolve() == t1_nii[0].resolve(), f"{subject}: actual T1 differs from QC manifest")
        require(qc_json.resolve() == t1_json[0].resolve(), f"{subject}: actual T1 JSON differs from QC manifest")
        if subject in special_names:
            require(t1_nii[0].name == special_names[subject], f"{subject}: unexpected manually selected T1")

        for source in (t1_nii[0], t1_json[0]):
            plans.append(
                CopyPlan(source, Path(subject) / "anat" / source.name, "t1")
            )
        t1_manifest.append(
            {
                "subject": subject,
                "selection_manifest": qc_row["_selection_manifest"],
                "source_t1_nii": str(t1_nii[0]),
                "source_t1_json": str(t1_json[0]),
                "source_t1_nii_sha256": "",
                "source_t1_json_sha256": "",
                "output_t1_nii": str(OUTPUT / subject / "anat" / t1_nii[0].name),
                "output_t1_json": str(OUTPUT / subject / "anat" / t1_json[0].name),
                "output_t1_nii_sha256": "",
                "output_t1_json_sha256": "",
                "hash_identical": "",
            }
        )

        func_dir = rest_dir / "func"
        require(func_dir.is_dir(), f"{subject}: missing rest func directory")
        rest_files = sorted(
            (path for path in func_dir.iterdir() if path.is_file() and "task-rest" in path.name),
            key=lambda path: path.name,
        )
        require(rest_files, f"{subject}: no task-rest files found")
        require(not any("task-humor" in path.name.lower() for path in rest_files), f"{subject}: humor file entered rest whitelist")
        rest_nii = [path for path in rest_files if path.name.endswith("_bold.nii.gz")]
        rest_json = [path for path in rest_files if path.name.endswith("_bold.json")]
        expected_runs = 2 if subject == TWO_RUN_SUBJECT else 1
        require(len(rest_nii) == expected_runs, f"{subject}: expected {expected_runs} rest NIfTI, found {len(rest_nii)}")
        require(len(rest_json) == expected_runs, f"{subject}: expected {expected_runs} rest JSON, found {len(rest_json)}")
        require(
            {path.name[:-7] for path in rest_nii} == {path.stem for path in rest_json},
            f"{subject}: rest NIfTI/JSON basename mismatch",
        )
        total_rest_nii += len(rest_nii)
        total_rest_json += len(rest_json)
        for source in rest_files:
            relative = Path(subject) / "func" / source.name
            plans.append(CopyPlan(source, relative, "rest"))
            rest_manifest_seed.append(
                {
                    "subject": subject,
                    "kind": "bold_nii_gz" if source.name.endswith(".nii.gz") else "bold_json" if source.suffix == ".json" else "other_rest",
                    "source_path": str(source),
                    "output_path": str(OUTPUT / relative),
                    "size_bytes": source.stat().st_size,
                    "source_sha256": "",
                    "output_sha256": "",
                    "status": "",
                }
            )

        # If fmap exists, retain only JSONs whose IntendedFor still targets copied rest data,
        # plus their same-stem companion files. No fmap is synthesized.
        fmap_dir = rest_dir / "fmap"
        if fmap_dir.is_dir():
            output_rest_relatives = {
                (Path(subject) / "func" / path.name).as_posix() for path in rest_nii
            }
            selected_fmap_sources: set[Path] = set()
            fmap_json_payloads: dict[Path, tuple[dict[str, Any], str]] = {}
            for fmap_json in sorted(fmap_dir.glob("*.json"), key=lambda path: path.name):
                with fmap_json.open("r", encoding="utf-8-sig") as stream:
                    payload = json.load(stream)
                original = payload.get("IntendedFor", [])
                if isinstance(original, str):
                    original_values = [original]
                elif isinstance(original, list):
                    original_values = [value for value in original if isinstance(value, str)]
                else:
                    original_values = []
                kept = [
                    value
                    for value in original_values
                    if "task-humor" not in value.lower()
                    and normalized_intended_target(subject, value) in output_rest_relatives
                ]
                if kept:
                    payload["IntendedFor"] = kept if isinstance(original, list) else kept[0]
                    removed = [value for value in original_values if value not in kept]
                    note = f"IntendedFor kept={kept!r}; removed={removed!r}"
                    fmap_json_payloads[fmap_json] = (payload, note)
                    selected_fmap_sources.add(fmap_json)
                    base = fmap_json.name[:-5]
                    selected_fmap_sources.update(
                        path for path in fmap_dir.glob(base + ".*") if path.is_file()
                    )
            for source in sorted(selected_fmap_sources, key=lambda path: path.name):
                relative = Path(subject) / "fmap" / source.name
                if source in fmap_json_payloads:
                    payload, note = fmap_json_payloads[source]
                    plans.append(CopyPlan(source, relative, "fmap", payload, note))
                else:
                    plans.append(CopyPlan(source, relative, "fmap"))

        print(f"[{index:02d}/70] {subject} PRECHECK PASS", flush=True)

    require(total_rest_nii == 71, f"Expected 71 rest BOLD NIfTI files, found {total_rest_nii}")
    require(total_rest_json == 71, f"Expected 71 rest BOLD JSON files, found {total_rest_json}")
    require(len({plan.relative_target for plan in plans}) == len(plans), "Duplicate output path in copy plan")
    return plans, t1_manifest, rest_manifest_seed


def copy_and_hash(
    plans: list[CopyPlan],
    staging: Path,
) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for plan in plans:
        actual_target = staging / plan.relative_target
        logical_target = OUTPUT / plan.relative_target
        actual_target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha256_file(plan.source)
        if plan.modified_json is None:
            shutil.copy2(plan.source, actual_target)
        else:
            write_json(actual_target, plan.modified_json)
        output_hash = sha256_file(actual_target)
        modified = plan.modified_json is not None
        status = "MODIFIED_ALLOWED" if modified else ("HASH_MATCH" if source_hash == output_hash else "HASH_MISMATCH")
        require(modified or source_hash == output_hash, f"Hash mismatch after copy: {plan.source}")
        audit.append(
            {
                "category": plan.category,
                "source_path": str(plan.source),
                "target_path": str(logical_target),
                "source_size_bytes": plan.source.stat().st_size,
                "target_size_bytes": actual_target.stat().st_size,
                "source_sha256": source_hash,
                "target_sha256": output_hash,
                "status": status,
                "note": plan.note,
            }
        )
    return audit


def participants_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return {(row.get("participant_id") or "").strip() for row in rows}


def contains_absolute_windows_path(value: Any) -> bool:
    if isinstance(value, str):
        return bool(DRIVE_PATH_RE.match(value))
    if isinstance(value, list):
        return any(contains_absolute_windows_path(item) for item in value)
    if isinstance(value, dict):
        return any(contains_absolute_windows_path(item) for item in value.values())
    return False


def inspect_nifti(path: Path) -> tuple[list[int], list[float], list[list[float]]]:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise BuildError("nibabel is required for the configured T1 verification participant NIfTI verification") from exc
    image = nib.load(str(path))
    shape = [int(value) for value in image.shape]
    voxel_size = [float(value) for value in image.header.get_zooms()[: len(shape)]]
    affine = [[float(value) for value in row] for row in image.affine.tolist()]
    return shape, voxel_size, affine


def validate_staging(
    staging: Path,
    plans: list[CopyPlan],
    audit: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    checks: dict[str, Any] = {}
    subjects = subject_dirs(staging)
    subject_names = [path.name for path in subjects]
    checks["subjects"] = len(subjects)

    rest_nii = sorted(staging.glob("sub-*/func/*task-rest*_bold.nii.gz"))
    rest_json = sorted(staging.glob("sub-*/func/*task-rest*_bold.json"))
    humor_files = sorted(path for path in staging.rglob("*") if path.is_file() and "task-humor" in path.name.lower())
    t1_nii = sorted(staging.glob("sub-*/anat/*_T1w.nii.gz"))
    t1_json = sorted(staging.glob("sub-*/anat/*_T1w.json"))
    zero_byte = sorted(path for path in staging.rglob("*") if path.is_file() and path.stat().st_size == 0)

    checks.update(
        {
            "task_rest_bold_nii_gz": len(rest_nii),
            "task_rest_bold_json": len(rest_json),
            "task_humor_files": len(humor_files),
            "t1w_nii_gz": len(t1_nii),
            "t1w_json": len(t1_json),
            "zero_byte_files": len(zero_byte),
        }
    )

    subject_rows: list[dict[str, Any]] = []
    exactly_one_t1_nii = 0
    exactly_one_t1_json = 0
    non0045_rest_ok = 0
    two_run_participant_count = 0
    for subject in EXPECTED_SUBJECTS:
        anat = staging / subject / "anat"
        func = staging / subject / "func"
        s_t1_nii = sorted(anat.glob("*_T1w.nii.gz"))
        s_t1_json = sorted(anat.glob("*_T1w.json"))
        s_rest_nii = sorted(func.glob("*task-rest*_bold.nii.gz"))
        s_rest_json = sorted(func.glob("*task-rest*_bold.json"))
        paired_t1 = (
            len(s_t1_nii) == 1
            and len(s_t1_json) == 1
            and t1_stem(s_t1_nii[0]) == s_t1_json[0].stem
        )
        exactly_one_t1_nii += int(len(s_t1_nii) == 1)
        exactly_one_t1_json += int(len(s_t1_json) == 1)
        expected_rest = 2 if subject == TWO_RUN_SUBJECT else 1
        if subject == TWO_RUN_SUBJECT:
            two_run_participant_count = len(s_rest_nii)
        else:
            non0045_rest_ok += int(len(s_rest_nii) == 1 and len(s_rest_json) == 1)
        passed = (
            len(s_t1_nii) == 1
            and len(s_t1_json) == 1
            and paired_t1
            and len(s_rest_nii) == expected_rest
            and len(s_rest_json) == expected_rest
        )
        subject_rows.append(
            {
                "subject": subject,
                "rest_bold_nii_gz": len(s_rest_nii),
                "rest_bold_json": len(s_rest_json),
                "t1w_nii_gz": len(s_t1_nii),
                "t1w_json": len(s_t1_json),
                "t1_pair_match": paired_t1,
                "status": "PASS" if passed else "FAIL",
            }
        )

    checks["subjects_with_exactly_one_t1_nii"] = exactly_one_t1_nii
    checks["subjects_with_exactly_one_t1_json"] = exactly_one_t1_json
    checks["two_run_participant_rest_runs"] = two_run_participant_count
    checks["other_subjects_with_one_rest_run"] = non0045_rest_ok

    audit_by_target = {row["target_path"]: row for row in audit}
    rest_audit = [row for row in audit if row["category"] == "rest"]
    t1_audit = [row for row in audit if row["category"] == "t1"]
    rest_all_hash_matches = sum(row["status"] == "HASH_MATCH" for row in rest_audit)
    t1_all_hash_matches = sum(row["status"] == "HASH_MATCH" for row in t1_audit)
    rest_nii_hash_matches = sum(
        row["status"] == "HASH_MATCH" and row["target_path"].endswith(".nii.gz")
        for row in rest_audit
    )
    t1_nii_hash_matches = sum(
        row["status"] == "HASH_MATCH" and row["target_path"].endswith(".nii.gz")
        for row in t1_audit
    )
    checks.update(
        {
            "rest_all_file_hash_matches": rest_all_hash_matches,
            "rest_all_file_count": len(rest_audit),
            "rest_nii_hash_matches": rest_nii_hash_matches,
            "t1_all_file_hash_matches": t1_all_hash_matches,
            "t1_all_file_count": len(t1_audit),
            "t1_nii_hash_matches": t1_nii_hash_matches,
            "copy_failures": sum(row["status"] == "HASH_MISMATCH" for row in audit),
        }
    )

    planned_targets = {str(OUTPUT / plan.relative_target) for plan in plans}
    actual_targets = {
        str(OUTPUT / path.relative_to(staging))
        for path in staging.rglob("*")
        if path.is_file()
    }
    unknown = sorted(actual_targets - planned_targets)
    missing = sorted(planned_targets - actual_targets)
    checks["unknown_source_files"] = len(unknown)
    checks["missing_planned_files"] = len(missing)

    absolute_json_paths: list[str] = []
    for json_path in sorted(staging.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"Unparseable JSON: {json_path.relative_to(staging)}: {exc}")
            continue
        if contains_absolute_windows_path(payload):
            absolute_json_paths.append(str(json_path.relative_to(staging)))
    checks["json_files_with_absolute_windows_paths"] = len(absolute_json_paths)

    expected_conditions = [
        (subject_names == EXPECTED_SUBJECTS, f"subjects list mismatch: {subject_names}"),
        (len(rest_nii) == 71, f"rest NIfTI count is {len(rest_nii)}, expected 71"),
        (len(rest_json) == 71, f"rest JSON count is {len(rest_json)}, expected 71"),
        (not humor_files, f"found task-humor filenames: {humor_files}"),
        (len(t1_nii) == 70, f"T1 NIfTI count is {len(t1_nii)}, expected 70"),
        (len(t1_json) == 70, f"T1 JSON count is {len(t1_json)}, expected 70"),
        (exactly_one_t1_nii == 70, f"only {exactly_one_t1_nii}/70 subjects have one T1 NIfTI"),
        (exactly_one_t1_json == 70, f"only {exactly_one_t1_json}/70 subjects have one T1 JSON"),
        (all(row["t1_pair_match"] for row in subject_rows), "at least one T1 NIfTI lacks matching JSON"),
        (two_run_participant_count == 2, f"the two-run participant has {two_run_participant_count} rest runs, expected 2"),
        (non0045_rest_ok == 69, f"only {non0045_rest_ok}/69 other subjects have one rest run"),
        ((staging / "dataset_description.json").is_file(), "dataset_description.json missing"),
        ((staging / "participants.tsv").is_file(), "participants.tsv missing"),
        (participants_ids(staging / "participants.tsv") == set(EXPECTED_SUBJECTS), "participants.tsv does not cover exactly 70 subjects"),
        (rest_all_hash_matches == len(rest_audit), "not all copied rest files match source hashes"),
        (t1_all_hash_matches == len(t1_audit), "not all copied T1 files match source hashes"),
        (not zero_byte, f"zero-byte files found: {zero_byte}"),
        (not unknown, f"unknown-source files found: {unknown}"),
        (not missing, f"planned files missing: {missing}"),
        (not absolute_json_paths, f"absolute Windows paths found in BIDS JSON: {absolute_json_paths}"),
    ]
    failures.extend(message for passed, message in expected_conditions if not passed)
    return checks, subject_rows, failures


def run_bids_validator(dataset: Path) -> dict[str, Any]:
    executable = shutil.which("bids-validator")
    if executable is None:
        return {
            "installed": False,
            "command": None,
            "returncode": None,
            "passed": None,
            "output": "bids-validator not installed; installation was not attempted.",
        }
    completed = subprocess.run(
        [executable, str(dataset)],
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    combined = (completed.stdout or "") + (completed.stderr or "")
    return {
        "installed": True,
        "command": f"{executable} {dataset}",
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "output": combined,
    }


def enrich_manifests(
    t1_manifest: list[dict[str, Any]],
    rest_manifest: list[dict[str, Any]],
    audit: list[dict[str, Any]],
) -> None:
    by_target = {row["target_path"]: row for row in audit}
    for row in t1_manifest:
        nii_audit = by_target[row["output_t1_nii"]]
        json_audit = by_target[row["output_t1_json"]]
        row["source_t1_nii_sha256"] = nii_audit["source_sha256"]
        row["source_t1_json_sha256"] = json_audit["source_sha256"]
        row["output_t1_nii_sha256"] = nii_audit["target_sha256"]
        row["output_t1_json_sha256"] = json_audit["target_sha256"]
        row["hash_identical"] = (
            nii_audit["status"] == "HASH_MATCH" and json_audit["status"] == "HASH_MATCH"
        )
    for row in rest_manifest:
        file_audit = by_target[row["output_path"]]
        row["source_sha256"] = file_audit["source_sha256"]
        row["output_sha256"] = file_audit["target_sha256"]
        row["status"] = file_audit["status"]


def write_selected_t1_verification(staging: Path, t1_manifest: list[dict[str, Any]]) -> None:
    row = next(item for item in t1_manifest if item["subject"] == T1_VERIFICATION_SUBJECT)
    output_name = Path(row["output_t1_nii"]).name
    actual_output = staging / T1_VERIFICATION_SUBJECT / "anat" / output_name
    shape, voxel_size, affine = inspect_nifti(actual_output)
    count = len(list((staging / T1_VERIFICATION_SUBJECT / "anat").glob("*_T1w.nii.gz")))
    lines = [
        "the configured T1 verification participant T1 verification",
        f"selected source T1: {row['source_t1_nii']}",
        f"source hash: {row['source_t1_nii_sha256']}",
        f"output T1: {row['output_t1_nii']}",
        f"output hash: {row['output_t1_nii_sha256']}",
        f"hash identical: {row['source_t1_nii_sha256'] == row['output_t1_nii_sha256']}",
        f"number of T1w images in output anat: {count}",
        f"shape: {shape}",
        f"voxel size: {voxel_size}",
        "affine:",
        json.dumps(affine, ensure_ascii=False, indent=2),
        "",
    ]
    (LOG_DIR / "selected_t1_verification.txt").write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    checks: dict[str, Any],
    failures: list[str],
    validator: dict[str, Any],
) -> None:
    validation_pass = not failures
    summary = {
        "sources": {
            "rest_source": str(REST_SOURCE),
            "single_t1_qc_directory": str(T1_QC_SOURCE),
            "resolved_single_t1_data_source": str(T1_DATA_SOURCE),
            "output": str(OUTPUT),
        },
        "checks": checks,
        "bids_validator": {
            key: value for key, value in validator.items() if key != "output"
        },
        "failures": failures,
        "VALIDATION_PASS": validation_pass,
    }
    write_json(LOG_DIR / "validation_summary.json", summary)
    (LOG_DIR / "bids_validator_output.txt").write_text(validator["output"], encoding="utf-8")

    validator_status = (
        "not installed (not treated as failure)"
        if not validator["installed"]
        else f"return code {validator['returncode']} ({'PASS' if validator['passed'] else 'FAIL'})"
    )
    failure_lines = ["- 无失败项"] if not failures else [f"- {item}" for item in failures]
    report_lines = [
        "# rest + manually confirmed singleT1 BIDS validation report",
        "",
        "## Provenance",
        "",
        f"- Rest source: `{REST_SOURCE}`",
        f"- Requested QC/selection directory: `{T1_QC_SOURCE}`",
        f"- Resolved 70-subject singleT1 data source: `{T1_DATA_SOURCE}`",
        f"- Output: `{OUTPUT}`",
        "- The QC manifest was read for all 70 subjects and matched to the actual unique T1w pair for every subject.",
        "",
        "## Core counts",
        "",
        f"- subjects = {checks['subjects']}",
        f"- task-rest bold.nii.gz = {checks['task_rest_bold_nii_gz']}",
        f"- task-rest bold.json = {checks['task_rest_bold_json']}",
        f"- task-humor files = {checks['task_humor_files']}",
        f"- T1w nii.gz = {checks['t1w_nii_gz']}",
        f"- T1w json = {checks['t1w_json']}",
        f"- subjects with exactly one T1 NIfTI = {checks['subjects_with_exactly_one_t1_nii']}/70",
        f"- subjects with exactly one T1 JSON = {checks['subjects_with_exactly_one_t1_json']}/70",
        f"- the two-run participant rest runs = {checks['two_run_participant_rest_runs']}",
        f"- other subjects with one rest run = {checks['other_subjects_with_one_rest_run']}/69",
        "",
        "## Hash and integrity audit",
        "",
        f"- rest NIfTI hash match = {checks['rest_nii_hash_matches']}/71",
        f"- all rest files hash match = {checks['rest_all_file_hash_matches']}/{checks['rest_all_file_count']}",
        f"- T1 NIfTI hash match = {checks['t1_nii_hash_matches']}/70",
        f"- all T1 files hash match = {checks['t1_all_file_hash_matches']}/{checks['t1_all_file_count']}",
        f"- zero-byte files = {checks['zero_byte_files']}",
        f"- copy failures = {checks['copy_failures']}",
        f"- unknown-source files = {checks['unknown_source_files']}",
        f"- absolute Windows paths in BIDS JSON = {checks['json_files_with_absolute_windows_paths']}",
        "",
        "## BIDS validator",
        "",
        f"- {validator_status}",
        f"- Full output: `{LOG_DIR / 'bids_validator_output.txt'}`",
        "",
        "## Failed checks",
        "",
        *failure_lines,
        "",
        f"**VALIDATION_PASS = {validation_pass}**",
        "",
    ]
    (LOG_DIR / "validation_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def print_summary(checks: dict[str, Any], failures: list[str]) -> None:
    print(f"subjects = {checks['subjects']}")
    print(f"task-rest bold.nii.gz = {checks['task_rest_bold_nii_gz']}")
    print(f"task-rest bold.json = {checks['task_rest_bold_json']}")
    print(f"task-humor files = {checks['task_humor_files']}")
    print(f"T1w nii.gz = {checks['t1w_nii_gz']}")
    print(f"T1w json = {checks['t1w_json']}")
    print(f"subjects with exactly one T1 NIfTI = {checks['subjects_with_exactly_one_t1_nii']}/70")
    print(f"subjects with exactly one T1 JSON = {checks['subjects_with_exactly_one_t1_json']}/70")
    print(f"the two-run participant rest runs = {checks['two_run_participant_rest_runs']}")
    print(f"rest hash match = {checks['rest_nii_hash_matches']}/71")
    print(f"T1 hash match = {checks['t1_nii_hash_matches']}/70")
    if failures:
        print("Failed checks:")
        for item in failures:
            print(f"- {item}")
    else:
        print("Failed checks: 无失败项")
    print(f"VALIDATION_PASS = {not failures}")


def main() -> int:
    staging = OUTPUT.parent / f".{OUTPUT.name}.building-{os.getpid()}"
    try:
        if OUTPUT.exists():
            raise BuildError(f"Output already exists; refusing to overwrite or merge: {OUTPUT}")
        if staging.exists():
            raise BuildError(f"Staging directory already exists: {staging}")

        print("Phase 1/5: preflight source inspection", flush=True)
        plans, t1_manifest, rest_manifest = inspect_sources()

        print(f"Phase 2/5: copying {len(plans)} whitelisted files and computing SHA256", flush=True)
        staging.mkdir(parents=False, exist_ok=False)
        audit = copy_and_hash(plans, staging)
        enrich_manifests(t1_manifest, rest_manifest, audit)

        print("Phase 3/5: core dataset validation", flush=True)
        checks, subject_rows, failures = validate_staging(staging, plans, audit)

        print("Phase 4/5: local bids-validator check", flush=True)
        validator = run_bids_validator(staging)
        if validator["installed"] and not validator["passed"]:
            failures.append(f"bids-validator returned non-zero exit code {validator['returncode']}")

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        write_csv(
            LOG_DIR / "selected_t1_manifest.csv",
            [
                "subject", "selection_manifest", "source_t1_nii", "source_t1_json",
                "source_t1_nii_sha256", "source_t1_json_sha256", "output_t1_nii",
                "output_t1_json", "output_t1_nii_sha256", "output_t1_json_sha256",
                "hash_identical",
            ],
            t1_manifest,
        )
        write_csv(
            LOG_DIR / "rest_file_manifest.csv",
            [
                "subject", "kind", "source_path", "output_path", "size_bytes",
                "source_sha256", "output_sha256", "status",
            ],
            rest_manifest,
        )
        write_csv(
            LOG_DIR / "copy_and_hash_audit.csv",
            [
                "category", "source_path", "target_path", "source_size_bytes",
                "target_size_bytes", "source_sha256", "target_sha256", "status", "note",
            ],
            audit,
        )
        write_csv(
            LOG_DIR / "subject_count_audit.csv",
            [
                "subject", "rest_bold_nii_gz", "rest_bold_json", "t1w_nii_gz",
                "t1w_json", "t1_pair_match", "status",
            ],
            subject_rows,
        )
        write_selected_t1_verification(staging, t1_manifest)
        write_reports(checks, failures, validator)

        if failures:
            print_summary(checks, failures)
            raise BuildError("Validation failed; final output was not published")

        print("Phase 5/5: publishing validated output", flush=True)
        staging.rename(OUTPUT)
        print_summary(checks, failures)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if not isinstance(exc, BuildError):
            traceback.print_exc()
        if staging.exists():
            shutil.rmtree(staging)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
