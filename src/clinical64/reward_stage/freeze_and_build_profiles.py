#!/usr/bin/env python3
"""CURRENT64 canonical rerun, stage 1 (Steps 1--5 only).

This script freezes the exact 64-subject inputs, validates every
2608 x 65 Fisher-z fingerprint, rebuilds Subject x Bin x ROI profiles from
the fingerprints using canonical Definition A, aggregates the frozen reward
stages, and compares the rebuilt stage table with the historical stage table.

It intentionally does not run inferential models, behavior analyses,
normative-deviation analyses, post-hoc tests, or figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SUBJECTS = 64
EXPECTED_GROUP_COUNTS = {"HC": 32, "SZ": 32}
EXPECTED_FP_SHAPE = (2608, 65)
EXPECTED_BIN_COUNTS = [260, 260, 261, 261, 261, 261, 261, 261, 261, 261]
EXPECTED_STAGE_COUNTS = {"Anticipation": 43, "Outcome": 19, "Not_assigned": 3}
BEHAVIOR_SOURCE_COLUMNS = [
    "TEPS_anticipatory_calc",
    "TEPS_consummatory_calc",
    "ACIPS_anticipatory_calc",
    "ACIPS_consummatory_calc",
]
BEHAVIOR_CANONICAL_COLUMNS = [
    "TEPS_anticipatory",
    "TEPS_consummatory",
    "ACIPS_anticipatory",
    "ACIPS_consummatory",
]
STAGE_ORDER = ["Anticipation", "Outcome"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New CURRENT64 canonical-rerun output directory.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Private project root containing the audited archive and metadata.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        help="Override the audited 64-person archive root.",
    )
    parser.add_argument(
        "--metadata-source",
        type=Path,
        help="Override the 65-ROI stage/subfamily metadata TSV.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all source inputs without writing frozen inputs or outputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        preview = "\n".join(existing[:12])
        extra = "" if len(existing) <= 12 else f"\n... and {len(existing) - 12} more"
        raise FileExistsError(
            "Default no-overwrite policy: the following planned outputs already exist:\n"
            f"{preview}{extra}"
        )


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_true(not path.exists(), f"Refusing to overwrite existing file: {path}")
    df.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.17g")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_true(not path.exists(), f"Refusing to overwrite existing file: {path}")
    path.write_text(text, encoding="utf-8", newline="\n")


def copy_frozen(
    role: str,
    source: Path,
    destination: Path,
    source_shape_or_rows: str,
    manifest_rows: list[dict[str, object]],
) -> None:
    assert_true(source.is_file(), f"Missing source file: {source}")
    assert_true(not destination.exists(), f"Refusing to overwrite frozen file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    shutil.copy2(source, destination)
    frozen_hash = sha256_file(destination)
    assert_true(source_hash == frozen_hash, f"Hash mismatch after freezing: {source}")
    manifest_rows.append(
        {
            "input_role": role,
            "source_path": str(source.resolve()),
            "frozen_path": str(destination.resolve()),
            "source_shape_or_rows": source_shape_or_rows,
            "source_sha256": source_hash,
            "frozen_sha256": frozen_hash,
            "sha256_match": True,
            "status": "FROZEN_EXACT_COPY",
        }
    )


def register_derived_frozen(
    role: str,
    destination: Path,
    source_paths: list[Path],
    source_shape_or_rows: str,
    manifest_rows: list[dict[str, object]],
) -> None:
    manifest_rows.append(
        {
            "input_role": role,
            "source_path": ";".join(str(path.resolve()) for path in source_paths),
            "frozen_path": str(destination.resolve()),
            "source_shape_or_rows": source_shape_or_rows,
            "source_sha256": ";".join(sha256_file(path) for path in source_paths),
            "frozen_sha256": sha256_file(destination),
            "sha256_match": "DERIVED_CANONICAL_TABLE",
            "status": "FROZEN_DERIVED_WITH_EXPLICIT_TRANSFORM",
        }
    )


def r_version() -> str:
    candidates = [
        shutil.which(os.environ.get("RSCRIPT", "Rscript")),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                result = subprocess.run(
                    [str(candidate), "--version"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=15,
                )
                version = (result.stdout or result.stderr).strip()
                if version:
                    return version
            except (OSError, subprocess.SubprocessError):
                pass
    return "Rscript not detected; R was not used in Steps 1-5"


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    project_root = args.project_root.resolve()
    archive_root = (
        args.archive_root.resolve()
        if args.archive_root
        else project_root / "99_archive_64" / "rebuild_oxford_65roi_v2"
    )

    manuscript_dir = (
        archive_root
        / "20_current64_reward_stage_fc"
        / "01_fixed375_stage_profiles_manuscript"
    )
    fp_root = archive_root / "17_current64_375matched_fingerprints"
    individual_root = archive_root / "21_current64_individual_stats"
    metadata_source = (
        args.metadata_source.resolve()
        if args.metadata_source
        else project_root
        / "03_figures"
        / "main"
        / "26_figure3_final"
        / "tables"
        / "source_tables"
        / "roi_metadata_stage_subfamily_v2.tsv"
    )

    sources = {
        "analysis_master": manuscript_dir / "analysis_master_64.tsv",
        "group_labels": archive_root
        / "08_current32HC_strict_LOO"
        / "01_inputs"
        / "current64_group_labels.tsv",
        "clinical": individual_root
        / "01_inputs"
        / "01_clinical"
        / "final64_clinical_for_server_v3.tsv",
        "motion": individual_root
        / "03_derived_metrics"
        / "motion"
        / "current64_motion_posttrim.tsv",
        "g1": manuscript_dir / "375_reference_G1_manuscript_2608.npy",
        "bins": manuscript_dir / "375_reference_bins_manuscript_2608.npy",
        "bin_alias": manuscript_dir / "fixed_375_G1_voxel_bins_2608.tsv",
        "bin_summary": manuscript_dir / "fixed_375_G1_bin_summary.tsv",
        "historical_stage": manuscript_dir / "subject_bin_stage_FC_long.tsv",
        "mapped_ijk": fp_root / "01_inputs" / "mapped_2608_voxel_ijk.npy",
        "mapped_flat": fp_root / "01_inputs" / "mapped_2608_voxel_indices.npy",
        "whole_ijk": fp_root / "01_inputs" / "whole_striatum_2847_voxel_ijk.npy",
        "whole_flat": fp_root / "01_inputs" / "whole_striatum_2847_voxel_indices.npy",
        "row_mapping": fp_root / "01_inputs" / "row_mapping_2847_to_2608.npy",
        "roi_names": fp_root / "01_inputs" / "roi_names_65_exact_order.txt",
        "roi_metadata": metadata_source,
        "fingerprint_manifest": fp_root / "04_qc" / "375matched_fingerprint_manifest.tsv",
    }
    for name, path in sources.items():
        assert_true(path.is_file(), f"Missing required source ({name}): {path}")

    if not args.check_only:
        output_root.mkdir(parents=True, exist_ok=True)
        log_dir = output_root / "17_LOGS"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        log_path = log_dir / f"stage1_steps1_to_5_{timestamp}.log"
        assert_true(not log_path.exists(), f"Log already exists: {log_path}")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        log_path = None

    logging.info("CURRENT64 canonical rerun stage 1 started")
    logging.info("Output root: %s", output_root)
    logging.info("Archive root is read-only input: %s", archive_root)

    master = pd.read_csv(sources["analysis_master"], sep="\t", dtype={"subject": str})
    labels = pd.read_csv(sources["group_labels"], sep="\t", dtype={"subject_id": str})
    clinical = pd.read_csv(sources["clinical"], sep="\t", dtype={"subject": str})
    motion = pd.read_csv(sources["motion"], sep="\t", dtype={"subject": str})

    assert_true(len(master) == EXPECTED_SUBJECTS, "analysis_master_64.tsv is not 64 rows")
    assert_true(master["subject"].is_unique, "Duplicate subjects in analysis master")
    assert_true(master["group"].value_counts().to_dict() == EXPECTED_GROUP_COUNTS, "Expected 32 HC and 32 SZ")
    subject_order = master["subject"].tolist()
    subject_set = set(subject_order)

    assert_true(set(labels["subject_id"]) == subject_set, "Group-label subjects do not match master")
    assert_true(set(clinical["subject"]) == subject_set, "Clinical subjects do not match master")
    assert_true(set(motion["subject"]) == subject_set, "Motion subjects do not match master")

    clinical_check = master[["subject", "group"]].merge(
        clinical[["subject", "group"]], on="subject", suffixes=("_master", "_clinical")
    )
    assert_true(
        bool((clinical_check["group_master"] == clinical_check["group_clinical"]).all()),
        "Clinical group labels disagree with analysis master",
    )
    group_crosswalk = master[["subject", "group"]].merge(
        labels[["subject_id", "group"]], left_on="subject", right_on="subject_id"
    )
    crosswalk_counts = group_crosswalk.groupby(["group_x", "group_y"]).size().to_dict()
    assert_true(
        crosswalk_counts == {("HC", 4): 32, ("SZ", 3): 32},
        f"Unexpected legacy group-code crosswalk: {crosswalk_counts}",
    )

    motion_check = master[["subject", "mean_FD_posttrim"]].merge(
        motion[["subject", "mean_FD_posttrim"]], on="subject", suffixes=("_master", "_motion")
    )
    motion_max_delta = float(
        np.max(
            np.abs(
                motion_check["mean_FD_posttrim_master"].to_numpy(dtype=float)
                - motion_check["mean_FD_posttrim_motion"].to_numpy(dtype=float)
            )
        )
    )
    assert_true(motion_max_delta <= 1e-15, f"Motion mismatch; max delta={motion_max_delta}")

    required_master = ["subject", "group", "age", "sex", "mean_FD_posttrim"]
    required_behavior = ["subject", "group", *BEHAVIOR_SOURCE_COLUMNS]
    assert_true(not master[required_master].isna().any().any(), "Missing required covariate values")
    assert_true(set(required_behavior).issubset(clinical.columns), "Missing required final behavior columns")
    assert_true(not clinical[["subject", "group"]].isna().any().any(), "Missing clinical subject/group labels")

    g1 = np.load(sources["g1"])
    bins = np.load(sources["bins"])
    bin_alias = pd.read_csv(sources["bin_alias"], sep="\t")
    mapped_ijk = np.load(sources["mapped_ijk"])
    mapped_flat = np.load(sources["mapped_flat"])
    whole_ijk = np.load(sources["whole_ijk"])
    whole_flat = np.load(sources["whole_flat"])
    row_mapping = np.load(sources["row_mapping"])

    assert_true(g1.shape == (2608,), f"Unexpected G1 shape: {g1.shape}")
    assert_true(bins.shape == (2608,), f"Unexpected bin shape: {bins.shape}")
    assert_true(np.isfinite(g1).all(), "Canonical G1 contains non-finite values")
    assert_true(set(np.unique(bins).tolist()) == set(range(1, 11)), "Bins are not exactly 1..10")
    bin_counts = [int(np.sum(bins == value)) for value in range(1, 11)]
    assert_true(bin_counts == EXPECTED_BIN_COUNTS, f"Definition A bin counts differ: {bin_counts}")
    assert_true(len(bin_alias) == 2608, "Canonical bin alias is not 2608 rows")
    assert_true(
        np.array_equal(bin_alias["bin"].to_numpy(dtype=int), bins.astype(int)),
        "Canonical bin NPY and exact TSV alias disagree",
    )
    assert_true(
        np.allclose(bin_alias["reference_G1"].to_numpy(dtype=float), g1, rtol=0, atol=1e-15),
        "Canonical G1 NPY and bin TSV alias disagree",
    )

    assert_true(mapped_ijk.shape == (2608, 3), f"Unexpected mapped IJK shape: {mapped_ijk.shape}")
    assert_true(mapped_flat.shape == (2608,), f"Unexpected mapped-flat shape: {mapped_flat.shape}")
    assert_true(row_mapping.shape == (2608,), f"Unexpected row-mapping shape: {row_mapping.shape}")
    recomputed_flat = np.ravel_multi_index(mapped_ijk.astype(int).T, (97, 115, 97), order="C")
    assert_true(np.array_equal(recomputed_flat, mapped_flat.astype(int)), "IJK-to-flat mapping mismatch")
    assert_true(np.array_equal(whole_flat[row_mapping], mapped_flat), "2847-to-2608 flat mapping mismatch")
    assert_true(np.array_equal(whole_ijk[row_mapping], mapped_ijk), "2847-to-2608 IJK mapping mismatch")

    roi_names = [line.strip() for line in sources["roi_names"].read_text(encoding="utf-8").splitlines() if line.strip()]
    roi_metadata = pd.read_csv(sources["roi_metadata"], sep="\t")
    assert_true(len(roi_names) == 65 and len(set(roi_names)) == 65, "ROI name list is not 65 unique rows")
    assert_true(len(roi_metadata) == 65, "ROI metadata is not 65 rows")
    assert_true(np.array_equal(roi_metadata["roi_index0"].to_numpy(), np.arange(65)), "ROI index order mismatch")
    assert_true(np.array_equal(roi_metadata["roi_id"].to_numpy(), np.arange(1, 66)), "ROI id order mismatch")
    stage_counts = roi_metadata["stage"].value_counts().to_dict()
    assert_true(stage_counts == EXPECTED_STAGE_COUNTS, f"Unexpected stage counts: {stage_counts}")
    assert_true(
        bool((roi_metadata.loc[roi_metadata["stage"].isin(STAGE_ORDER), "include_two_family"] == 1).all()),
        "Assigned reward-stage ROIs are not all included",
    )
    assert_true(
        bool((roi_metadata.loc[roi_metadata["stage"] == "Not_assigned", "include_two_family"] == 0).all()),
        "Not-assigned ROIs are not all excluded",
    )

    fingerprint_files = {
        subject: fp_root
        / "03_fingerprints"
        / subject
        / f"{subject}_fingerprint_z_2608x65.npy"
        for subject in subject_order
    }
    discovered = sorted((fp_root / "03_fingerprints").glob("*/sub-*_fingerprint_z_2608x65.npy"))
    assert_true(len(discovered) == EXPECTED_SUBJECTS, f"Found {len(discovered)} fingerprint files, expected 64")
    assert_true(set(path.parent.name for path in discovered) == subject_set, "Fingerprint subjects do not match master")

    fingerprint_qc_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    profile_qc_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    fp_hashes: dict[str, str] = {}
    group_by_subject = dict(zip(master["subject"], master["group"]))
    covariate_by_subject = master.set_index("subject")[["age", "sex", "mean_FD_posttrim"]]
    g1_bin_means = {value: float(np.mean(g1[bins == value])) for value in range(1, 11)}

    logging.info("Auditing fingerprints and rebuilding profiles directly from arrays")
    for subject in subject_order:
        path = fingerprint_files[subject]
        assert_true(path.is_file(), f"Missing fingerprint: {path}")
        fp = np.load(path)
        n_na = int(np.isnan(fp).sum())
        n_inf = int(np.isinf(fp).sum())
        shape_ok = fp.shape == EXPECTED_FP_SHAPE
        pass_qc = shape_ok and n_na == 0 and n_inf == 0
        finite_values = fp[np.isfinite(fp)].astype(np.float64, copy=False)
        fingerprint_hash = sha256_file(path)
        fp_hashes[subject] = fingerprint_hash
        fingerprint_qc_rows.append(
            {
                "Subject": subject,
                "Group": group_by_subject[subject],
                "n_voxel": int(fp.shape[0]) if fp.ndim >= 1 else 0,
                "n_roi": int(fp.shape[1]) if fp.ndim >= 2 else 0,
                "n_na": n_na,
                "n_inf": n_inf,
                "mean_fc": float(np.mean(finite_values)) if finite_values.size else np.nan,
                "sd_fc": float(np.std(finite_values, ddof=0)) if finite_values.size else np.nan,
                "min_fc": float(np.min(finite_values)) if finite_values.size else np.nan,
                "max_fc": float(np.max(finite_values)) if finite_values.size else np.nan,
                "file": str(path.resolve()),
                "sha256": fingerprint_hash,
                "pass_qc": pass_qc,
            }
        )
        assert_true(pass_qc, f"Fingerprint QC failed for {subject}")

        # Source fingerprints are float32.  Promote before averaging so both
        # voxel and ROI reductions use explicit float64 accumulation.
        fp_float64 = fp.astype(np.float64, copy=False)
        bin_roi = np.vstack([fp_float64[bins == bin_value, :].mean(axis=0) for bin_value in range(1, 11)])
        assert_true(bin_roi.shape == (10, 65), f"Unexpected binned profile shape for {subject}")
        assert_true(np.isfinite(bin_roi).all(), f"Non-finite binned profile for {subject}")
        for bin_index, bin_value in enumerate(range(1, 11)):
            wide_row: dict[str, object] = {
                "Subject": subject,
                "Group": group_by_subject[subject],
                "Bin": bin_value,
            }
            for roi_index, roi_name in enumerate(roi_names):
                value = float(bin_roi[bin_index, roi_index])
                metadata_row = roi_metadata.iloc[roi_index]
                profile_rows.append(
                    {
                        "Subject": subject,
                        "Group": group_by_subject[subject],
                        "Bin": bin_value,
                        "ROI": roi_name,
                        "Subfamily": metadata_row["subfamily"],
                        "RewardStage": metadata_row["stage"],
                        "FC": value,
                    }
                )
                wide_row[roi_name] = value
            wide_rows.append(wide_row)
            profile_qc_rows.append(
                {
                    "Subject": subject,
                    "Group": group_by_subject[subject],
                    "Bin": bin_value,
                    "n_voxel": int(np.sum(bins == bin_value)),
                    "n_roi_expected": 65,
                    "n_roi_observed": int(bin_roi.shape[1]),
                    "n_missing_fc": int(np.isnan(bin_roi[bin_index]).sum()),
                    "all_finite": bool(np.isfinite(bin_roi[bin_index]).all()),
                }
            )
            for stage in STAGE_ORDER:
                stage_mask = roi_metadata["stage"].to_numpy() == stage
                stage_rows.append(
                    {
                        "Subject": subject,
                        "Group": group_by_subject[subject],
                        "Bin": bin_value,
                        "G1_bin_center": float(bin_value - 5.5),
                        "Stage": stage,
                        "FC": float(np.mean(bin_roi[bin_index, stage_mask])),
                        "Age": covariate_by_subject.loc[subject, "age"],
                        "Sex": covariate_by_subject.loc[subject, "sex"],
                        "MeanFD": covariate_by_subject.loc[subject, "mean_FD_posttrim"],
                    }
                )

    fingerprint_qc = pd.DataFrame(fingerprint_qc_rows)
    profile_long = pd.DataFrame(profile_rows)
    profile_wide = pd.DataFrame(wide_rows)
    profile_qc = pd.DataFrame(profile_qc_rows)
    rebuilt_stage = pd.DataFrame(stage_rows)

    assert_true(len(fingerprint_qc) == 64 and bool(fingerprint_qc["pass_qc"].all()), "Not all fingerprints passed")
    assert_true(len(profile_long) == 41600, f"Expected 41,600 profile rows; got {len(profile_long)}")
    assert_true(len(profile_wide) == 640, f"Expected 640 wide-profile rows; got {len(profile_wide)}")
    assert_true(len(profile_wide.columns) == 68, "Wide table must have 3 keys + 65 ROI columns")
    assert_true(not profile_long.duplicated(["Subject", "Bin", "ROI"]).any(), "Duplicate Subject-Bin-ROI keys")
    assert_true(len(rebuilt_stage) == 1280, f"Expected 1,280 stage rows; got {len(rebuilt_stage)}")
    assert_true(not rebuilt_stage.duplicated(["Subject", "Bin", "Stage"]).any(), "Duplicate Subject-Bin-Stage keys")
    assert_true(bool(np.isfinite(rebuilt_stage["FC"]).all()), "Rebuilt stage table contains non-finite FC")

    historical = pd.read_csv(sources["historical_stage"], sep="\t")
    assert_true(len(historical) == 1280, "Historical stage table is not 1,280 rows")
    historical_keys = historical[["subject", "bin", "stage"]].rename(
        columns={"subject": "Subject", "bin": "Bin", "stage": "Stage"}
    )
    assert_true(not historical_keys.duplicated().any(), "Duplicate keys in historical stage table")
    comparison = rebuilt_stage.merge(
        historical.rename(
            columns={
                "subject": "Subject",
                "group": "Historical_Group",
                "bin": "Bin",
                "G1_centered": "Historical_G1_bin_center",
                "reference_G1_mean": "Historical_reference_G1_mean",
                "stage": "Stage",
                "FC": "Historical_FC",
            }
        ),
        on=["Subject", "Bin", "Stage"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    assert_true(len(comparison) == 1280, "Historical/rebuilt comparison is not 1,280 rows")
    assert_true(bool((comparison["_merge"] == "both").all()), "Historical/rebuilt keys do not match")
    comparison["Rebuilt_FC"] = comparison["FC"]
    comparison["Difference_rebuilt_minus_historical"] = comparison["Rebuilt_FC"] - comparison["Historical_FC"]
    comparison["AbsDifference"] = comparison["Difference_rebuilt_minus_historical"].abs()
    comparison["Group_match"] = comparison["Group"] == comparison["Historical_Group"]
    comparison["G1_bin_center_difference"] = (
        comparison["G1_bin_center"] - comparison["Historical_G1_bin_center"]
    )
    max_abs_difference = float(comparison["AbsDifference"].max())
    mean_abs_difference = float(comparison["AbsDifference"].mean())
    pearson_r = float(np.corrcoef(comparison["Rebuilt_FC"], comparison["Historical_FC"])[0, 1])
    assert_true(bool(comparison["Group_match"].all()), "Historical/rebuilt group labels differ")
    assert_true(
        float(comparison["G1_bin_center_difference"].abs().max()) == 0.0,
        "Historical/rebuilt centered-bin values differ",
    )
    comparison = comparison[
        [
            "Subject",
            "Group",
            "Bin",
            "G1_bin_center",
            "Stage",
            "Historical_FC",
            "Rebuilt_FC",
            "Difference_rebuilt_minus_historical",
            "AbsDifference",
            "Group_match",
            "Historical_reference_G1_mean",
        ]
    ]

    bin_count_table = pd.DataFrame(
        {
            "Bin": range(1, 11),
            "n_voxel": bin_counts,
            "G1_bin_center": [value - 5.5 for value in range(1, 11)],
            "reference_G1_mean": [g1_bin_means[value] for value in range(1, 11)],
            "reference_G1_min": [float(np.min(g1[bins == value])) for value in range(1, 11)],
            "reference_G1_max": [float(np.max(g1[bins == value])) for value in range(1, 11)],
        }
    )
    roi_order = pd.DataFrame(
        {
            "roi_index": np.arange(65, dtype=int),
            "roi_name": roi_names,
            "subfamily": roi_metadata["subfamily"],
            "reward_stage": roi_metadata["stage"],
            "atlas_source": roi_metadata["brainnetome_label"],
        }
    )
    voxel_order = pd.DataFrame(
        {
            "voxel_row": np.arange(2608, dtype=int),
            "i": mapped_ijk[:, 0].astype(int),
            "j": mapped_ijk[:, 1].astype(int),
            "k": mapped_ijk[:, 2].astype(int),
            "flat_index_c_order_97x115x97": mapped_flat.astype(int),
            "source_row_2847": row_mapping.astype(int),
            "recovery_status": "RECONSTRUCTED_FROM_DERIVED_DATA_EXACT_MATCH",
        }
    )
    subject_table = master[["subject", "group"]].rename(columns={"subject": "Subject", "group": "Group"})
    covariates = master[["subject", "group", "age", "sex", "mean_FD_posttrim"]].rename(
        columns={"subject": "Subject", "group": "Group", "age": "Age", "sex": "Sex", "mean_FD_posttrim": "MeanFD"}
    )
    clinical_ordered = clinical.set_index("subject").loc[subject_order].reset_index()
    behavior = clinical_ordered[["subject", "group", *BEHAVIOR_SOURCE_COLUMNS]].rename(
        columns={
            "subject": "Subject",
            "group": "Group",
            **dict(zip(BEHAVIOR_SOURCE_COLUMNS, BEHAVIOR_CANONICAL_COLUMNS)),
        }
    )
    behavior_missingness = pd.DataFrame(
        {
            "Variable": BEHAVIOR_CANONICAL_COLUMNS,
            "n_total": [len(behavior)] * len(BEHAVIOR_CANONICAL_COLUMNS),
            "n_nonmissing": [int(behavior[column].notna().sum()) for column in BEHAVIOR_CANONICAL_COLUMNS],
            "n_missing": [int(behavior[column].isna().sum()) for column in BEHAVIOR_CANONICAL_COLUMNS],
            "handling": ["FROZEN_AS_IS_NO_IMPUTATION_NO_RESCORING"] * len(BEHAVIOR_CANONICAL_COLUMNS),
        }
    )
    mapping_qc = pd.DataFrame(
        [
            {"check": "roi_count", "observed": 65, "expected": 65, "pass": True},
            {"check": "roi_index0_exact_order", "observed": "0..64", "expected": "0..64", "pass": True},
            {"check": "roi_names_unique", "observed": 65, "expected": 65, "pass": True},
            {"check": "stage_Anticipation_count", "observed": stage_counts["Anticipation"], "expected": 43, "pass": True},
            {"check": "stage_Outcome_count", "observed": stage_counts["Outcome"], "expected": 19, "pass": True},
            {"check": "stage_Not_assigned_count", "observed": stage_counts["Not_assigned"], "expected": 3, "pass": True},
            {"check": "two_stage_analysis_excludes_Not_assigned", "observed": 62, "expected": 62, "pass": True},
        ]
    )
    comparison_summary = pd.DataFrame(
        [
            {"metric": "n_rows", "value": len(comparison)},
            {"metric": "max_abs_rebuilt_minus_historical", "value": max_abs_difference},
            {"metric": "mean_abs_rebuilt_minus_historical", "value": mean_abs_difference},
            {"metric": "pearson_r_rebuilt_vs_historical", "value": pearson_r},
            {"metric": "all_keys_match", "value": True},
            {"metric": "all_groups_match", "value": bool(comparison["Group_match"].all())},
        ]
    )

    summary = {
        "scope": "CURRENT64_CANONICAL_RERUN Steps 1-5 only",
        "completed_at": datetime.now().astimezone().isoformat(),
        "fingerprints": {
            "n_subjects": int(len(fingerprint_qc)),
            "all_pass": bool(fingerprint_qc["pass_qc"].all()),
            "shape": list(EXPECTED_FP_SHAPE),
            "total_na": int(fingerprint_qc["n_na"].sum()),
            "total_inf": int(fingerprint_qc["n_inf"].sum()),
        },
        "profile_table": {
            "long_rows": int(len(profile_long)),
            "wide_rows": int(len(profile_wide)),
            "roi_columns": 65,
            "all_finite": bool(np.isfinite(profile_long["FC"]).all()),
        },
        "stage_table": {
            "rows": int(len(rebuilt_stage)),
            "stages": STAGE_ORDER,
            "all_finite": bool(np.isfinite(rebuilt_stage["FC"]).all()),
        },
        "historical_comparison": {
            "rows": int(len(comparison)),
            "max_abs_difference": max_abs_difference,
            "mean_abs_difference": mean_abs_difference,
            "pearson_r": pearson_r,
        },
        "canonical_definition_a_bin_counts": bin_counts,
        "canonical_G1": {
            "min": float(np.min(g1)),
            "max": float(np.max(g1)),
            "mean": float(np.mean(g1)),
            "sd_ddof0": float(np.std(g1, ddof=0)),
            "direction": "lower=VST/Caudate-rich; higher=Putamen-rich",
        },
        "roi_stage_mapping_problem": False,
        "behavior_frozen_without_rescoring": {
            column: int(behavior[column].isna().sum()) for column in BEHAVIOR_CANONICAL_COLUMNS
        },
        "inferential_statistics_run": False,
        "figures_run": False,
    }

    if args.check_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        logging.info("Check-only validation completed; no files written")
        return 0

    planned_outputs = [
        output_root / "00_README" / "INPUT_MANIFEST.tsv",
        output_root / "00_README" / "SOFTWARE_VERSIONS.txt",
        output_root / "01_FROZEN_INPUTS" / "subject_table" / "current64_subject_table.tsv",
        output_root / "01_FROZEN_INPUTS" / "covariates" / "current64_covariates.tsv",
        output_root / "01_FROZEN_INPUTS" / "behavior" / "current64_behavior_final_scores.tsv",
        output_root / "01_FROZEN_INPUTS" / "behavior" / "current64_behavior_missingness.tsv",
        output_root / "01_FROZEN_INPUTS" / "voxel_order" / "canonical_voxel_order_2608.tsv",
        output_root / "01_FROZEN_INPUTS" / "roi_order" / "canonical_roi_order_65.tsv",
        output_root / "03_FC_FINGERPRINTS" / "subject_fingerprint_qc.tsv",
        output_root / "04_BINWISE_FC_PROFILES" / "bin_voxel_counts.tsv",
        output_root / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_FC_long.tsv",
        output_root / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_FC_wide.tsv",
        output_root / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_profile_qc.tsv",
        output_root / "06_REWARD_STAGE" / "subject_bin_stage_FC_long.tsv",
        output_root / "06_REWARD_STAGE" / "historical_vs_rebuilt_stage_table.tsv",
        output_root / "06_REWARD_STAGE" / "historical_vs_rebuilt_stage_summary.tsv",
        output_root / "16_VALIDATION" / "roi_stage_mapping_qc.tsv",
        output_root / "16_VALIDATION" / "stage1_steps1_to_5_summary.json",
    ]
    planned_outputs.extend(
        output_root / "01_FROZEN_INPUTS" / "fc_fingerprints_2608x65" / f"{subject}_fingerprint_z_2608x65.npy"
        for subject in subject_order
    )
    refuse_existing(planned_outputs)

    manifest_rows: list[dict[str, object]] = []
    logging.info("Writing canonical tables and freezing exact source files")

    subject_table_path = output_root / "01_FROZEN_INPUTS" / "subject_table" / "current64_subject_table.tsv"
    covariate_path = output_root / "01_FROZEN_INPUTS" / "covariates" / "current64_covariates.tsv"
    behavior_path = output_root / "01_FROZEN_INPUTS" / "behavior" / "current64_behavior_final_scores.tsv"
    behavior_missingness_path = output_root / "01_FROZEN_INPUTS" / "behavior" / "current64_behavior_missingness.tsv"
    voxel_order_path = output_root / "01_FROZEN_INPUTS" / "voxel_order" / "canonical_voxel_order_2608.tsv"
    roi_order_path = output_root / "01_FROZEN_INPUTS" / "roi_order" / "canonical_roi_order_65.tsv"
    roi_subfamily_path = output_root / "01_FROZEN_INPUTS" / "roi_subfamily_mapping" / "roi_subfamily_mapping_65.tsv"
    reward_mapping_path = output_root / "01_FROZEN_INPUTS" / "reward_stage_mapping" / "reward_stage_mapping_65.tsv"

    write_tsv(subject_table, subject_table_path)
    write_tsv(covariates, covariate_path)
    write_tsv(behavior, behavior_path)
    write_tsv(behavior_missingness, behavior_missingness_path)
    write_tsv(voxel_order, voxel_order_path)
    write_tsv(roi_order, roi_order_path)
    write_tsv(roi_order[["roi_index", "roi_name", "subfamily", "atlas_source"]], roi_subfamily_path)
    write_tsv(roi_order[["roi_index", "roi_name", "reward_stage"]], reward_mapping_path)

    register_derived_frozen("canonical_subject_table", subject_table_path, [sources["analysis_master"], sources["group_labels"]], "64 rows", manifest_rows)
    register_derived_frozen("canonical_covariates", covariate_path, [sources["analysis_master"], sources["motion"]], "64 rows", manifest_rows)
    register_derived_frozen("canonical_behavior_final_scores_no_rescoring", behavior_path, [sources["clinical"]], "64 rows x 4 frozen scores", manifest_rows)
    register_derived_frozen("canonical_behavior_missingness_qc", behavior_missingness_path, [sources["clinical"]], "4 rows", manifest_rows)
    register_derived_frozen("canonical_voxel_order_table", voxel_order_path, [sources["mapped_ijk"], sources["mapped_flat"], sources["row_mapping"]], "2608 rows", manifest_rows)
    register_derived_frozen("canonical_roi_order_table", roi_order_path, [sources["roi_names"], sources["roi_metadata"]], "65 rows", manifest_rows)
    register_derived_frozen("canonical_roi_subfamily_mapping", roi_subfamily_path, [sources["roi_names"], sources["roi_metadata"]], "65 rows", manifest_rows)
    register_derived_frozen("canonical_reward_stage_mapping", reward_mapping_path, [sources["roi_names"], sources["roi_metadata"]], "65 rows; 62 assigned + 3 not assigned", manifest_rows)

    exact_copy_specs = [
        ("source_analysis_master", sources["analysis_master"], output_root / "01_FROZEN_INPUTS" / "subject_table" / "source_analysis_master_64.tsv", "64 rows"),
        ("source_legacy_group_labels", sources["group_labels"], output_root / "01_FROZEN_INPUTS" / "subject_table" / "source_current64_group_labels.tsv", "64 rows; codes 3=SZ, 4=HC"),
        ("source_clinical_behavior", sources["clinical"], output_root / "01_FROZEN_INPUTS" / "behavior" / "source_final64_clinical_for_server_v3.tsv", "64 rows"),
        ("source_motion", sources["motion"], output_root / "01_FROZEN_INPUTS" / "covariates" / "source_current64_motion_posttrim.tsv", "64 rows"),
        ("canonical_G1_npy", sources["g1"], output_root / "01_FROZEN_INPUTS" / "canonical_G1" / sources["g1"].name, "2608"),
        ("canonical_bins_definition_A_npy", sources["bins"], output_root / "01_FROZEN_INPUTS" / "canonical_bins" / sources["bins"].name, "2608"),
        ("canonical_bins_definition_A_exact_alias", sources["bin_alias"], output_root / "01_FROZEN_INPUTS" / "canonical_bins" / sources["bin_alias"].name, "2608 rows"),
        ("canonical_bin_summary", sources["bin_summary"], output_root / "01_FROZEN_INPUTS" / "canonical_bins" / sources["bin_summary"].name, "10 rows"),
        ("mapped_voxel_ijk_source", sources["mapped_ijk"], output_root / "01_FROZEN_INPUTS" / "voxel_order" / sources["mapped_ijk"].name, "2608 x 3"),
        ("mapped_voxel_flat_source", sources["mapped_flat"], output_root / "01_FROZEN_INPUTS" / "voxel_order" / sources["mapped_flat"].name, "2608"),
        ("row_mapping_2847_to_2608_source", sources["row_mapping"], output_root / "01_FROZEN_INPUTS" / "voxel_order" / sources["row_mapping"].name, "2608"),
        ("roi_names_exact_order_source", sources["roi_names"], output_root / "01_FROZEN_INPUTS" / "roi_order" / sources["roi_names"].name, "65 lines"),
        ("roi_metadata_stage_subfamily_source", sources["roi_metadata"], output_root / "01_FROZEN_INPUTS" / "reward_stage_mapping" / sources["roi_metadata"].name, "65 rows"),
        ("historical_stage_validation_source", sources["historical_stage"], output_root / "01_FROZEN_INPUTS" / "reward_stage_mapping" / "historical_subject_bin_stage_FC_long.tsv", "1280 rows; validation only"),
        ("historical_fingerprint_manifest_source", sources["fingerprint_manifest"], output_root / "03_FC_FINGERPRINTS" / "source_375matched_fingerprint_manifest.tsv", "64 rows"),
    ]
    for role, source, destination, shape_info in exact_copy_specs:
        copy_frozen(role, source, destination, shape_info, manifest_rows)

    for subject in subject_order:
        destination = output_root / "01_FROZEN_INPUTS" / "fc_fingerprints_2608x65" / f"{subject}_fingerprint_z_2608x65.npy"
        copy_frozen(
            "subject_fingerprint_z_2608x65",
            fingerprint_files[subject],
            destination,
            "2608 x 65",
            manifest_rows,
        )
        assert_true(fp_hashes[subject] == sha256_file(destination), f"Frozen fingerprint differs for {subject}")

    write_tsv(fingerprint_qc, output_root / "03_FC_FINGERPRINTS" / "subject_fingerprint_qc.tsv")
    write_tsv(bin_count_table, output_root / "04_BINWISE_FC_PROFILES" / "bin_voxel_counts.tsv")
    write_tsv(profile_long, output_root / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_FC_long.tsv")
    write_tsv(profile_wide, output_root / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_FC_wide.tsv")
    write_tsv(profile_qc, output_root / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_profile_qc.tsv")
    write_tsv(rebuilt_stage, output_root / "06_REWARD_STAGE" / "subject_bin_stage_FC_long.tsv")
    write_tsv(comparison, output_root / "06_REWARD_STAGE" / "historical_vs_rebuilt_stage_table.tsv")
    write_tsv(comparison_summary, output_root / "06_REWARD_STAGE" / "historical_vs_rebuilt_stage_summary.tsv")
    write_tsv(mapping_qc, output_root / "16_VALIDATION" / "roi_stage_mapping_qc.tsv")

    manifest = pd.DataFrame(manifest_rows).sort_values(["input_role", "frozen_path"]).reset_index(drop=True)
    write_tsv(manifest, output_root / "00_README" / "INPUT_MANIFEST.tsv")

    software = "\n".join(
        [
            f"Generated: {datetime.now().astimezone().isoformat()}",
            f"Platform: {platform.platform()}",
            f"Python: {sys.version.replace(chr(10), ' ')}",
            f"NumPy: {np.__version__}",
            f"pandas: {pd.__version__}",
            f"R: {r_version()}",
            "Steps 1-5 computation backend: Python/NumPy/pandas",
            "R was not used for Steps 1-5.",
        ]
    ) + "\n"
    write_text(software, output_root / "00_README" / "SOFTWARE_VERSIONS.txt")
    write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        output_root / "16_VALIDATION" / "stage1_steps1_to_5_summary.json",
    )

    logging.info("All 64 fingerprints passed shape and finiteness QC")
    logging.info("Rebuilt profile rows: %d", len(profile_long))
    logging.info("Rebuilt stage rows: %d", len(rebuilt_stage))
    logging.info("Historical comparison max absolute difference: %.17g", max_abs_difference)
    logging.info("No inferential statistics or figures were run")
    logging.info("CURRENT64 canonical rerun stage 1 completed")
    if log_path is not None:
        print(f"LOG_PATH={log_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
