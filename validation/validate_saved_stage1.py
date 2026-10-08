#!/usr/bin/env python3
"""Independent saved-output validation for CURRENT64 Steps 1--5."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_VALUE = os.environ.get("STRIATAL_CURRENT64_ROOT")
if not ROOT_VALUE:
    raise RuntimeError("Set STRIATAL_CURRENT64_ROOT to the private canonical-rerun root")
ROOT = Path(ROOT_VALUE).resolve()
OUT_TSV = ROOT / "16_VALIDATION" / "stage1_saved_output_validation.tsv"
OUT_JSON = ROOT / "16_VALIDATION" / "stage1_saved_output_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if OUT_TSV.exists() or OUT_JSON.exists():
        raise FileExistsError("Default no-overwrite policy: saved-output validation already exists")

    fp_qc = pd.read_csv(ROOT / "03_FC_FINGERPRINTS" / "subject_fingerprint_qc.tsv", sep="\t")
    profile = pd.read_csv(ROOT / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_FC_long.tsv", sep="\t")
    wide = pd.read_csv(ROOT / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_FC_wide.tsv", sep="\t")
    stage = pd.read_csv(ROOT / "06_REWARD_STAGE" / "subject_bin_stage_FC_long.tsv", sep="\t")
    comparison = pd.read_csv(ROOT / "06_REWARD_STAGE" / "historical_vs_rebuilt_stage_table.tsv", sep="\t")
    manifest = pd.read_csv(ROOT / "00_README" / "INPUT_MANIFEST.tsv", sep="\t", dtype=str)

    frozen_dir = ROOT / "01_FROZEN_INPUTS" / "fc_fingerprints_2608x65"
    frozen_files = sorted(frozen_dir.glob("sub-*_fingerprint_z_2608x65.npy"))
    qc_hashes = dict(zip(fp_qc["Subject"], fp_qc["sha256"]))
    frozen_hash_match = all(
        sha256_file(path) == qc_hashes[path.name.split("_fingerprint", 1)[0]]
        for path in frozen_files
    )

    stage_from_profile = (
        profile.loc[profile["RewardStage"].isin(["Anticipation", "Outcome"])]
        .groupby(["Subject", "Group", "Bin", "RewardStage"], as_index=False, sort=False)["FC"]
        .mean()
        .rename(columns={"RewardStage": "Stage", "FC": "FC_from_saved_ROI_table"})
    )
    stage_check = stage.merge(
        stage_from_profile,
        on=["Subject", "Group", "Bin", "Stage"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    stage_check["abs_difference"] = (stage_check["FC"] - stage_check["FC_from_saved_ROI_table"]).abs()
    stage_roundtrip_max = float(stage_check["abs_difference"].max())

    exact_copies = manifest[manifest["status"] == "FROZEN_EXACT_COPY"]
    exact_copy_manifest_pass = bool(
        (exact_copies["source_sha256"] == exact_copies["frozen_sha256"]).all()
    )
    later_dirs = [
        "07_STAGE_GRADIENT_MODELS",
        "08_CATEGORICAL_SHAPE",
        "09_INDIVIDUAL_SLOPES",
        "10_BRAIN_BEHAVIOR",
        "11_NORMATIVE_FC_DEVIATION",
        "12_BIN_POSTHOC",
        "13_BIN10_DECOMPOSITION",
        "14_R_FIGURES/png",
        "14_R_FIGURES/pdf",
        "14_R_FIGURES/svg",
        "15_FINAL_TABLES",
    ]
    later_output_file_count = sum(
        1 for directory in later_dirs for path in (ROOT / directory).rglob("*") if path.is_file()
    )

    checks = [
        ("fingerprint_qc_rows", len(fp_qc), 64, len(fp_qc) == 64),
        ("fingerprint_all_pass", bool(fp_qc["pass_qc"].all()), True, bool(fp_qc["pass_qc"].all())),
        ("fingerprint_total_na", int(fp_qc["n_na"].sum()), 0, int(fp_qc["n_na"].sum()) == 0),
        ("fingerprint_total_inf", int(fp_qc["n_inf"].sum()), 0, int(fp_qc["n_inf"].sum()) == 0),
        ("frozen_fingerprint_count", len(frozen_files), 64, len(frozen_files) == 64),
        ("frozen_fingerprint_hashes_match", frozen_hash_match, True, frozen_hash_match),
        ("profile_long_rows", len(profile), 41600, len(profile) == 41600),
        ("profile_unique_keys", int(profile.drop_duplicates(["Subject", "Bin", "ROI"]).shape[0]), 41600, not profile.duplicated(["Subject", "Bin", "ROI"]).any()),
        ("profile_finite_fc", bool(np.isfinite(profile["FC"]).all()), True, bool(np.isfinite(profile["FC"]).all())),
        ("profile_wide_rows", len(wide), 640, len(wide) == 640),
        ("profile_wide_columns", len(wide.columns), 68, len(wide.columns) == 68),
        ("stage_rows", len(stage), 1280, len(stage) == 1280),
        ("stage_unique_keys", int(stage.drop_duplicates(["Subject", "Bin", "Stage"]).shape[0]), 1280, not stage.duplicated(["Subject", "Bin", "Stage"]).any()),
        ("stage_roundtrip_rows", len(stage_check), 1280, len(stage_check) == 1280 and bool((stage_check["_merge"] == "both").all())),
        ("stage_roundtrip_max_abs_difference", stage_roundtrip_max, "<=1e-15", stage_roundtrip_max <= 1e-15),
        ("historical_comparison_rows", len(comparison), 1280, len(comparison) == 1280),
        ("historical_max_abs_difference", float(comparison["AbsDifference"].max()), "<=1e-15", float(comparison["AbsDifference"].max()) <= 1e-15),
        ("input_manifest_exact_copy_hashes", exact_copy_manifest_pass, True, exact_copy_manifest_pass),
        ("later_module_output_file_count", later_output_file_count, 0, later_output_file_count == 0),
    ]
    result = pd.DataFrame(checks, columns=["check", "observed", "expected", "pass"])
    all_pass = bool(result["pass"].all())
    result.to_csv(OUT_TSV, sep="\t", index=False)
    OUT_JSON.write_text(
        json.dumps(
            {
                "all_pass": all_pass,
                "n_checks": len(result),
                "n_failed": int((~result["pass"]).sum()),
                "stage_roundtrip_max_abs_difference": stage_roundtrip_max,
                "historical_max_abs_difference": float(comparison["AbsDifference"].max()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    print(f"ALL_PASS={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
