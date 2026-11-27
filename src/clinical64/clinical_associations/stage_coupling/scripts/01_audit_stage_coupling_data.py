from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _stage_coupling_common import (
    BIN10_STAGE,
    BRAIN_BEHAVIOR,
    CLINICAL_IMAGING,
    FROZEN_BEHAVIOR,
    INDIVIDUAL_SLOPES,
    OUT_DIR,
    RAW_ITEM_TABLE,
    ROI_BEHAVIOR,
    ensure_output_dirs,
    read_tsv,
    require,
    sha256,
    write_json,
)


def main() -> None:
    ensure_output_dirs()
    inputs = [
        BRAIN_BEHAVIOR,
        INDIVIDUAL_SLOPES,
        FROZEN_BEHAVIOR,
        BIN10_STAGE,
        CLINICAL_IMAGING,
        ROI_BEHAVIOR,
        RAW_ITEM_TABLE,
    ]
    for path in inputs:
        require(path.is_file(), f"Missing required input: {path}")

    bb = read_tsv(BRAIN_BEHAVIOR)
    slopes = read_tsv(INDIVIDUAL_SLOPES)
    behavior = read_tsv(FROZEN_BEHAVIOR)
    bin10 = read_tsv(BIN10_STAGE)
    clinical = read_tsv(CLINICAL_IMAGING)
    roi = read_tsv(ROI_BEHAVIOR)

    audits: list[dict[str, str | int | float]] = []

    def add(check: str, passed: bool, details: str, source: Path) -> None:
        audits.append(
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "details": details,
                "source": str(source),
                "sha256": sha256(source),
            }
        )
        require(passed, f"Audit failed: {check}: {details}")

    add(
        "Frozen brain-behavior cohort",
        len(bb) == 64 and bb["Subject"].nunique() == 64
        and bb["Group"].value_counts().to_dict() == {"SZ": 32, "HC": 32},
        f"rows={len(bb)}; unique subjects={bb['Subject'].nunique()}; groups={bb['Group'].value_counts().to_dict()}",
        BRAIN_BEHAVIOR,
    )
    add(
        "Overall slope completeness",
        bb[["Anticipation_slope", "Outcome_slope"]].notna().all().all(),
        "Both stage slopes are present for every frozen participant.",
        BRAIN_BEHAVIOR,
    )
    slope_compare = bb.merge(
        slopes[["Subject", "Anticipation_slope", "Outcome_slope"]],
        on="Subject",
        suffixes=("_bb", "_canonical"),
        validate="one_to_one",
    )
    slopes_identical = all(
        np.allclose(
            slope_compare[f"{stage}_slope_bb"],
            slope_compare[f"{stage}_slope_canonical"],
            rtol=0,
            atol=1e-14,
        )
        for stage in ["Anticipation", "Outcome"]
    )
    add(
        "Overall slopes match canonical individual-slope output",
        slopes_identical,
        "Values match the saved canonical subject-stage OLS slopes to 1e-14.",
        INDIVIDUAL_SLOPES,
    )

    b_compare = bb.merge(behavior, on=["Subject", "Group"], suffixes=("_bb", "_frozen"), validate="one_to_one")
    teps_identical = all(
        np.allclose(
            b_compare[f"{name}_bb"],
            b_compare[f"{name}_frozen"],
            equal_nan=True,
        )
        for name in ["TEPS_anticipatory", "TEPS_consummatory"]
    )
    add(
        "TEPS values match frozen behavior table",
        teps_identical,
        "Individual TEPS Anticipatory and Consummatory values match exactly, including missingness.",
        FROZEN_BEHAVIOR,
    )

    roi_compare = bb.merge(
        roi[["subject", "TEPS_anticipatory_calc", "TEPS_consummatory_calc"]],
        left_on="Subject",
        right_on="subject",
        validate="one_to_one",
    )
    roi_teps_identical = np.allclose(
        roi_compare["TEPS_anticipatory"], roi_compare["TEPS_anticipatory_calc"], equal_nan=True
    ) and np.allclose(
        roi_compare["TEPS_consummatory"], roi_compare["TEPS_consummatory_calc"], equal_nan=True
    )
    add(
        "TEPS values match prior ROI-behavior extension",
        roi_teps_identical,
        "The same participant-level TEPS values and frozen imaging IDs are used.",
        ROI_BEHAVIOR,
    )

    sz = bb.loc[bb["Group"].eq("SZ")].copy()
    eligible = sz.dropna(subset=["TEPS_anticipatory", "TEPS_consummatory"])
    missing_ant = sorted(sz.loc[sz["TEPS_anticipatory"].isna(), "Subject"])
    missing_cons = sorted(sz.loc[sz["TEPS_consummatory"].isna(), "Subject"])
    add(
        "SZ TEPS stage-pair availability",
        len(eligible) == 28 and missing_ant == missing_cons,
        f"complete stage pairs=28/32; missing both={','.join(missing_ant)}",
        BRAIN_BEHAVIOR,
    )

    add(
        "Bin 10 stage table structure",
        len(bin10) == 128
        and bin10["subject"].nunique() == 64
        and set(bin10["stage"]) == {"Anticipation", "Outcome"}
        and set(bin10["bin"]) == {10}
        and not bin10.duplicated(["subject", "stage"]).any(),
        "64 participants x 2 stages; Bin=10 only; no duplicate subject-stage rows.",
        BIN10_STAGE,
    )
    roi_counts = bin10.groupby("stage")["n_roi"].unique().to_dict()
    add(
        "Bin 10 stage-specific ROI definitions",
        roi_counts.get("Anticipation", np.array([])).tolist() == [43]
        and roi_counts.get("Outcome", np.array([])).tolist() == [19],
        f"stage ROI counts={{{', '.join(f'{k}: {v.tolist()}' for k, v in roi_counts.items())}}}",
        BIN10_STAGE,
    )
    sz_bin10 = bin10.loc[bin10["group"].eq("SZ")]
    add(
        "SZ Bin 10 HC reference",
        set(sz_bin10["reference"]) == {"HC32"}
        and sz_bin10["profile_z"].notna().all(),
        "Every SZ stage value uses the all-32-HC reference and has Fisher-z similarity.",
        BIN10_STAGE,
    )

    pivot = bin10.pivot(index=["subject", "group"], columns="stage", values="profile_z").reset_index()
    sim_compare = clinical.merge(pivot, on=["subject", "group"], validate="one_to_one")
    sim_identical = np.allclose(
        sim_compare["bin10_anticipation_profile_z"], sim_compare["Anticipation"], atol=1e-14
    ) and np.allclose(
        sim_compare["bin10_outcome_profile_z"], sim_compare["Outcome"], atol=1e-14
    )
    add(
        "Bin 10 values match clinical-extension dataset",
        sim_identical,
        "Both stage-specific Fisher-z similarity columns match exactly.",
        CLINICAL_IMAGING,
    )

    raw = read_tsv(RAW_ITEM_TABLE)
    teps_item_cols = [f"TEPS_item{i:02d}_raw" for i in range(1, 21)]
    item_values = raw[teps_item_cols].stack().dropna()
    add(
        "TEPS item scale audit",
        set(item_values.unique()) <= {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
        "Observed item responses are integers 1-6; Anticipatory sums 9 items (range 9-54) and Consummatory sums 10 (range 10-60).",
        RAW_ITEM_TABLE,
    )

    audit_df = pd.DataFrame(audits)
    audit_df.to_csv(OUT_DIR / "stage_coupling_data_audit.csv", index=False)

    metadata = {
        "effective_sz_n": int(len(eligible)),
        "missing_teps_stage_pair_ids": missing_ant,
        "behavior": {
            "anticipatory_variable": "TEPS_anticipatory",
            "consummatory_variable": "TEPS_consummatory",
            "direction": "higher retained score indicates greater pleasure endorsement",
            "anticipatory_items": 9,
            "consummatory_items": 10,
            "item_response_range": [1, 6],
            "theoretical_ranges": {"anticipatory": [9, 54], "consummatory": [10, 60]},
        },
        "overall_slope": {
            "definition": "Within-person OLS slope of stage-specific mean raw FC across ten fixed G1 bins using G1_bin_center = Bin - 5.5.",
            "same_definition_across_stages": True,
            "fixed_G1": True,
            "direction": "Positive values indicate increasing FC from low-G1 VST/caudate-rich bins toward high-G1 putamen-rich bins; negative values indicate decreasing FC.",
        },
        "bin10_similarity": {
            "definition": "Fisher z of Pearson correlation between the participant's Bin 10 stage-specific FC fingerprint and the HC reference fingerprint.",
            "anticipation_roi_count": 43,
            "outcome_roi_count": 19,
            "sz_reference": "mean of all 32 HC participants",
            "same_similarity_definition_across_stages": True,
        },
    }
    write_json(OUT_DIR / "stage_coupling_audit_metadata.json", metadata)
    print(f"Audit complete: {len(audit_df)} checks passed; eligible SZ N={len(eligible)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
