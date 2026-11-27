from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _stage_coupling_common import (
    BIN10_STAGE,
    BRAIN_BEHAVIOR,
    OUT_DIR,
    SUBJECT_DATA,
    distribution_summary,
    ensure_output_dirs,
    read_tsv,
    require,
    zscore,
)


def main() -> None:
    ensure_output_dirs()
    require((OUT_DIR / "stage_coupling_data_audit.csv").is_file(), "Run the audit script first")

    bb = read_tsv(BRAIN_BEHAVIOR).rename(columns={"Subject": "subject", "Group": "group"})
    bin10 = read_tsv(BIN10_STAGE)
    bin_wide = (
        bin10.pivot(index=["subject", "group"], columns="stage", values="profile_z")
        .reset_index()
        .rename(
            columns={
                "Anticipation": "Bin10_Ant_similarity_fisher_z",
                "Outcome": "Bin10_Out_similarity_fisher_z",
            }
        )
    )
    data = bb.loc[bb["group"].eq("SZ")].merge(
        bin_wide, on=["subject", "group"], how="left", validate="one_to_one"
    )
    require(len(data) == 32 and data["subject"].nunique() == 32, "Expected 32 unique SZ participants")
    required = [
        "TEPS_anticipatory",
        "TEPS_consummatory",
        "Anticipation_slope",
        "Outcome_slope",
        "Bin10_Ant_similarity_fisher_z",
        "Bin10_Out_similarity_fisher_z",
    ]
    data["eligible_stage_coupling"] = data[required].notna().all(axis=1)
    require(data["eligible_stage_coupling"].sum() == 28, "Expected 28 complete SZ stage-coupling cases")

    eligible = data["eligible_stage_coupling"]
    standardized = {
        "TEPS_Ant_z": "TEPS_anticipatory",
        "TEPS_Cons_z": "TEPS_consummatory",
        "Ant_slope_z": "Anticipation_slope",
        "Out_slope_z": "Outcome_slope",
        "Bin10_Ant_similarity_z": "Bin10_Ant_similarity_fisher_z",
        "Bin10_Out_similarity_z": "Bin10_Out_similarity_fisher_z",
    }
    for output, source in standardized.items():
        data[output] = np.nan
        data.loc[eligible, output] = zscore(data.loc[eligible, source]).to_numpy()

    data["TEPS_stage_contrast"] = data["TEPS_Ant_z"] - data["TEPS_Cons_z"]
    data["NeuralSlope_stage_contrast"] = data["Ant_slope_z"] - data["Out_slope_z"]
    data["NeuralSlope_raw_difference"] = data["Anticipation_slope"] - data["Outcome_slope"]
    data["Bin10Similarity_stage_contrast"] = (
        data["Bin10_Ant_similarity_z"] - data["Bin10_Out_similarity_z"]
    )
    data["Bin10Similarity_raw_difference"] = (
        data["Bin10_Ant_similarity_fisher_z"] - data["Bin10_Out_similarity_fisher_z"]
    )

    data["TEPS_Ant_theoretical_floor"] = data["TEPS_anticipatory"].eq(9)
    data["TEPS_Ant_theoretical_ceiling"] = data["TEPS_anticipatory"].eq(54)
    data["TEPS_Cons_theoretical_floor"] = data["TEPS_consummatory"].eq(10)
    data["TEPS_Cons_theoretical_ceiling"] = data["TEPS_consummatory"].eq(60)
    data = data.sort_values("subject").reset_index(drop=True)
    data.to_csv(SUBJECT_DATA, index=False, float_format="%.12g")

    variables = [
        "TEPS_anticipatory",
        "TEPS_consummatory",
        "TEPS_Ant_z",
        "TEPS_Cons_z",
        "TEPS_stage_contrast",
        "Anticipation_slope",
        "Outcome_slope",
        "NeuralSlope_stage_contrast",
        "Bin10_Ant_similarity_fisher_z",
        "Bin10_Out_similarity_fisher_z",
        "Bin10Similarity_stage_contrast",
    ]
    summary = distribution_summary(data.loc[eligible], variables)
    summary.to_csv(OUT_DIR / "stage_coupling_distribution_audit.csv", index=False)
    print(f"Built stage contrasts for {eligible.sum()} complete SZ participants (32 retained in audit table)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
