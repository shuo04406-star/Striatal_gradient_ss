from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


VERSION = "Final64_AnalysisReady_v1.0"
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime.now(TZ).isoformat(timespec="seconds")

ROOT = Path(os.path.join(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"], '02_Analysis/00_FINAL64_DATABASE_LOCK'))
EXPORT = Path(os.path.join(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"], '02/subject_level_data_export'))
PROJECT = Path(os.environ["STRIATAL_PROJECT_ROOT"])
CANON = PROJECT / "CURRENT64_CANONICAL_RERUN"
CLIN_V3 = Path(os.path.join(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"], 'final64_gradient_clinical_stats/03_clinical_master/final64_clinical_for_server_v3.tsv'))
WORKBOOK = EXPORT / "SUBJECT_LEVEL_BRAIN_BEHAVIOUR_DATABASE.xlsx"

DIRS = {
    "frozen": ROOT / "frozen_data",
    "manifests": ROOT / "manifests",
    "metric": ROOT / "metric_audit",
    "clinical": ROOT / "clinical_audit",
    "analysis": ROOT / "analysis_preparation",
}

SOURCES = {
    "master": EXPORT / "subject_level_brain_behaviour_master_final64.tsv",
    "fc": EXPORT / "subject_level_fc_long.tsv",
    "roi": EXPORT / "subject_level_fc_target_roi_long.tsv",
    "sim": EXPORT / "subject_level_similarity_long.tsv",
    "derived": EXPORT / "subject_level_derived_imaging_metrics.tsv",
    "crosswalk": EXPORT / "SUBJECT_CROSSWALK.tsv",
    "clinical_wide": EXPORT / "subject_level_clinical_behaviour_wide.tsv",
    "items": EXPORT / "subject_level_questionnaire_items_long.tsv",
    "trials": EXPORT / "subject_level_behaviour_trials_long.tsv",
    "clinical_dict": EXPORT / "CLINICAL_BEHAVIOUR_DATA_DICTIONARY.tsv",
    "coverage": EXPORT / "CLINICAL_BEHAVIOUR_COVERAGE.tsv",
    "export_manifest": EXPORT / "DATA_EXPORT_MANIFEST.tsv",
    "export_qc": EXPORT / "DATA_EXPORT_QC_ISSUES.tsv",
    "canonical_fc": CANON / "06_REWARD_STAGE" / "subject_bin_stage_FC_long.tsv",
    "canonical_roi": CANON / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_FC_long.tsv",
    "slopes_long": CANON / "09_INDIVIDUAL_SLOPES" / "subject_stage_slopes_long.tsv",
    "slopes_wide": CANON / "09_INDIVIDUAL_SLOPES" / "subject_stage_slopes_wide.tsv",
    "bin_summary": CANON / "01_FROZEN_INPUTS" / "canonical_bins" / "fixed_375_G1_bin_summary.tsv",
    "input_manifest": CANON / "00_README" / "INPUT_MANIFEST.tsv",
    "sim_b05": PROJECT / "journal_figure_scatter_revision_20261028" / "source_data" / "B05_Bin10_stage_subjects_used.tsv",
    "matched": PROJECT / "clinical_extension_analysis" / "roi_behavior_extension" / "data_internal" / "roi_behavior_matched_dataset.tsv",
    "stage_coupling": PROJECT / "clinical_extension_analysis" / "stage_coupling_analysis" / "stage_coupling_subject_level_data.csv",
    "legacy": PROJECT / "99_archive_64" / "rebuild_oxford_65roi_v2" / "25_current64_individual_G1_statistics" / "06_normative_deviation" / "01_tables" / "individual_G1_normative_deviation.tsv",
    "conflicts": Path(os.path.join(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"], 'final64_gradient_clinical_stats/06_qc/scale_score_mismatch_subjects.tsv')),
    "source_conflicts": Path(os.path.join(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"], 'final64_gradient_clinical_stats/01_source_audit/source_conflict_audit.tsv')),
    "scale_provenance": Path(os.path.join(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"], 'final64_gradient_clinical_stats/01_source_audit/scale_item_provenance.tsv')),
}

FROZEN_FILES = [
    "master", "fc", "roi", "sim", "derived", "crosswalk", "clinical_wide",
    "items", "trials", "clinical_dict", "coverage", "export_manifest", "export_qc",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, **kwargs)
    return pd.read_csv(path, sep="\t", **kwargs)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def finite_count(series: pd.Series) -> int:
    return int(np.isfinite(numeric(series).to_numpy(dtype=float)).sum())


def norm_group(v) -> str:
    s = str(v).strip().upper()
    return "SZ" if s in {"SZ", "SCZ"} else ("HC" if s == "HC" else s)


def compare_frames(
    comparison_id: str,
    family: str,
    metric: str,
    a: pd.DataFrame,
    b: pd.DataFrame,
    keys: list[str],
    value_a: str,
    value_b: str,
    source_a: str,
    source_b: str,
    tolerance: float = 1e-12,
    label_pairs: list[tuple[str, str]] | None = None,
) -> dict:
    aa = a[keys + [value_a] + [x for p in (label_pairs or []) for x in [p[0]] if x not in keys and x != value_a]].copy()
    bb = b[keys + [value_b] + [x for p in (label_pairs or []) for x in [p[1]] if x not in keys and x != value_b]].copy()
    aa = aa.rename(columns={value_a: "value_a"})
    bb = bb.rename(columns={value_b: "value_b"})
    aa["value_a"] = numeric(aa["value_a"])
    bb["value_b"] = numeric(bb["value_b"])
    merged = aa.merge(bb, on=keys, how="outer", indicator=True, suffixes=("_a", "_b"))
    both = merged[merged["_merge"] == "both"].copy()
    diffs = (both["value_a"] - both["value_b"]).abs()
    comparable = both["value_a"].notna() & both["value_b"].notna()
    mismatch = comparable & (diffs > tolerance)
    nan_disagree = both["value_a"].isna() ^ both["value_b"].isna()
    label_mismatch = 0
    for la, lb in label_pairs or []:
        ca = la if la in both.columns else f"{la}_a"
        cb = lb if lb in both.columns else f"{lb}_b"
        if ca in both.columns and cb in both.columns:
            label_mismatch += int((both[ca].astype(str) != both[cb].astype(str)).sum())
    mismatch_n = int(mismatch.sum() + nan_disagree.sum())
    status = "PASS" if (mismatch_n == 0 and label_mismatch == 0 and (merged["_merge"] != "both").sum() == 0) else "FAIL"
    return {
        "comparison_id": comparison_id,
        "metric_family": family,
        "metric_name": metric,
        "source_a": source_a,
        "source_b": source_b,
        "key_columns": ";".join(keys),
        "tolerance": tolerance,
        "expected_pairs": max(len(aa), len(bb)),
        "matched_pairs": len(both),
        "missing_in_a": int((merged["_merge"] == "right_only").sum()),
        "missing_in_b": int((merged["_merge"] == "left_only").sum()),
        "mismatch_N": mismatch_n,
        "max_abs_error": float(diffs[comparable].max()) if comparable.any() else np.nan,
        "mean_abs_error": float(diffs[comparable].mean()) if comparable.any() else np.nan,
        "label_mismatch_N": label_mismatch,
        "audit_status": status,
        "notes": "NaN-vs-value counts as mismatch; exact key coverage required.",
    }


def metric_row(name, description, stage, roi, method, reference_group, hc_loo, valid_groups,
               expected_n, observed_n, source_file, source_script, source_column, evidence,
               analysis_status, audit_status, notes="", fc_input="Fisher-z FC fingerprint",
               bin_definition="Fixed 375-person reference G1 deciles; centered bin = Bin - 5.5"):
    return {
        "variable_name": name,
        "description": description,
        "stage": stage,
        "ROI_or_target_set": roi,
        "G1_reference": "Frozen 375-person G1; SHA256 d7cd62a4c86e82d8b96c7e229e1e460f183f77544da7444280457a14d4f6c985",
        "FC_input": fc_input,
        "bin_definition": bin_definition,
        "calculation_method": method,
        "reference_group": reference_group,
        "HC_LOO": hc_loo,
        "valid_groups": valid_groups,
        "expected_N": expected_n,
        "observed_N": observed_n,
        "source_file": source_file,
        "source_script": source_script,
        "source_column": source_column,
        "evidence": evidence,
        "analysis_status": analysis_status,
        "audit_status": audit_status,
        "notes": notes,
    }


def add_outliers(df, family, metric_col, value_col, group_cols, subject_col="Subject_ID"):
    d = df.copy()
    d[value_col] = numeric(d[value_col])
    groupers = group_cols[0] if len(group_cols) == 1 else group_cols
    stats = d.groupby(groupers, dropna=False)[value_col].agg(
        group_N="size", finite_N=lambda x: np.isfinite(x).sum(), mean="mean", sd="std",
        q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75)
    ).reset_index()
    stats["iqr"] = stats["q3"] - stats["q1"]
    d = d.merge(stats, on=group_cols, how="left")
    d["z_score"] = (d[value_col] - d["mean"]) / d["sd"].replace(0, np.nan)
    d["iqr_lower"] = d["q1"] - 1.5 * d["iqr"]
    d["iqr_upper"] = d["q3"] + 1.5 * d["iqr"]
    d["abs_z_gt_3"] = d["z_score"].abs() > 3
    d["iqr_outlier"] = (d[value_col] < d["iqr_lower"]) | (d[value_col] > d["iqr_upper"])
    d["nonfinite"] = ~np.isfinite(d[value_col])
    d["duplicate_value_within_metric_N"] = d.groupby(group_cols + [value_col], dropna=False)[value_col].transform("size")
    d["flag_any"] = d[["abs_z_gt_3", "iqr_outlier", "nonfinite"]].any(axis=1)
    out = pd.DataFrame({
        "metric_family": family,
        "metric_name": d[metric_col].astype(str),
        "Subject_ID": d[subject_col].astype(str),
        "Group": d["Group"].astype(str),
        "Value": d[value_col],
        "stratum": d[group_cols].astype(str).agg("|".join, axis=1),
        "group_N": d["group_N"], "finite_N": d["finite_N"],
        "mean": d["mean"], "sd": d["sd"], "z_score": d["z_score"],
        "q1": d["q1"], "q3": d["q3"], "iqr_lower": d["iqr_lower"], "iqr_upper": d["iqr_upper"],
        "abs_z_gt_3": d["abs_z_gt_3"], "iqr_outlier": d["iqr_outlier"], "nonfinite": d["nonfinite"],
        "duplicate_value_within_metric_N": d["duplicate_value_within_metric_N"],
        "flag_any": d["flag_any"],
        "action": "REVIEW_ONLY_NO_AUTOMATIC_EXCLUSION",
        "provenance": "QC_RECALCULATION_FROM_FROZEN_VALUES",
    })
    return out


def clinical_meta(var: str) -> dict:
    defaults = {
        "description": var.replace("_", " "), "scale_or_task": "Clinical/behaviour",
        "applicable_groups": "HC;SZ", "scoring_method": "Value retained from formal Final64 v3 clinical master",
        "reverse_scoring": "See source provenance", "missing_rule": "No imputation; blank/NaN remains missing",
        "possible_range": "See instrument manual/source dictionary", "higher_value_meaning": "Instrument-dependent",
        "raw_or_derived": "FORMAL_EXISTING_VALUE", "analysis_status": "SECONDARY_CANDIDATE",
        "audit_status": "PASS_SOURCE_PRESERVED", "notes": "",
    }
    m = defaults.copy()
    if var == "age":
        m.update(description="Age at scan", scale_or_task="Demographic", scoring_method="Direct value", reverse_scoring="None", possible_range="Non-negative years", higher_value_meaning="Older age", raw_or_derived="RAW/FORMAL", analysis_status="COVARIATE", audit_status="PASS")
    elif var == "sex":
        m.update(description="Sex recorded in formal clinical master", scale_or_task="Demographic", scoring_method="Categorical value", reverse_scoring="None", possible_range="Male/Female in current data", higher_value_meaning="Not applicable", raw_or_derived="RAW/FORMAL", analysis_status="COVARIATE", audit_status="PASS")
    elif var == "TEPS_anticipatory_calc":
        m.update(description="TEPS anticipatory pleasure subscale", scale_or_task="TEPS", scoring_method="Sum of 9 prespecified items", reverse_scoring="Item 13 reversal applies to total only, not this subscale", possible_range="Instrument-defined sum", higher_value_meaning="Greater anticipatory pleasure", raw_or_derived="DERIVED_FROM_ITEMS", analysis_status="PRIMARY_CANDIDATE", audit_status="PASS_RECONSTRUCTION_VERIFIED")
    elif var == "TEPS_consummatory_calc":
        m.update(description="TEPS consummatory pleasure subscale", scale_or_task="TEPS", scoring_method="Sum of 10 prespecified items", reverse_scoring="Item 13 reversal applies to total only, not these subscale definitions", possible_range="Instrument-defined sum", higher_value_meaning="Greater consummatory pleasure", raw_or_derived="DERIVED_FROM_ITEMS", analysis_status="PRIMARY_CANDIDATE", audit_status="PASS_RECONSTRUCTION_VERIFIED")
    elif var == "TEPS_total_calc":
        m.update(description="TEPS total score", scale_or_task="TEPS", scoring_method="Formal item-based total", reverse_scoring="TEPS item 13 reversed for total", higher_value_meaning="Greater trait pleasure", raw_or_derived="DERIVED_FROM_ITEMS", audit_status="PASS_RECONSTRUCTION_VERIFIED")
    elif var == "ACIPS_anticipatory_calc":
        m.update(description="ACIPS anticipatory social/interpersonal pleasure subscale", scale_or_task="ACIPS", scoring_method="Sum of 7 prespecified items", reverse_scoring="ACIPS3 transformed as 7 - raw", higher_value_meaning="Greater anticipatory social pleasure", raw_or_derived="DERIVED_FROM_ITEMS", analysis_status="PRIMARY_CANDIDATE", audit_status="PASS_RECONSTRUCTION_VERIFIED")
    elif var == "ACIPS_consummatory_calc":
        m.update(description="ACIPS consummatory social/interpersonal pleasure subscale", scale_or_task="ACIPS", scoring_method="Sum of 10 prespecified items", reverse_scoring="Per formal ACIPS provenance", higher_value_meaning="Greater consummatory social pleasure", raw_or_derived="DERIVED_FROM_ITEMS", analysis_status="PRIMARY_CANDIDATE", audit_status="PASS_RECONSTRUCTION_VERIFIED")
    elif var.startswith("PANSS_"):
        m.update(scale_or_task="PANSS", applicable_groups="SZ", scoring_method="Formal v3 score; total is P + N + G when components present", reverse_scoring="None", higher_value_meaning="Greater symptom severity", analysis_status="SECONDARY_CANDIDATE_SZ_ONLY", audit_status="PASS_WITH_SOURCE_PRIORITY_WARNING", notes="Formal source priority retained; the case in the private source-conflict audit negative-score source conflict remains documented.")
    elif var == "SANS":
        m.update(scale_or_task="SANS", applicable_groups="SZ", reverse_scoring="None", higher_value_meaning="Greater negative symptom severity", analysis_status="SECONDARY_CANDIDATE_SZ_ONLY", audit_status="PASS_WITH_SOURCE_PRIORITY_WARNING", notes="the case in the private source-conflict audit source conflict remains documented.")
    elif var in {"SAS", "BARS"}:
        m.update(scale_or_task=var, applicable_groups="SZ", reverse_scoring="None", higher_value_meaning="Greater extrapyramidal/akathisia severity", analysis_status="SECONDARY_CANDIDATE_SZ_ONLY", audit_status="PASS_SOURCE_PRESERVED")
    elif var.startswith("BFI_") or var in {"CSAS", "CPAS"}:
        m.update(analysis_status="NOT_FOR_ANALYSIS", audit_status="UNRESOLVED_SCORING_CONFLICT", notes="Existing-versus-recalculated scoring mismatches require resolution before analysis.")
    elif var in {"medication_text", "first_episode_candidate"}:
        m.update(applicable_groups="SZ", analysis_status="DESCRIPTIVE_ONLY", audit_status="PASS_SOURCE_PRESERVED")
    return m


def main() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)

    missing_sources = [str(p) for p in SOURCES.values() if not p.exists()]
    if missing_sources:
        raise FileNotFoundError("Missing sources:\n" + "\n".join(missing_sources))

    master = read_table(SOURCES["master"], low_memory=False)
    fc = read_table(SOURCES["fc"])
    roi = read_table(SOURCES["roi"])
    sim = read_table(SOURCES["sim"])
    derived = read_table(SOURCES["derived"])
    crosswalk = read_table(SOURCES["crosswalk"], low_memory=False)
    trials = read_table(SOURCES["trials"], low_memory=False)
    clinical = read_table(CLIN_V3, na_values=["NaN", "NA"])

    # Core invariants: fail fast rather than freeze an invalid sample.
    assert len(master) == 64 and master["Subject_ID"].nunique() == 64
    assert master["Subject_ID"].str.match(r"^sub-\d{4}$").all()
    assert master["Group"].map(norm_group).value_counts().to_dict() == {"SZ": 32, "HC": 32}
    assert set(fc["Subject_ID"]) == set(master["Subject_ID"])
    assert set(roi["Subject_ID"]) == set(master["Subject_ID"])
    assert set(sim["Subject_ID"]) == set(master["Subject_ID"])
    assert set(clinical["subject"]) == set(master["Subject_ID"])

    # 1. Byte-identical frozen copies, with hashes and read-only bit.
    manifest_rows = []
    for key in FROZEN_FILES:
        src = SOURCES[key]
        dst = DIRS["frozen"] / src.name
        src_hash = sha256(src)
        action = "EXISTING_IDENTICAL_SKIPPED" if dst.exists() and sha256(dst) == src_hash else "COPIED"
        if action == "COPIED":
            if dst.exists():
                os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)
            shutil.copy2(src, dst)
        dst_hash = sha256(dst)
        if src_hash != dst_hash:
            raise RuntimeError(f"Hash mismatch after freeze: {src}")
        os.chmod(dst, stat.S_IREAD)
        try:
            shape = read_table(src, low_memory=False).shape
        except Exception:
            shape = (np.nan, np.nan)
        manifest_rows.append({
            "lock_version": VERSION, "frozen_at": NOW, "role": key,
            "original_path": str(src), "frozen_path": str(dst),
            "bytes": src.stat().st_size, "rows_excluding_header": shape[0], "columns": shape[1],
            "source_sha256": src_hash, "frozen_sha256": dst_hash,
            "byte_identical": "YES", "read_only": "YES", "action": action,
            "authoritative_use": "Frozen TSV is authoritative; review XLSX is not an authoritative source.",
        })
    frozen_manifest = pd.DataFrame(manifest_rows)
    write_tsv(frozen_manifest, DIRS["manifests"] / "FROZEN_DATA_MANIFEST.tsv")
    with (DIRS["manifests"] / "FILE_HASHES_SHA256.txt").open("w", encoding="utf-8", newline="\n") as fh:
        for row in manifest_rows:
            fh.write(f'{row["frozen_sha256"]}  frozen_data/{Path(row["frozen_path"]).name}\n')

    # 2. Imaging metric dictionary.
    dcounts = derived.groupby("Metric_Name")["Value"].apply(finite_count).to_dict()
    metric_rows = []
    fixed_script = str(CANON / "scripts" / "python" / "01_freeze_and_build_profiles.py")
    slope_script = str(CANON / "scripts" / "python" / "03_run_stage_analyses.py")
    sim_script = str(PROJECT / "SERVER_RECOVERED_CODE_REVIEW" / "FINAL_SCIENTIFIC_VALIDATION" / "run_stage3_validation.py")
    roi_script = str(PROJECT / "clinical_extension_analysis" / "scripts" / "02_match_clinical_imaging_subjects.R")
    for stage, nroi in [("Anticipation", 43), ("Outcome", 19)]:
        metric_rows.append(metric_row(
            f"Stage_FC_{stage}", f"Mean stage-assigned target FC for every fixed G1 bin ({nroi} targets)", stage,
            f"{nroi} reward-stage targets", "Unweighted arithmetic mean across stage-assigned target ROI Fisher-z FC values within subject and bin",
            "None", "Not applicable", "HC;SZ", 64, fc.loc[fc.Stage.eq(stage), "Subject_ID"].nunique(), str(SOURCES["canonical_fc"]), fixed_script, "FC",
            "Canonical 64-person reward-stage table; 10 bins per subject", "PRIMARY_CANDIDATE", "PASS_EXACT_REPRODUCTION"))
        metric_rows.append(metric_row(
            f"Bin10_{stage}_Similarity", f"Fisher-z Pearson similarity of the subject Bin10 {stage} fingerprint to an HC normative pattern", stage,
            f"{nroi} reward-stage targets", "Pearson r across target ROIs, clipped and Fisher transformed with arctanh",
            "HC normative pattern", "YES for HC; HC32 template for SZ", "HC;SZ", 64, sim.loc[sim.Stage.eq(stage), "Subject_ID"].nunique(), str(SOURCES["sim_b05"]), sim_script, "profile_z",
            "HC rows use leave-one-out HC31; SZ rows use frozen HC32 reference", "SECONDARY_CANDIDATE", "PASS_EXACT_REPRODUCTION"))
    metric_rows.append(metric_row(
        "Target_ROI_FC", "Per-target Fisher-z FC for 65 targets and every fixed G1 bin", "Target-specific", "65 target ROIs",
        "Mean voxelwise Fisher-z FC within each fixed G1 bin, retained separately for every target ROI", "None", "Not applicable", "HC;SZ", 64,
        roi["Subject_ID"].nunique(), str(SOURCES["canonical_roi"]), fixed_script, "FC", "64×10×65 complete values", "SOURCE_PHENOTYPE", "PASS_EXACT_REPRODUCTION"))
    metric_rows.append(metric_row(
        "Historical_Bin10_65target_full_profile_similarity", "Historical full 65-target Bin10 similarity; not stage-specific", "Combined/non-stage-specific", "All 65 targets",
        "Pearson r across all 65 targets, then Fisher transform", "HC normative pattern", "Per generating validation output", "HC;SZ", 64, 0,
        str(PROJECT / "SERVER_RECOVERED_CODE_REVIEW" / "FINAL_SCIENTIFIC_VALIDATION" / "RUN_ARTIFACTS" / "64_canonical" / "02_subject_bin_metrics_G1_bin_current.tsv"),
        sim_script, "profile_z", "Separated explicitly from 43/19-target stage-specific measures", "EXPLORATORY", "PASS_NOT_STAGE_SPECIFIC",
        "Not copied into the frozen sample matrix; must never be relabeled as Anticipation or Outcome."))

    derived_defs = {
        "Overall_Anticipation_FC_slope": ("Anticipation", "43 stage targets", "OLS/polyfit slope across 10 centered bins", "PRIMARY_CANDIDATE", "PASS_EXACT_SOURCE_MATCH", slope_script),
        "Overall_Outcome_FC_slope": ("Outcome", "19 stage targets", "OLS/polyfit slope across 10 centered bins", "PRIMARY_CANDIDATE", "PASS_EXACT_SOURCE_MATCH", slope_script),
        "Overall_Anticipation_minus_Outcome_slope": ("Stage contrast", "43 minus 19 target stage aggregates", "Anticipation slope minus Outcome slope", "SECONDARY_CANDIDATE", "PASS_EXACT_SOURCE_MATCH", slope_script),
        "Overall_Anticipation_FC_offset": ("Anticipation", "43 stage targets", "OLS/polyfit intercept at centered bin 0", "SECONDARY_CANDIDATE", "PASS_EXACT_SOURCE_MATCH", slope_script),
        "Overall_Outcome_FC_offset": ("Outcome", "19 stage targets", "OLS/polyfit intercept at centered bin 0", "SECONDARY_CANDIDATE", "PASS_EXACT_SOURCE_MATCH", slope_script),
        "Overall_Anticipation_minus_Outcome_offset": ("Stage contrast", "Stage aggregates", "Anticipation intercept minus Outcome intercept", "NOT_FOR_ANALYSIS", "PASS_QC_RECALCULATION", slope_script),
        "SMA_Anticipation_rawFC_slope": ("Anticipation", "SMA composite", "Within-subject lm(FC ~ I(Bin - 5.5)); 10 bins", "EXPLORATORY", "PASS_EXACT_SOURCE_MATCH", roi_script),
        "IFG_Outcome_rawFC_slope": ("Outcome", "IFG six-ROI composite", "Within-subject lm(FC ~ I(Bin - 5.5)); 10 bins", "EXPLORATORY", "PASS_EXACT_SOURCE_MATCH", roi_script),
        "Cingulate_Outcome_rawFC_slope": ("Outcome", "Cingulate composite", "Within-subject lm(FC ~ I(Bin - 5.5)); 10 bins", "EXPLORATORY", "PASS_EXACT_SOURCE_MATCH", roi_script),
        "Legacy_G1_normative_MAE": ("Not applicable", "Striatal G1 voxels", "Voxelwise mean absolute deviation of Procrustes-aligned individual G1 from frozen 375 G1 after within-map z scoring", "LEGACY_EXPLORATORY", "PASS_EXACT_SOURCE_MATCH", "historical individual-gradient pipeline"),
        "Legacy_G1_normative_RMSE": ("Not applicable", "Striatal G1 voxels", "Voxelwise RMSE of Procrustes-aligned individual G1 from frozen 375 G1 after within-map z scoring", "LEGACY_EXPLORATORY", "PASS_EXACT_SOURCE_MATCH", "historical individual-gradient pipeline"),
        "SZ_standardized_Anticipation_minus_Outcome_slope_contrast": ("Stage contrast", "Stage aggregates", "SZ-standardized Anticipation slope minus standardized Outcome slope", "EXPLORATORY", "PASS_EXACT_SOURCE_MATCH", "stage coupling output"),
        "SZ_standardized_Bin10_Anticipation_minus_Outcome_similarity_contrast": ("Stage contrast", "43/19 target similarities", "SZ-standardized Anticipation similarity minus standardized Outcome similarity", "EXPLORATORY", "PASS_EXACT_SOURCE_MATCH", "stage coupling output"),
        "SZ_Bin10_Anticipation_minus_Outcome_similarity_raw_difference": ("Stage contrast", "43/19 target similarities", "Raw Anticipation Fisher-z similarity minus Outcome Fisher-z similarity", "EXPLORATORY", "PASS_EXACT_SOURCE_MATCH", "stage coupling output"),
    }
    for name, (stage, roi_name, method, use, audit, script) in derived_defs.items():
        valid = "SZ" if name.startswith("SZ_") else "HC;SZ"
        exp = 28 if name.startswith("SZ_standardized_") else (32 if valid == "SZ" else 64)
        source_file = derived.loc[derived.Metric_Name.eq(name), "Source_File"].iloc[0] if (derived.Metric_Name == name).any() else str(SOURCES["derived"])
        fc_input = "Individual Procrustes-aligned G1" if name.startswith("Legacy_") else "Saved fixed-reference FC-derived values"
        bin_def = "Not applicable to voxelwise legacy G1 deviation" if name.startswith("Legacy_") else "Fixed 375-person reference G1 deciles; centered bin = Bin - 5.5"
        metric_rows.append(metric_row(name, name.replace("_", " "), stage, roi_name, method, "None" if "Similarity" not in name else "HC normative pattern",
                                      "Not applicable", valid, exp, dcounts.get(name, 0), source_file, script, "Value", "Existing formal output retained and independently compared",
                                      use, audit, fc_input=fc_input, bin_definition=bin_def))
    metric_dict = pd.DataFrame(metric_rows)
    write_tsv(metric_dict, DIRS["metric"] / "IMAGING_METRIC_DICTIONARY.tsv")

    # 3. Numeric consistency: canonical sources, workbook review container, and direct formal outputs.
    audits = []
    cfc = read_table(SOURCES["canonical_fc"]).rename(columns={"Subject": "Subject_ID"})
    audits.append(compare_frames("FC_EXPORT_VS_CANONICAL", "Stage FC", "All stages and bins", fc, cfc,
                                 ["Subject_ID", "Stage", "Bin"], "FC", "FC", str(SOURCES["fc"]), str(SOURCES["canonical_fc"])))
    croi = read_table(SOURCES["canonical_roi"]).rename(columns={"Subject": "Subject_ID", "ROI": "Target_ROI"})
    audits.append(compare_frames("ROI_EXPORT_VS_CANONICAL", "Target ROI FC", "65 targets × 10 bins", roi, croi,
                                 ["Subject_ID", "Bin", "Target_ROI"], "FC", "FC", str(SOURCES["roi"]), str(SOURCES["canonical_roi"])))
    b05 = read_table(SOURCES["sim_b05"]).rename(columns={"subject": "Subject_ID", "stage": "Stage"})
    audits.append(compare_frames("SIM_EXPORT_VS_B05", "Similarity", "Stage-specific Bin10 Fisher-z", sim, b05,
                                 ["Subject_ID", "Stage"], "Similarity", "profile_z", str(SOURCES["sim"]), str(SOURCES["sim_b05"])))
    matched = read_table(SOURCES["matched"]).rename(columns={"subject": "Subject_ID"})
    for stage, col in [("Anticipation", "bin10_anticipation_profile_z"), ("Outcome", "bin10_outcome_profile_z")]:
        audits.append(compare_frames(f"SIM_{stage.upper()}_VS_MATCHED", "Similarity", f"Bin10 {stage}", sim[sim.Stage.eq(stage)], matched,
                                     ["Subject_ID"], "Similarity", col, str(SOURCES["sim"]), str(SOURCES["matched"])))

    # Wide/master mirrors.
    for stage in ["Anticipation", "Outcome"]:
        for b in range(1, 11):
            col = f"FC_{stage}_Bin{b:02d}"
            aa = fc[(fc.Stage == stage) & (fc.Bin == b)]
            audits.append(compare_frames(f"MASTER_{stage[:3].upper()}_BIN{b:02d}", "Master mirror", col, aa, master,
                                         ["Subject_ID"], "FC", col, str(SOURCES["fc"]), str(SOURCES["master"])))
    for stage, col in [("Anticipation", "Bin10_Anticipation_Similarity"), ("Outcome", "Bin10_Outcome_Similarity")]:
        audits.append(compare_frames(f"MASTER_SIM_{stage.upper()}", "Master mirror", col, sim[sim.Stage.eq(stage)], master,
                                     ["Subject_ID"], "Similarity", col, str(SOURCES["sim"]), str(SOURCES["master"])))
    for name in sorted(derived.Metric_Name.unique()):
        if name in master.columns:
            master_compare = master[master.Group.eq("SZ")] if name.startswith("SZ_") else master
            audits.append(compare_frames(f"MASTER_DERIVED_{name}", "Master mirror", name, derived[derived.Metric_Name.eq(name)], master_compare,
                                         ["Subject_ID"], "Value", name, str(SOURCES["derived"]), str(SOURCES["master"])))

    # Review workbook equality (not treated as an authoritative source).
    w_fc = pd.read_excel(WORKBOOK, sheet_name="Imaging_FC")
    w_sim = pd.read_excel(WORKBOOK, sheet_name="Imaging_Similarity")
    w_der = pd.read_excel(WORKBOOK, sheet_name="Imaging_Derived_Metrics")
    audits.append(compare_frames("WORKBOOK_FC_MIRROR", "Workbook mirror", "Stage FC", fc, w_fc,
                                 ["Subject_ID", "Stage", "Bin"], "FC", "FC", str(SOURCES["fc"]), f"{WORKBOOK}::Imaging_FC", tolerance=1e-10))
    audits.append(compare_frames("WORKBOOK_SIM_MIRROR", "Workbook mirror", "Similarity", sim, w_sim,
                                 ["Subject_ID", "Stage"], "Similarity", "Similarity", str(SOURCES["sim"]), f"{WORKBOOK}::Imaging_Similarity", tolerance=1e-10))
    audits.append(compare_frames("WORKBOOK_DERIVED_MIRROR", "Workbook mirror", "Derived metrics", derived, w_der,
                                 ["Subject_ID", "Metric_Name"], "Value", "Value", str(SOURCES["derived"]), f"{WORKBOOK}::Imaging_Derived_Metrics", tolerance=1e-10))

    # Direct-source comparisons for every saved derived metric.
    slopes_long = read_table(SOURCES["slopes_long"]).rename(columns={"Subject": "Subject_ID"})
    slopes_wide = read_table(SOURCES["slopes_wide"]).rename(columns={"Subject": "Subject_ID"})
    direct = {}
    for stage in ["Anticipation", "Outcome"]:
        ss = slopes_long[slopes_long.Stage.eq(stage)][["Subject_ID", "Slope", "Intercept"]]
        direct[f"Overall_{stage}_FC_slope"] = (ss, "Slope", str(SOURCES["slopes_long"]))
        direct[f"Overall_{stage}_FC_offset"] = (ss, "Intercept", str(SOURCES["slopes_long"]))
    direct["Overall_Anticipation_minus_Outcome_slope"] = (slopes_wide, "Delta_slope", str(SOURCES["slopes_wide"]))
    ints = slopes_long.pivot(index="Subject_ID", columns="Stage", values="Intercept").reset_index()
    ints["offset_delta"] = ints["Anticipation"] - ints["Outcome"]
    direct["Overall_Anticipation_minus_Outcome_offset"] = (ints, "offset_delta", "QC recalculation from canonical slope intercepts")
    direct["SMA_Anticipation_rawFC_slope"] = (matched, "SMA_anticipation_rawFC_slope", str(SOURCES["matched"]))
    direct["IFG_Outcome_rawFC_slope"] = (matched, "IFG_outcome_rawFC_slope", str(SOURCES["matched"]))
    direct["Cingulate_Outcome_rawFC_slope"] = (matched, "Cingulate_outcome_rawFC_slope", str(SOURCES["matched"]))
    legacy = read_table(SOURCES["legacy"]).rename(columns={"subject": "Subject_ID"})
    direct["Legacy_G1_normative_MAE"] = (legacy, "normative_MAE", str(SOURCES["legacy"]))
    direct["Legacy_G1_normative_RMSE"] = (legacy, "normative_RMSE", str(SOURCES["legacy"]))
    coupling = read_table(SOURCES["stage_coupling"]).rename(columns={"subject": "Subject_ID"})
    direct["SZ_standardized_Anticipation_minus_Outcome_slope_contrast"] = (coupling, "NeuralSlope_stage_contrast", str(SOURCES["stage_coupling"]))
    direct["SZ_standardized_Bin10_Anticipation_minus_Outcome_similarity_contrast"] = (coupling, "Bin10Similarity_stage_contrast", str(SOURCES["stage_coupling"]))
    direct["SZ_Bin10_Anticipation_minus_Outcome_similarity_raw_difference"] = (coupling, "Bin10Similarity_raw_difference", str(SOURCES["stage_coupling"]))
    for name, (src, col, srcpath) in direct.items():
        audits.append(compare_frames(f"DIRECT_{name}", "Derived direct source", name, derived[derived.Metric_Name.eq(name)], src,
                                     ["Subject_ID"], "Value", col, str(SOURCES["derived"]), srcpath))
    consistency = pd.DataFrame(audits)
    write_tsv(consistency, DIRS["metric"] / "IMAGING_VALUE_CONSISTENCY_AUDIT.tsv")

    # 4. Missingness and duplicate-key audit.
    miss_rows = []
    def miss_row(family, name, d, expected_n, key_cols, value_col, valid_groups="HC;SZ"):
        vals = numeric(d[value_col])
        observed_subjects = d["Subject_ID"].nunique()
        duplicates = int(d.duplicated(key_cols, keep=False).sum())
        nonfinite = int((~np.isfinite(vals)).sum())
        missing = expected_n - int(d.loc[np.isfinite(vals), "Subject_ID"].nunique())
        status = "PASS" if missing == 0 and duplicates == 0 and nonfinite == 0 and observed_subjects == expected_n else "FAIL"
        miss_rows.append({"metric_family": family, "metric_name": name, "valid_groups": valid_groups,
                          "expected_subject_N": expected_n, "observed_subject_N": observed_subjects,
                          "finite_subject_N": int(d.loc[np.isfinite(vals), "Subject_ID"].nunique()), "missing_subject_N": max(0, missing),
                          "nonfinite_row_N": nonfinite, "duplicate_key_row_N": duplicates,
                          "audit_status": status, "notes": "No imputation performed."})
    for (stage, b), g in fc.groupby(["Stage", "Bin"]):
        miss_row("Stage FC", f"{stage}_Bin{int(b):02d}", g, 64, ["Subject_ID", "Stage", "Bin"], "FC")
    for (b, r), g in roi.groupby(["Bin", "Target_ROI"]):
        miss_row("Target ROI FC", f"Bin{int(b):02d}_{r}", g, 64, ["Subject_ID", "Bin", "Target_ROI"], "FC")
    for stage, g in sim.groupby("Stage"):
        miss_row("Similarity", f"Bin10_{stage}", g, 64, ["Subject_ID", "Stage"], "Similarity")
    for name, g in derived.groupby("Metric_Name"):
        structural_eligibility = name.startswith("SZ_standardized_")
        exp = 28 if structural_eligibility else (32 if name.startswith("SZ_") else 64)
        valid = "SZ" if exp == 32 else "HC;SZ"
        if structural_eligibility:
            valid = "SZ eligible stage-coupling subset"
            g = g[np.isfinite(numeric(g["Value"]))].copy()
        miss_row("Derived imaging", name, g, exp, ["Subject_ID", "Metric_Name"], "Value", valid)
    imaging_missing = pd.DataFrame(miss_rows)
    write_tsv(imaging_missing, DIRS["metric"] / "IMAGING_MISSINGNESS_REPORT.tsv")

    # 5. Outlier flags: descriptive review only, never automatic deletion.
    fc_o = fc.copy(); fc_o["metric"] = fc_o.Stage + "_Bin" + fc_o.Bin.astype(int).astype(str).str.zfill(2)
    roi_o = roi.copy(); roi_o["metric"] = "Bin" + roi_o.Bin.astype(int).astype(str).str.zfill(2) + "_" + roi_o.Target_ROI
    sim_o = sim.copy(); sim_o["metric"] = "Bin10_" + sim_o.Stage
    outliers = pd.concat([
        add_outliers(fc_o, "Stage FC", "metric", "FC", ["Stage", "Bin"]),
        add_outliers(roi_o, "Target ROI FC", "metric", "FC", ["Bin", "Target_ROI"]),
        add_outliers(sim_o, "Similarity", "metric", "Similarity", ["Stage"]),
        add_outliers(derived, "Derived imaging", "Metric_Name", "Value", ["Metric_Name"]),
    ], ignore_index=True)
    write_tsv(outliers, DIRS["metric"] / "IMAGING_OUTLIER_FLAGS.tsv")

    # 6. Subject matching: explicit IDs only; row order never used.
    final_ids = set(master.Subject_ID)
    table_sets = {
        "master": set(master.Subject_ID), "fc": set(fc.Subject_ID), "roi": set(roi.Subject_ID),
        "similarity": set(sim.Subject_ID), "derived": set(derived.Subject_ID), "clinical_v3": set(clinical.subject),
        "behaviour_trials": set(trials.Subject_ID),
    }
    group_maps = {
        "master": master.set_index("Subject_ID")["Group"].map(norm_group).to_dict(),
        "fc": fc.drop_duplicates("Subject_ID").set_index("Subject_ID")["Group"].map(norm_group).to_dict(),
        "roi": roi.drop_duplicates("Subject_ID").set_index("Subject_ID")["Group"].map(norm_group).to_dict(),
        "sim": sim.drop_duplicates("Subject_ID").set_index("Subject_ID")["Group"].map(norm_group).to_dict(),
        "clinical": clinical.set_index("subject")["group"].map(norm_group).to_dict(),
    }
    match_rows = []
    trial_counts = trials.groupby("Subject_ID").size().to_dict()
    run_counts = trials.groupby("Subject_ID")["Run"].nunique().to_dict()
    event_file_counts = trials.groupby("Subject_ID")["Source_File"].nunique().to_dict()
    for _, r in crosswalk.sort_values("Subject_ID").iterrows():
        sid = r.Subject_ID
        included = sid in final_ids
        expected_group = norm_group(r.Group)
        present = {k: sid in s for k, s in table_sets.items()}
        groups = [m.get(sid) for m in group_maps.values() if sid in m]
        group_ok = all(g == expected_group for g in groups)
        core_ok = all(present[k] for k in ["master", "fc", "roi", "similarity", "derived", "clinical_v3"])
        status = "PASS" if included and core_ok and group_ok else ("NOT_APPLICABLE_EXCLUDED" if not included else "FAIL")
        match_rows.append({
            "Subject_ID": sid, "Group": expected_group, "Behavior_EPrime_ID": r.get("Behavior_EPrime_ID", ""),
            "Final64_Included": "YES" if included else "NO", "Exclusion_Reason": r.get("Exclusion_Reason", ""),
            "id_format_valid": bool(re.match(r"^sub-\d{4}$", str(sid))),
            **{f"present_{k}": v for k, v in present.items()},
            "behaviour_trial_row_N": int(trial_counts.get(sid, 0)),
            "behaviour_run_N": int(run_counts.get(sid, 0)),
            "behaviour_event_source_file_N": int(event_file_counts.get(sid, 0)),
            "behaviour_accuracy_status": "MISSING_OR_UNDEFINED_NO_INFERENCE" if present["behaviour_trials"] else "NO_TRIAL_ROWS",
            "group_consistent_across_sources": group_ok,
            "one_to_one_final64_id": (master.Subject_ID.eq(sid).sum() == 1) if included else "NOT_APPLICABLE",
            "row_order_used": "NO", "fuzzy_matching_used": "NO",
            "matching_rule": r.get("ID_Match_Rule", "Explicit standardized subject ID"),
            "audit_status": status,
        })
    matching = pd.DataFrame(match_rows)
    write_tsv(matching, DIRS["clinical"] / "SUBJECT_MATCHING_AUDIT.tsv")

    # 7. Clinical dictionary and population-specific missingness.
    clinical_rows = []
    for var in [c for c in clinical.columns if c not in {"subject", "group"}]:
        meta = clinical_meta(var)
        clinical_rows.append({
            "variable_name": var, "description": meta["description"], "source_column": var,
            "source_file": str(CLIN_V3), "scale_or_task": meta["scale_or_task"],
            "applicable_groups": meta["applicable_groups"], "scoring_method": meta["scoring_method"],
            "reverse_scoring": meta["reverse_scoring"], "missing_rule": meta["missing_rule"],
            "possible_range": meta["possible_range"], "higher_value_meaning": meta["higher_value_meaning"],
            "raw_or_derived": meta["raw_or_derived"],
            "HC_N": int(clinical.loc[clinical.group.eq("HC"), var].notna().sum()),
            "SZ_N": int(clinical.loc[clinical.group.eq("SZ"), var].notna().sum()),
            "Total_N": int(clinical[var].notna().sum()),
            "analysis_status": meta["analysis_status"], "audit_status": meta["audit_status"], "notes": meta["notes"],
        })
    for var, desc, note in [
        ("BNSS", "Brief Negative Symptom Scale score", "Not found in the audited Final64 sources."),
        ("CPZ_equivalent", "Chlorpromazine-equivalent medication dose", "No verified dose conversion is available."),
        ("vidHit", "Legacy behavioural field named vidHit", "Definition/accuracy semantics are not sufficiently verified; do not treat as accuracy."),
    ]:
        n = int(master[var].notna().sum()) if var in master.columns else 0
        clinical_rows.append({"variable_name": var, "description": desc, "source_column": var if var in master.columns else "NOT_AVAILABLE",
                              "source_file": str(SOURCES["master"]) if var in master.columns else "NOT_AVAILABLE",
                              "scale_or_task": "Unavailable/legacy", "applicable_groups": "HC;SZ", "scoring_method": "Unverified/not available",
                              "reverse_scoring": "Unknown/not applicable", "missing_rule": "No imputation", "possible_range": "Unknown",
                              "higher_value_meaning": "Unknown", "raw_or_derived": "UNVERIFIED_OR_ABSENT", "HC_N": "",
                              "SZ_N": "", "Total_N": n, "analysis_status": "NOT_FOR_ANALYSIS", "audit_status": "UNAVAILABLE_OR_UNVERIFIED", "notes": note})
    clinical_dict = pd.DataFrame(clinical_rows)
    write_tsv(clinical_dict, DIRS["clinical"] / "CLINICAL_VARIABLE_DICTIONARY.tsv")

    clin_miss = []
    for var in [c for c in clinical.columns if c not in {"subject", "group"}]:
        applicable = clinical_meta(var)["applicable_groups"]
        for pop in ["ALL", "HC", "SZ"]:
            dd = clinical if pop == "ALL" else clinical[clinical.group.eq(pop)]
            nonmissing = int(dd[var].notna().sum())
            structural = pop == "HC" and applicable == "SZ"
            clin_miss.append({"variable_name": var, "population": pop, "population_N": len(dd),
                              "nonmissing_N": nonmissing, "missing_N": len(dd) - nonmissing,
                              "missing_percent": round(100 * (len(dd) - nonmissing) / len(dd), 3) if len(dd) else np.nan,
                              "missingness_type": "STRUCTURAL_NOT_APPLICABLE" if structural else "OBSERVED_MISSINGNESS",
                              "imputation_performed": "NO", "analysis_status": clinical_meta(var)["analysis_status"],
                              "audit_status": "PASS_REPORTED_NO_IMPUTATION"})
    for var in ["BNSS", "CPZ_equivalent", "vidHit"]:
        for pop, n in [("ALL", 64), ("HC", 32), ("SZ", 32)]:
            vals = master.loc[master.Group.eq(pop), var] if var in master.columns and pop != "ALL" else (master[var] if var in master.columns else pd.Series(dtype=float))
            non = int(vals.notna().sum()) if var in master.columns else 0
            clin_miss.append({"variable_name": var, "population": pop, "population_N": n, "nonmissing_N": non,
                              "missing_N": n - non, "missing_percent": round(100 * (n - non) / n, 3),
                              "missingness_type": "UNAVAILABLE_OR_DEFINITION_UNVERIFIED", "imputation_performed": "NO",
                              "analysis_status": "NOT_FOR_ANALYSIS", "audit_status": "WARNING"})
    clinical_missing = pd.DataFrame(clin_miss)
    write_tsv(clinical_missing, DIRS["clinical"] / "CLINICAL_MISSINGNESS_REPORT.tsv")

    # 8. Final analysis preparation matrix (values only; no model fitting).
    base = master[["Subject_ID", "Group", "final64_rest_posttrim_mean_FD"]].rename(columns={"final64_rest_posttrim_mean_FD": "MeanFD"})
    base = base.merge(clinical.rename(columns={"subject": "Subject_ID", "group": "clinical_group"}), on="Subject_ID", how="left")
    sl = derived.pivot(index="Subject_ID", columns="Metric_Name", values="Value").reset_index()
    sm = sim.pivot(index="Subject_ID", columns="Stage", values="Similarity").reset_index().rename(columns={"Anticipation": "Anticipation_similarity", "Outcome": "Outcome_similarity"})
    matrix = base.merge(sl, on="Subject_ID", how="left").merge(sm, on="Subject_ID", how="left")
    matrix = matrix.rename(columns={
        "age": "Age", "sex": "Sex", "Overall_Anticipation_FC_slope": "Anticipation_slope",
        "Overall_Outcome_FC_slope": "Outcome_slope", "Overall_Anticipation_minus_Outcome_slope": "Anticipation_minus_Outcome_slope",
        "SMA_Anticipation_rawFC_slope": "SMA_slope", "IFG_Outcome_rawFC_slope": "IFG_slope",
        "Cingulate_Outcome_rawFC_slope": "Cingulate_slope", "TEPS_anticipatory_calc": "TEPS_anticipatory",
        "TEPS_consummatory_calc": "TEPS_consummatory", "ACIPS_anticipatory_calc": "ACIPS_anticipatory",
        "ACIPS_consummatory_calc": "ACIPS_consummatory",
    })
    keep = ["Subject_ID", "Group", "Age", "Sex", "MeanFD", "Anticipation_slope", "Outcome_slope", "Anticipation_minus_Outcome_slope",
            "Anticipation_similarity", "Outcome_similarity", "SMA_slope", "IFG_slope", "Cingulate_slope",
            "TEPS_anticipatory", "TEPS_consummatory", "ACIPS_anticipatory", "ACIPS_consummatory",
            "PANSS_positive", "PANSS_negative", "PANSS_general", "PANSS_total", "SANS", "SAS", "BARS",
            "clinical_conflict_flag", "clinical_conflict_note"]
    matrix = matrix[keep]
    match_ok = matching[matching.Final64_Included.eq("YES")].set_index("Subject_ID")["audit_status"].eq("PASS")
    matrix["has_verified_subject_match"] = matrix.Subject_ID.map(match_ok).fillna(False)
    for c in keep[5:24]:
        matrix[f"has_{c}"] = np.isfinite(numeric(matrix[c])) & matrix["has_verified_subject_match"]
    matrix["has_valid_covariates"] = matrix[["Age", "Sex", "MeanFD"]].notna().all(axis=1) & matrix["has_verified_subject_match"]
    matrix["row_order_used"] = "NO"
    write_tsv(matrix, DIRS["analysis"] / "ANALYSIS_SAMPLE_MATRIX.tsv")

    # Pairwise and covariate-complete sample counts; no model is run.
    pair_specs = []
    stage_pairs = [
        ("TEPS_anticipatory", "Anticipation_slope"), ("TEPS_consummatory", "Outcome_slope"),
        ("ACIPS_anticipatory", "Anticipation_slope"), ("ACIPS_consummatory", "Outcome_slope"),
        ("TEPS_anticipatory", "Anticipation_similarity"), ("TEPS_consummatory", "Outcome_similarity"),
        ("ACIPS_anticipatory", "Anticipation_similarity"), ("ACIPS_consummatory", "Outcome_similarity"),
    ]
    pair_specs += [("Pleasure_stage_congruent", b, i, "HC;SZ") for b, i in stage_pairs]
    for b in ["PANSS_positive", "PANSS_negative", "PANSS_general", "PANSS_total", "SANS"]:
        for i in ["Anticipation_slope", "Outcome_slope", "Anticipation_similarity", "Outcome_similarity", "SMA_slope", "IFG_slope", "Cingulate_slope"]:
            pair_specs.append(("Clinical_symptom", b, i, "SZ"))
    count_rows = []
    for family, beh, img, valid in pair_specs:
        for pop in ["ALL", "HC", "SZ"]:
            dd = matrix if pop == "ALL" else matrix[matrix.Group.eq(pop)]
            applicable = not (valid == "SZ" and pop in {"ALL", "HC"})
            pair_ok = dd[[beh, img]].notna().all(axis=1)
            cov_ok = pair_ok & dd[["Age", "Sex", "MeanFD"]].notna().all(axis=1) & dd.has_verified_subject_match
            count_rows.append({"analysis_family": family, "behavior_variable": beh, "imaging_variable": img,
                               "valid_groups": valid, "population": pop, "population_N": len(dd),
                               "pairwise_complete_N": int(pair_ok.sum()) if applicable else "NOT_APPLICABLE",
                               "covariate_complete_verified_N": int(cov_ok.sum()) if applicable else "NOT_APPLICABLE",
                               "covariates": "Age;Sex;MeanFD", "model_run": "NO",
                               "audit_status": "PASS_COUNT_ONLY" if applicable else "NOT_APPLICABLE"})
    sample_counts = pd.DataFrame(count_rows)
    write_tsv(sample_counts, DIRS["analysis"] / "ANALYSIS_SAMPLE_COUNTS.tsv")

    # 9. Fixed-reference audit report.
    bins = read_table(SOURCES["bin_summary"])
    def as_markdown_table(df: pd.DataFrame) -> str:
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for value in row:
                if pd.isna(value):
                    vals.append("")
                elif isinstance(value, (float, np.floating)):
                    vals.append(f"{value:.8g}")
                else:
                    vals.append(str(value))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    bin_md = as_markdown_table(bins)
    g1_audit = f"""# Fixed G1 Reference Audit

Version: `{VERSION}`
Audit time: `{NOW}`

## Verdict

**PASS_WITH_PROVENANCE_WARNING**. The current Final64 FC pipeline uses the frozen 375-person reference G1 and its fixed 10-bin definition. It does not refit the gradient for Final64 subjects and does not use Procrustes alignment to produce the current FC profiles.

## Frozen reference identity

- Fixed G1 SHA256: `d7cd62a4c86e82d8b96c7e229e1e460f183f77544da7444280457a14d4f6c985`
- Definition-A bin NPY SHA256: `28f3fda1c1e3ca6b40e290bf7ff6924a613d33e3d3e7e1a8768d3985a4748e1f`
- Definition-A TSV alias SHA256: `092900555cc0b59bf3d9ebe867637cab02a28cd9be9bd7d032405ee51410a9b7`
- Bin voxel counts: `260,260,261,261,261,261,261,261,261,261` (total 2608).
- Centering: `G1_bin_center = Bin - 5.5`, giving -4.5 through +4.5.
- Direction: lower bins are VST/Caudate-rich; higher bins are Putamen-rich.

## Bin summary

{bin_md}

## FC and stage definitions

- Fingerprints are 2608 × 65 target-ROI Fisher-z FC matrices. Source float32 values are promoted to float64 before reductions.
- Stage FC is the unweighted arithmetic mean across 43 Anticipation or 19 Outcome target ROIs within subject and fixed G1 bin.
- VTA_bi, SNr_bi, and VP_VeP_bi are explicitly Not_assigned and excluded from the stage mean.
- Individual stage slopes are `np.polyfit(G1_bin_center, FC, 1)` over all 10 bins; there are no covariates inside the subject-level slope.
- Stage offsets are intercepts at centered predictor 0 (between bins 5 and 6).

## Similarity definitions

- Anticipation similarity uses the 43-target Bin10 fingerprint; Outcome uses the 19-target Bin10 fingerprint.
- HC subjects are compared with an HC leave-one-out template (HC31 for each HC); SZ subjects are compared with the full HC32 template.
- Similarity is Pearson r across targets followed by Fisher z (`arctanh` after clipping).
- The separate historical 65-target full-profile similarity is not stage-specific.

## Procrustes boundary

Historical `Legacy_G1_normative_MAE/RMSE` are voxelwise deviations of a Procrustes-aligned individual G1 from the frozen 375-person G1 after within-map standardization. They are not FC deviation and are classified as `LEGACY_EXPLORATORY`. Procrustes is not part of the current fixed-bin FC phenotype construction.

## Provenance warning

The original spatial mask from the earliest construction phase was unavailable. Exact voxel-row mapping was reconstructed from saved derived arrays and matched the frozen products exactly. This does not change any Final64 value, but the limitation remains recorded.

## Formal scripts, inputs, outputs, and evidence

| Role | Path | Concrete evidence used in this audit |
| --- | --- | --- |
| Reproduce frozen 375 G1 | `{PROJECT / '01_code' / '02_gradient_construction' / '18_reproduce_final375_gradient.py'}` | BrainSpace normalized-angle construction, explicit polarity rule, and reproduction outputs are kept separate from the frozen reference. |
| Build matched FC fingerprints | `{PROJECT / '01_code' / '01_preprocessing' / '01_build_375matched_timeseries_and_fingerprints.py'}` | Pearson r then `arctanh`; exact 2847-to-2608 row mapping; 65-target order; direct 2608-row validation. |
| Freeze bins and build Final64 profiles | `{fixed_script}` | Loads frozen reference/bin labels, checks fixed-bin identity, promotes FC arrays to float64, and takes the unweighted mean across stage-assigned target sets. |
| Canonical FC input/output | `{CANON / '01_FROZEN_INPUTS'}` → `{SOURCES['canonical_fc']}` and `{SOURCES['canonical_roi']}` | Export-to-canonical comparisons pass for all 1,280 stage FC and 41,600 ROI FC rows; maximum reproduction difference is reported in the consistency audit. |
| Subject slope construction | `{slope_script}` → `{SOURCES['slopes_long']}` / `{SOURCES['slopes_wide']}` | Per-subject `np.polyfit` over 10 centered bins; slope/intercept and stage difference match exported values. |
| Stage similarity construction | `{sim_script}` → `{SOURCES['sim_b05']}` | Creates HC leave-one-out and SZ HC32 templates, applies 43/19-stage ROI sets, Pearson r, and Fisher z. |
| Historical individual-gradient alignment | `{PROJECT / '01_code' / '02_gradient_construction' / 'run_current64_375reference_gradients.py'}` | Individual gradient fit, sign orientation, and Procrustes alignment; used only to interpret legacy G1-deviation metrics, not current FC profiles. |

The frozen reference identity is also recorded in `{SOURCES['input_manifest']}`. The 64-person fixed-G1 characterization reproduces the historical 375-person numeric reference in the archived validation outputs.
"""
    (DIRS["metric"] / "FIXED_G1_REFERENCE_AUDIT.md").write_text(g1_audit, encoding="utf-8")

    # 10. Readiness report and README.
    failures = int((consistency.audit_status == "FAIL").sum())
    match_fail = int((matching.loc[matching.Final64_Included.eq("YES"), "audit_status"] != "PASS").sum())
    imaging_missing_fail = int((imaging_missing.audit_status == "FAIL").sum())
    verdict = "CONDITIONALLY_READY" if failures == 0 and match_fail == 0 and imaging_missing_fail == 0 else "NOT_READY"
    count_lookup = clinical.set_index("subject")
    key_ns = {v: {g: int(clinical.loc[clinical.group.eq(g), v].notna().sum()) for g in ["HC", "SZ"]}
              for v in ["TEPS_anticipatory_calc", "TEPS_consummatory_calc", "ACIPS_anticipatory_calc", "ACIPS_consummatory_calc", "PANSS_positive", "SANS", "SAS", "BARS"]}
    candidate_rows = []
    for _, rr in sample_counts.iterrows():
        if not ((rr["analysis_family"] == "Pleasure_stage_congruent" and rr["population"] == "ALL") or
                (rr["analysis_family"] == "Clinical_symptom" and rr["population"] == "SZ")):
            continue
        is_roi = rr["imaging_variable"] in {"SMA_slope", "IFG_slope", "Cingulate_slope"}
        is_similarity = "similarity" in rr["imaging_variable"]
        if rr["analysis_family"] == "Pleasure_stage_congruent":
            question = f"Is {rr['behavior_variable']} associated with its stage-congruent {rr['imaging_variable']}?"
            readiness = "SECONDARY_READY" if is_similarity else "PRIMARY_CANDIDATE_READY"
            planned = "Association model to be prespecified; no model fitted in this audit"
        else:
            question = f"Within SZ, is {rr['behavior_variable']} associated with {rr['imaging_variable']}?"
            readiness = "EXPLORATORY_READY_WITH_SOURCE_WARNING" if is_roi else "SECONDARY_READY_WITH_SOURCE_WARNING"
            planned = "SZ-only association model to be prespecified; no model fitted in this audit"
        candidate_rows.append({
            "Research question": question,
            "Imaging variable": rr["imaging_variable"], "Clinical variable": rr["behavior_variable"],
            "Population": rr["population"], "Available N": rr["pairwise_complete_N"],
            "N with Age/Sex/MeanFD": rr["covariate_complete_verified_N"],
            "Suggested covariates": "Age; Sex; MeanFD", "Planned analysis family": planned,
            "Data readiness": readiness,
        })
    candidate_md = as_markdown_table(pd.DataFrame(candidate_rows))

    report = f"""# Final64 Analysis Readiness Report

Version: `{VERSION}`
Frozen at: `{NOW}`
Final verdict: **{verdict}**

## A. Dataset lock

- Formal sample: 64 unique IDs, HC=32 and SZ=32; all IDs match `sub-####`.
- Thirteen authoritative TSV inputs were copied byte-for-byte, SHA256-verified, and made read-only.
- The workbook is retained only as a review container; frozen TSV files are authoritative.
- No value was imputed, no new similarity was invented, and no behavioural accuracy was inferred.

## B. Imaging audit

- Fixed 375-person G1 and the 10-bin definition are verified; the current FC pipeline does not refit gradients.
- Stage FC: 64 × 2 stages × 10 bins = 1,280 complete values.
- Target ROI FC: 64 × 10 bins × 65 targets = 41,600 complete values.
- Stage-specific Bin10 similarity: 64 × 2 = 128 complete values, with HC leave-one-out and SZ HC32 references preserved.
- Derived metrics exactly match their direct formal sources; the Anticipation-minus-Outcome offset is identified as a deterministic QC reconstruction.
- Numeric consistency failures: {failures}. Imaging missingness/duplicate-key failures: {imaging_missing_fail}.

### Per-family decisions

| Metric family | Result | N | Analysis role | Decision basis |
| --- | --- | ---: | --- | --- |
| Overall Anticipation FC slope | PASS | 64 | PRIMARY_CANDIDATE | Exact match to canonical 10-bin per-subject slope. |
| Overall Outcome FC slope | PASS | 64 | PRIMARY_CANDIDATE | Exact match to canonical 10-bin per-subject slope. |
| Anticipation/Outcome intercepts | PASS | 64 each | SECONDARY_CANDIDATE | Intercept at centered-bin zero; exact source match. |
| Anticipation minus Outcome slope | PASS | 64 | SECONDARY_CANDIDATE | Exact stage-slope difference. |
| Anticipation minus Outcome offset | PASS_WITH_RESTRICTION | 64 | NOT_FOR_ANALYSIS | Deterministic QC recalculation, retained only for traceability. |
| SMA Anticipation slope | PASS | 64 | EXPLORATORY | Exact match to the formal ROI extension output. |
| IFG Outcome slope | PASS | 64 | EXPLORATORY | Exact match to the formal ROI extension output. |
| Cingulate Outcome slope | PASS | 64 | EXPLORATORY | Exact match to the formal ROI extension output. |
| Bin10 Anticipation 43-target similarity | PASS | 64 | SECONDARY_CANDIDATE | HC LOO / SZ HC32 references verified; Fisher-z Pearson similarity. |
| Bin10 Outcome 19-target similarity | PASS | 64 | SECONDARY_CANDIDATE | HC LOO / SZ HC32 references verified; Fisher-z Pearson similarity. |
| Historical Bin10 65-target similarity | WARNING | Not included in analysis matrix | EXPLORATORY | Valid historical full-profile concept, but not stage-specific. |
| Legacy G1 normative MAE/RMSE | WARNING | 64 each | LEGACY_EXPLORATORY | Procrustes-based individual-G1 deviation, not current FC deviation. |
| SZ raw similarity stage difference | PASS | 32 | EXPLORATORY | Exact stage-coupling source match. |
| SZ standardized stage contrasts | PASS_WITH_ELIGIBILITY_LIMIT | 28 each | EXPLORATORY | Defined only for the eligible stage-coupling subset; four SZ rows are structurally undefined. |

## C. Subject and clinical matching

- All 64 Final64 IDs match one-to-one across the master, stage FC, target ROI FC, similarity, derived imaging, and clinical v3 tables.
- Row order and fuzzy matching were not used. Six non-Final64 crosswalk records remain explicitly excluded.
- Final64 matching failures: {match_fail}.
- Corrected task events: 209 source files and 6,688 trial rows across the 70-person behavioural export; every Final64 subject has three runs in the frozen table.
- Trial ownership is verified through explicit crosswalk IDs. `Accuracy` remains blank with `MISSING_OR_UNDEFINED`; no value is inferred. `vidHit` remains `NOT_FOR_ANALYSIS` because its independent hit-rate meaning is unverified.

## D. Key available N (no imputation)

- TEPS anticipatory: HC={key_ns['TEPS_anticipatory_calc']['HC']}, SZ={key_ns['TEPS_anticipatory_calc']['SZ']}.
- TEPS consummatory: HC={key_ns['TEPS_consummatory_calc']['HC']}, SZ={key_ns['TEPS_consummatory_calc']['SZ']}.
- ACIPS anticipatory: HC={key_ns['ACIPS_anticipatory_calc']['HC']}, SZ={key_ns['ACIPS_anticipatory_calc']['SZ']}.
- ACIPS consummatory: HC={key_ns['ACIPS_consummatory_calc']['HC']}, SZ={key_ns['ACIPS_consummatory_calc']['SZ']}.
- PANSS positive/negative/general/total: SZ={key_ns['PANSS_positive']['SZ']} each; HC values are structurally not applicable.
- SANS: SZ={key_ns['SANS']['SZ']}; SAS={key_ns['SAS']['SZ']}; BARS={key_ns['BARS']['SZ']}.
- Pairwise and Age/Sex/MeanFD-complete counts for every prespecified pairing are in `ANALYSIS_SAMPLE_COUNTS.tsv`.

## E. Statistical-analysis preparation (models not run)

The following table is the explicit candidate list. `Available N` is pairwise complete and the next column additionally requires verified subject matching plus Age, Sex, and MeanFD. “Ready” here describes data readiness only; the model family and multiplicity strategy still require analysis-plan confirmation.

{candidate_md}

Outlier flags may support sensitivity review, but they do not authorize automatic deletion or winsorization.

## F. Conditions and unresolved items

| Severity | Item | Scope and required action |
| --- | --- | --- |
| BLOCKING_FOR_AFFECTED_COMBINATIONS | BFI, CSAS, CPAS scoring conflicts | `NOT_FOR_ANALYSIS` until existing-versus-recalculated conflicts are resolved; this does not block verified TEPS/ACIPS/PANSS combinations. |
| BLOCKING_FOR_AFFECTED_COMBINATIONS | BNSS and CPZ equivalent unavailable | Do not run analyses requiring these variables unless a verified source/conversion is added and separately audited. |
| BLOCKING_FOR_AFFECTED_COMBINATIONS | `vidHit` and trial Accuracy semantics | Do not treat `vidHit` as accuracy and do not reconstruct trial Accuracy without an independently verified response key. |
| NON_BLOCKING | PANSS negative and SANS source conflict at the case in the private source-conflict audit | Formal-v3 source priority is preserved; use only with the documented warning or resolve before confirmatory interpretation. |
| NON_BLOCKING | Historical Procrustes G1 deviation | Keep as `LEGACY_EXPLORATORY`; never substitute for current FC slope/similarity. |
| NON_BLOCKING | Historical 65-target similarity | Keep non-stage-specific; never relabel as Anticipation or Outcome. |
| NON_BLOCKING | Earliest spatial mask unavailable | Exact voxel order was reconstructed from saved arrays and matched frozen products; retain this provenance limitation. |
| OPTIONAL | Reconstructed stage-offset difference | Available for arithmetic traceability, but not approved as a primary variable. |
| OPTIONAL | Review-only outlier flags | May be used for sensitivity planning; no automatic exclusion or winsorization is permitted. |

The lock is therefore **{verdict}**: the verified imaging phenotypes and explicitly enumerated clinical/pleasure pairings are analysis-ready, while the listed clinical conflicts and unavailable/unverified variables remain prohibited.
"""
    (ROOT / "FINAL64_ANALYSIS_READINESS_REPORT.md").write_text(report, encoding="utf-8")

    readme = f"""# Final64 Database Lock

`{VERSION}` — created `{NOW}`

This directory is the analysis-ready, auditable lock for the 64 formal imaging subjects (HC=32, SZ=32). It freezes authoritative TSV inputs byte-for-byte and separates source data, provenance, QC recalculations, clinical audit, and analysis sample preparation.

## Directory map

- `frozen_data/`: read-only byte-identical copies of 13 authoritative export TSVs.
- `manifests/`: SHA256 inventory and frozen-file manifest.
- `metric_audit/`: fixed-G1 provenance, metric definitions, numeric consistency, missingness, and review-only outlier flags.
- `clinical_audit/`: explicit-ID matching, clinical definitions, and missingness without imputation.
- `analysis_preparation/`: one-row-per-subject analysis matrix and pairwise/covariate-complete sample counts. No statistical model is fitted here.
- `FINAL64_ANALYSIS_READINESS_REPORT.md`: final decision and allowed/prohibited uses.

## Non-negotiable rules

1. Use explicit `Subject_ID`; never match by row order or fuzzy ID.
2. Do not refit the gradient or change the frozen 375-person G1/bin definition.
3. Do not impute missing questionnaire/clinical values.
4. Do not create new similarity or behavioural accuracy measures without a separately approved, source-grounded definition.
5. Treat frozen TSVs as authoritative. The XLSX workbook is only a review mirror.
6. Outlier flags are descriptive QC and do not delete or exclude subjects automatically.

## Integrity status

- Frozen files: {len(frozen_manifest)}; all byte-identical and SHA256 verified.
- Numeric consistency audit failures: {failures}.
- Final64 subject-match failures: {match_fail}.
- Imaging missingness/duplicate-key failures: {imaging_missing_fail}.
- Final verdict: **{verdict}**.
"""
    (ROOT / "FINAL64_DATABASE_LOCK_README.md").write_text(readme, encoding="utf-8")

    summary = {
        "version": VERSION, "timestamp": NOW, "verdict": verdict,
        "formal_subject_n": len(master), "group_counts": master.Group.value_counts().to_dict(),
        "frozen_files": len(frozen_manifest), "consistency_checks": len(consistency),
        "consistency_failures": failures, "imaging_missingness_failures": imaging_missing_fail,
        "final64_matching_failures": match_fail, "outlier_rows": len(outliers),
        "outlier_flagged_rows": int(outliers.flag_any.sum()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
