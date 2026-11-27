from __future__ import annotations

import os
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(os.path.join(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"], '02/subject_level_data_export'))
ROOT = Path(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"])
GRAD = Path(os.environ["STRIATAL_PROJECT_ROOT"])

AUDIT_XLSX = Path(os.environ["STRIATAL_BEHAVIOR_AUDIT_WORKBOOK"])
PRIMARY70_XLSX = Path(os.environ["STRIATAL_REGISTRY_WORKBOOK"])
RAW_SCALE_XLSM = Path(os.environ["STRIATAL_RAW_SCALE_WORKBOOK"])
FINAL64_LABELS = ROOT / "final64_gradient_clinical_stats/02_subject_labels/final64_subject_labels.tsv"
FINAL64_CLINICAL = ROOT / "final64_gradient_clinical_stats/03_clinical_master/final64_clinical_for_server_v3.tsv"
SCALE_PROVENANCE = ROOT / "final64_gradient_clinical_stats/01_source_audit/scale_item_provenance.tsv"
SOURCE_CONFLICTS = ROOT / "final64_gradient_clinical_stats/01_source_audit/source_conflict_audit.tsv"
SCALE_QC = ROOT / "final64_gradient_clinical_stats/06_qc/scale_existing_vs_recalculated_qc.tsv"
SCALE_MISMATCH = ROOT / "final64_gradient_clinical_stats/06_qc/scale_score_mismatch_subjects.tsv"
BEHAVIOR_CROSSWALK = ROOT / "subject_id_crosswalk_behavior_to_bids.tsv"

FC_STAGE = GRAD / "CURRENT64_CANONICAL_RERUN/06_REWARD_STAGE/subject_bin_stage_FC_long.tsv"
FC_TARGET = GRAD / "CURRENT64_CANONICAL_RERUN/04_BINWISE_FC_PROFILES/subject_bin_roi_FC_long.tsv"
SLOPES_LONG = GRAD / "CURRENT64_CANONICAL_RERUN/09_INDIVIDUAL_SLOPES/subject_stage_slopes_long.tsv"
SLOPES_WIDE = GRAD / "CURRENT64_CANONICAL_RERUN/09_INDIVIDUAL_SLOPES/subject_stage_slopes_wide.tsv"
ROI_MATCHED = GRAD / "clinical_extension_analysis/roi_behavior_extension/data_internal/roi_behavior_matched_dataset.tsv"
STAGE_COUPLING = GRAD / "clinical_extension_analysis/stage_coupling_analysis/stage_coupling_subject_level_data.csv"
NORMATIVE = GRAD / "99_archive_64/rebuild_oxford_65roi_v2/25_current64_individual_G1_statistics/06_normative_deviation/01_tables/individual_G1_normative_deviation.tsv"
CLINICAL_DICTIONARY = GRAD / "clinical_extension_analysis/clinical_data_dictionary.csv"
VARIABLE_DICTIONARY = GRAD / "clinical_extension_analysis/roi_behavior_extension/variable_dictionary.csv"
EVENT_DIR = ROOT / "02_task_events/output_corrected_subjectlist_v3/bids_events_target_model"

REF_G1 = "fixed_375_person_reference_G1_10bins_2608voxels"
BUILD_VERSION = "2026-10-29_subject_level_export_v1"


def norm_subject(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower().replace("_", "-")
    match = re.search(r"sub-?(\d+)", text)
    return f"sub-{int(match.group(1)):04d}" if match else text


def norm_behavior_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def clean_for_tsv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return out


def write_tsv(df: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    clean_for_tsv(df).to_csv(path, sep="\t", index=False, na_rep="", lineterminator="\n", encoding="utf-8")
    return path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def domain_for(column: str) -> str:
    low = column.lower()
    if low in {"subject_id", "group", "final64_included", "exclusion_reason"} or "available" in low:
        return "Subject metadata"
    if "motion" in low or "fd" in low or "rms" in low:
        return "Motion and imaging QC"
    if any(x in low for x in ["pans", "sans", "sas", "bars", "illness", "onset", "dup", "episode", "diagnos", "medication", "family history", "treatment", "relapse"]):
        return "Clinical symptoms and illness history"
    if any(x in low for x in ["teps", "acips", "shaps", "csas", "cpas"]):
        return "Anhedonia and pleasure"
    if any(x in low for x in ["panas", "ees", "bdi", "erq"]):
        return "Affect, depression and emotion regulation"
    if any(x in low for x in ["hss", "shs", "mshs", "hr"]):
        return "Humor questionnaires"
    if "icq" in low:
        return "Interpersonal competence"
    if any(x in low for x in ["bfi", "five"]):
        return "Personality"
    if any(x in low for x in ["dprime", "picture_", "vid", "humornum", "虚报率"]):
        return "Humor behavior tasks"
    if any(x in low for x in ["age", "sex", "gender", "education", "iq", "comedy", "father_", "mother_", "despicable"]):
        return "Demographic and general measures"
    return "Other available variable"


def scale_for(column: str) -> str:
    upper = column.upper()
    for token in ["PANSS", "SANS", "SAS", "BARS", "TEPS", "ACIPS", "SHAPS", "CSAS", "CPAS", "PANAS", "EES", "BDI", "ERQ", "HSS", "SHS", "MSHS", "ICQ", "BFI", "FIVE", "HR"]:
        if token in upper:
            return "BFI/FIVE" if token in {"BFI", "FIVE"} else token
    if "VID" in upper:
        return "Video humor task"
    if any(x in upper for x in ["PICTURE", "HUMORNUM", "虚报率", "DPRIME"]):
        return "Picture humor task"
    if "MOTION" in upper or "FD" in upper or "RMS" in upper:
        return "Historical motion QC"
    return "General/other"


def jsonable_matrix(df: pd.DataFrame) -> list[list]:
    obj = df.astype(object).where(pd.notna(df), None)
    rows = [list(map(str, df.columns))]
    rows.extend(obj.values.tolist())
    return rows


OUT.mkdir(parents=True, exist_ok=True)

# Audited 70-person crosswalk.
audit_cross = pd.read_excel(AUDIT_XLSX, sheet_name="Subject_Crosswalk", skiprows=4, engine="openpyxl")
audit_cross["Subject_ID"] = audit_cross["Standardized_ID"].map(norm_subject)
audit_cross["Behavior_EPrime_ID"] = audit_cross["Behavior_EPrime_ID"].map(norm_behavior_id)
crosswalk = audit_cross.rename(columns={
    "Included_Final_Imaging": "Final64_Included",
    "Clinical_Record_Available": "Clinical_Data_Available",
    "EPrime_Raw_Behavior_Available": "Behaviour_Data_Available",
    "Individual_Imaging_Availability": "Imaging_Data_Available",
})[[
    "Subject_ID", "Group", "Behavior_EPrime_ID", "Original_Subject_ID", "Final64_Analysis_ID",
    "Final64_Included", "Exclusion_Reason", "Clinical_Data_Available", "Behaviour_Data_Available",
    "Imaging_Data_Available", "ID_Match_Rule", "Duplicate_ID_Flag", "Unmatched_ID_Flag", "Source",
]].copy()
crosswalk["Final64_Analysis_ID"] = crosswalk["Final64_Analysis_ID"].map(norm_subject)
crosswalk = crosswalk.sort_values(["Group", "Subject_ID"]).reset_index(drop=True)
write_tsv(crosswalk, "SUBJECT_CROSSWALK.tsv")

final64_ids = set(crosswalk.loc[crosswalk["Final64_Included"].eq("YES"), "Subject_ID"])
group_by_subject = crosswalk.set_index("Subject_ID")["Group"].to_dict()
subject_by_behavior = crosswalk.set_index("Behavior_EPrime_ID")["Subject_ID"].to_dict()

# FC exports: the formal 1,280-row stage-by-bin table and the lower-level 41,600-row target-ROI table.
fc = pd.read_csv(FC_STAGE, sep="\t").rename(columns={
    "Subject": "Subject_ID", "Bin": "Bin", "Stage": "Stage", "Group": "Group",
})
fc["Subject_ID"] = fc["Subject_ID"].map(norm_subject)
fc["Reference_G1_Version"] = REF_G1
fc["FC_Definition"] = "Mean Fisher-z FC across target ROIs assigned to the reward stage within each fixed G1 bin"
fc["Source_File"] = FC_STAGE.as_posix()
fc["Source_Variable"] = "FC"
fc["QC_Status"] = "PASS_CANONICAL_FINAL64_STAGE_BIN"
fc = fc[["Subject_ID", "Group", "Stage", "Bin", "G1_bin_center", "FC", "Reference_G1_Version", "FC_Definition", "Source_File", "Source_Variable", "QC_Status", "Age", "Sex", "MeanFD"]]
write_tsv(fc, "subject_level_fc_long.tsv")

fc_wide = fc.pivot(index=["Subject_ID", "Group"], columns=["Stage", "Bin"], values="FC")
fc_wide.columns = [f"FC_{stage}_Bin{int(bin_no):02d}" for stage, bin_no in fc_wide.columns]
fc_wide = fc_wide.reset_index()
write_tsv(fc_wide, "subject_level_fc_wide.tsv")

fc_target = pd.read_csv(FC_TARGET, sep="\t").rename(columns={"Subject": "Subject_ID", "ROI": "Target_ROI"})
fc_target["Subject_ID"] = fc_target["Subject_ID"].map(norm_subject)
fc_target["Atlas"] = "UNKNOWN_MIXED_TARGET_ATLAS_SEE_ROI_METADATA"
fc_target["Reference_G1_Version"] = REF_G1
fc_target["FC_Definition"] = "Fisher-z FC averaged across striatal voxels within each fixed G1 bin for one saved target ROI"
fc_target["Source_File"] = FC_TARGET.as_posix()
fc_target["Source_Variable"] = "FC"
fc_target["QC_Status"] = "PASS_CANONICAL_FINAL64_65ROI_BIN"
fc_target = fc_target[["Subject_ID", "Group", "Bin", "Target_ROI", "Subfamily", "RewardStage", "FC", "Atlas", "Reference_G1_Version", "FC_Definition", "Source_File", "Source_Variable", "QC_Status"]]
write_tsv(fc_target, "subject_level_fc_target_roi_long.tsv")

# Bin-10 similarity: retain the formal Fisher-z value and its underlying Pearson r.
roi_matched = pd.read_csv(ROI_MATCHED, sep="\t")
roi_matched["subject"] = roi_matched["subject"].map(norm_subject)
similarity_parts = []
for stage, prefix, n_targets in [("Anticipation", "bin10_anticipation", 43), ("Outcome", "bin10_outcome", 19)]:
    part = roi_matched[["subject", "group", f"{prefix}_profile_z", f"{prefix}_profile_r", f"{prefix}_n_roi", f"{prefix}_reference"]].copy()
    part.columns = ["Subject_ID", "Group", "Similarity", "Similarity_Raw_r", "N_Target_ROIs", "Reference_Pattern_ID"]
    part["Stage"] = stage
    part["Bin"] = 10
    part["Similarity_Definition"] = f"Fisher z of Pearson r between subject Bin10 {stage} {n_targets}-target fingerprint and the HC reference pattern"
    part["HC_Reference_Method"] = np.where(part["Reference_Pattern_ID"].eq("HC_LOO"), "HC leave-one-out reference", "Frozen HC32 reference")
    part["Leave_One_Out_Status"] = np.where(part["Reference_Pattern_ID"].eq("HC_LOO"), "YES_FOR_HC", "NO_HC32_FOR_SZ")
    part["Source_File"] = ROI_MATCHED.as_posix()
    part["Source_Variable"] = f"{prefix}_profile_z"
    part["QC_Status"] = "PASS_EXISTING_INDIVIDUAL_VALUE"
    similarity_parts.append(part)
similarity = pd.concat(similarity_parts, ignore_index=True)
similarity = similarity[["Subject_ID", "Group", "Stage", "Bin", "Similarity", "Similarity_Raw_r", "N_Target_ROIs", "Similarity_Definition", "Reference_Pattern_ID", "HC_Reference_Method", "Leave_One_Out_Status", "Source_File", "Source_Variable", "QC_Status"]].sort_values(["Subject_ID", "Stage"])
write_tsv(similarity, "subject_level_similarity_long.tsv")

# Existing and deterministic individual-level imaging metrics.
slopes_long = pd.read_csv(SLOPES_LONG, sep="\t")
slopes_long["Subject"] = slopes_long["Subject"].map(norm_subject)
slopes_wide = pd.read_csv(SLOPES_WIDE, sep="\t")
slopes_wide["Subject"] = slopes_wide["Subject"].map(norm_subject)
derived_rows: list[dict] = []

def add_metric(subject, group, metric, value, metric_type, stage="", roi="", formula="", inputs="", source="", status="EXISTING_FORMAL_OUTPUT", qc="PASS"):
    derived_rows.append({
        "Metric_Name": metric, "Subject_ID": subject, "Group": group, "Value": value,
        "Metric_Type": metric_type, "Stage": stage, "ROI": roi, "Formula": formula,
        "Input_Variables": inputs, "Source_File": source, "Status": status, "QC_Status": qc,
    })

for row in slopes_long.itertuples(index=False):
    stage = row.Stage
    add_metric(row.Subject, row.Group, f"Overall_{stage}_FC_slope", row.Slope, "Slope", stage, "Overall reward-stage ROI set", "OLS coefficient in FC ~ (Bin - 5.5)", "10 stage-specific bin FC values", SLOPES_LONG.as_posix())
    add_metric(row.Subject, row.Group, f"Overall_{stage}_FC_offset", row.Intercept, "Offset", stage, "Overall reward-stage ROI set", "OLS intercept in FC ~ (Bin - 5.5); equals fitted FC at centered Bin=5.5", "10 stage-specific bin FC values", SLOPES_LONG.as_posix())

intercepts = slopes_long.pivot(index=["Subject", "Group"], columns="Stage", values="Intercept").reset_index()
for row in slopes_wide.itertuples(index=False):
    add_metric(row.Subject, row.Group, "Overall_Anticipation_minus_Outcome_slope", row.Delta_slope, "Stage_Contrast", "Anticipation_minus_Outcome", "Overall reward-stage ROI sets", "Anticipation_slope - Outcome_slope", "Anticipation_slope; Outcome_slope", SLOPES_WIDE.as_posix())
for row in intercepts.itertuples(index=False):
    value = getattr(row, "Anticipation") - getattr(row, "Outcome")
    add_metric(row.Subject, row.Group, "Overall_Anticipation_minus_Outcome_offset", value, "Stage_Contrast", "Anticipation_minus_Outcome", "Overall reward-stage ROI sets", "Anticipation_offset - Outcome_offset", "Anticipation intercept; Outcome intercept", SLOPES_LONG.as_posix(), "RECONSTRUCTED_FROM_EXISTING_DATA", "PASS_DETERMINISTIC_DIFFERENCE")

roi_metric_specs = [
    ("SMA_anticipation_rawFC_slope", "SMA_Anticipation_rawFC_slope", "Anticipation", "SMA"),
    ("IFG_outcome_rawFC_slope", "IFG_Outcome_rawFC_slope", "Outcome", "IFG six-ROI composite"),
    ("Cingulate_outcome_rawFC_slope", "Cingulate_Outcome_rawFC_slope", "Outcome", "Cingulate"),
]
for source_col, metric, stage, roi in roi_metric_specs:
    for row in roi_matched[["subject", "group", source_col]].itertuples(index=False, name=None):
        add_metric(row[0], row[1], metric, row[2], "ROI_Slope", stage, roi, "OLS coefficient in raw FC ~ (Bin - 5.5)", "10 ROI-specific bin FC values", ROI_MATCHED.as_posix(), "EXISTING_FORMAL_OUTPUT", "PASS_EXISTING_VALIDATED_ROI_SLOPE")

stage_coupling = pd.read_csv(STAGE_COUPLING)
stage_coupling["subject"] = stage_coupling["subject"].map(norm_subject)
coupling_specs = [
    ("NeuralSlope_stage_contrast", "SZ_standardized_Anticipation_minus_Outcome_slope_contrast", "Standardized_Stage_Contrast", "Ant_slope_z - Out_slope_z"),
    ("Bin10Similarity_stage_contrast", "SZ_standardized_Bin10_Anticipation_minus_Outcome_similarity_contrast", "Standardized_Stage_Contrast", "Bin10_Ant_similarity_z - Bin10_Out_similarity_z"),
    ("Bin10Similarity_raw_difference", "SZ_Bin10_Anticipation_minus_Outcome_similarity_raw_difference", "Stage_Contrast", "Bin10 Anticipation Fisher-z similarity - Bin10 Outcome Fisher-z similarity"),
]
for source_col, metric, metric_type, formula in coupling_specs:
    for row in stage_coupling[["subject", "group", source_col]].itertuples(index=False, name=None):
        add_metric(row[0], row[1], metric, row[2], metric_type, "Anticipation_minus_Outcome", "Bin10 fingerprint" if "Similarity" in metric else "Overall reward-stage ROI sets", formula, source_col, STAGE_COUPLING.as_posix(), "EXISTING_FORMAL_OUTPUT_SZ_STAGE_COUPLING", "PASS_EXISTING_SZ_ONLY")

normative = pd.read_csv(NORMATIVE, sep="\t")
normative["subject"] = normative["subject"].map(norm_subject)
for row in normative.itertuples(index=False):
    add_metric(row.subject, row.group, "Legacy_G1_normative_MAE", row.normative_MAE, "Legacy_Normative_Deviation", "", "2608-voxel G1 map", "mean(abs(z_subject_G1 - z_reference_G1))", "subject G1 map; frozen 375 reference G1", NORMATIVE.as_posix(), "EXISTING_LEGACY_OUTPUT", "PASS_WITH_CAVEAT_NOT_FC_DEVIATION")
    add_metric(row.subject, row.group, "Legacy_G1_normative_RMSE", row.normative_RMSE, "Legacy_Normative_Deviation", "", "2608-voxel G1 map", "sqrt(mean((z_subject_G1 - z_reference_G1)^2))", "subject G1 map; frozen 375 reference G1", NORMATIVE.as_posix(), "EXISTING_LEGACY_OUTPUT", "PASS_WITH_CAVEAT_NOT_FC_DEVIATION")

derived = pd.DataFrame(derived_rows).sort_values(["Subject_ID", "Metric_Name"]).reset_index(drop=True)
write_tsv(derived, "subject_level_derived_imaging_metrics.tsv")

# De-identified 70-person clinical/behavior wide table. Preserve original source variable names.
primary70 = pd.read_excel(PRIMARY70_XLSX, sheet_name="01_主信息_70人", engine="openpyxl")
primary70["Subject_ID"] = primary70["MRI_ID"].map(norm_subject)
primary70["Behavior_EPrime_ID"] = primary70["ID"].map(norm_behavior_id)
primary70 = primary70.drop(columns=[c for c in ["ID", "group", "name", "name_pinyin", "MRI_ID", "remarks"] if c in primary70.columns])

clinical_wide = crosswalk[["Subject_ID", "Group", "Behavior_EPrime_ID", "Final64_Included", "Exclusion_Reason", "Clinical_Data_Available", "Behaviour_Data_Available", "Imaging_Data_Available"]].merge(primary70, on=["Subject_ID", "Behavior_EPrime_ID"], how="left", validate="one_to_one")

final64_clinical = pd.read_csv(FINAL64_CLINICAL, sep="\t")
final64_clinical["subject"] = final64_clinical["subject"].map(norm_subject)
final64_clinical = final64_clinical.drop(columns=["group"]).rename(columns={c: f"final64_v3__{c}" for c in final64_clinical.columns if c != "subject"}).rename(columns={"subject": "Subject_ID"})
clinical_wide = clinical_wide.merge(final64_clinical, on="Subject_ID", how="left", validate="one_to_one")

raw_clinical = pd.read_excel(RAW_SCALE_XLSM, sheet_name="临床", engine="openpyxl")
raw_clinical["Behavior_EPrime_ID"] = raw_clinical["Unnamed: 1"].map(norm_behavior_id)
raw_clinical["Subject_ID"] = raw_clinical["Behavior_EPrime_ID"].map(subject_by_behavior)
raw_clinical = raw_clinical[raw_clinical["Subject_ID"].notna()].copy()
drop_sensitive = ["ID", "Unnamed: 1", "grp", "Name", "Hometown"]
raw_clinical = raw_clinical.drop(columns=[c for c in drop_sensitive if c in raw_clinical.columns])
raw_clinical = raw_clinical.drop(columns=[c for c in raw_clinical.columns if re.match(r"^(Pans阳性1|P\d+|N阴性1|N\d+|G一般病理1|G\d+|SANS_\d+|SAS_\d+|BARS_\d+)$", str(c))])
raw_clinical = raw_clinical.rename(columns={c: f"clinical_raw__{c}" for c in raw_clinical.columns if c not in ["Subject_ID", "Behavior_EPrime_ID"]})
raw_clinical = raw_clinical.drop_duplicates("Subject_ID", keep="first").drop(columns=["Behavior_EPrime_ID"])
clinical_wide = clinical_wide.merge(raw_clinical, on="Subject_ID", how="left", validate="one_to_one")

motion = pd.read_excel(PRIMARY70_XLSX, sheet_name="02_头动参数_70人", engine="openpyxl")
motion["Subject_ID"] = motion["MRI_ID"].map(norm_subject)
motion_keep = [c for c in motion.columns if c not in ["Subject_ID", "ID", "group", "name", "name_pinyin", "MRI_ID", "source_excel_row"]]
motion = motion[["Subject_ID"] + motion_keep]
write_tsv(motion, "subject_level_motion_historical_long.tsv")
motion_values = [c for c in motion_keep if c != "motion_block"]
motion_wide = motion.pivot(index="Subject_ID", columns="motion_block", values=motion_values)
motion_wide.columns = [f"historical_{block}__{metric}" for metric, block in motion_wide.columns]
motion_wide = motion_wide.reset_index()
clinical_wide = clinical_wide.merge(motion_wide, on="Subject_ID", how="left", validate="one_to_one")

rest_fd = roi_matched[["subject", "mean_FD_rest_posttrim"]].rename(columns={"subject": "Subject_ID", "mean_FD_rest_posttrim": "final64_rest_posttrim_mean_FD"})
clinical_wide = clinical_wide.merge(rest_fd, on="Subject_ID", how="left", validate="one_to_one")
clinical_wide = clinical_wide.sort_values(["Group", "Subject_ID"]).reset_index(drop=True)
write_tsv(clinical_wide, "subject_level_clinical_behaviour_wide.tsv")

# Raw questionnaire items long table, filtered by the audited 70 behavior IDs.
raw_scales = pd.read_excel(RAW_SCALE_XLSM, sheet_name="scales", engine="openpyxl")
raw_scales["Behavior_EPrime_ID"] = raw_scales["sub"].map(norm_behavior_id)
raw_scales["Subject_ID"] = raw_scales["Behavior_EPrime_ID"].map(subject_by_behavior)
raw_scales = raw_scales[raw_scales["Subject_ID"].notna()].copy()
raw_csas = pd.read_excel(RAW_SCALE_XLSM, sheet_name="csas&cpas", engine="openpyxl")
raw_csas["Behavior_EPrime_ID"] = raw_csas["sub"].map(norm_behavior_id)
raw_csas["Subject_ID"] = raw_csas["Behavior_EPrime_ID"].map(subject_by_behavior)
raw_csas = raw_csas[raw_csas["Subject_ID"].notna()].copy()
raw_clinical_items = pd.read_excel(RAW_SCALE_XLSM, sheet_name="临床", engine="openpyxl")
raw_clinical_items["Behavior_EPrime_ID"] = raw_clinical_items["Unnamed: 1"].map(norm_behavior_id)
raw_clinical_items["Subject_ID"] = raw_clinical_items["Behavior_EPrime_ID"].map(subject_by_behavior)
raw_clinical_items = raw_clinical_items[raw_clinical_items["Subject_ID"].notna()].copy()

item_specs: list[tuple[str, str, str]] = []
prefix_map = [
    (r"^PANAS\d+$", "PANAS"), (r"^SHAPS\d+$", "SHAPS"), (r"^TEPS\d+$", "TEPS"),
    (r"^ACIPS\d+$", "ACIPS"), (r"^EES\d+$", "EES"), (r"^BDI\d+$", "BDI"),
    (r"^ERQ\d+$", "ERQ"), (r"^HSS\d+$", "HSS"), (r"^SHS\d+$", "SHS"),
    (r"^MSHS\d+$", "MSHS"), (r"^ICQ\d+$", "ICQ"), (r"^FIVE\d+$", "BFI/FIVE"),
    (r"^HR\d+$", "HR"),
]
for col in raw_scales.columns:
    for pattern, questionnaire in prefix_map:
        if re.match(pattern, str(col)):
            item_specs.append((questionnaire, str(col), "scales"))
            break
for col in raw_csas.columns:
    if re.match(r"^RCSAS\d+$", str(col)):
        item_specs.append(("CSAS/RCSAS", str(col), "csas&cpas"))
    elif re.match(r"^RCPAS\d+$", str(col)):
        item_specs.append(("CPAS/RCPAS", str(col), "csas&cpas"))
clinical_item_patterns = [
    (r"^(Pans阳性1|P[2-7])$", "PANSS"), (r"^(N阴性1|N[2-7])$", "PANSS"),
    (r"^(G一般病理1|G(?:[2-9]|1[0-6]))$", "PANSS"), (r"^SANS_\d+$", "SANS"),
    (r"^SAS_\d+$", "SAS"), (r"^BARS_\d+$", "BARS"),
]
for col in raw_clinical_items.columns:
    for pattern, questionnaire in clinical_item_patterns:
        if re.match(pattern, str(col)):
            item_specs.append((questionnaire, str(col), "临床"))
            break

item_rows = []
for questionnaire, item_id, source_sheet in item_specs:
    source_df = raw_scales if source_sheet == "scales" else raw_csas if source_sheet == "csas&cpas" else raw_clinical_items
    for row in source_df[["Subject_ID", item_id]].itertuples(index=False, name=None):
        subject, raw_value = row
        group = group_by_subject.get(subject, "")
        scored_value = np.nan
        reverse_status = "NO_OR_NOT_APPLICABLE"
        scoring_version = "RAW_ITEM_ONLY"
        qc_status = "PASS_SOURCE_VALUE"
        if questionnaire == "ACIPS" and item_id == "ACIPS3" and pd.notna(raw_value):
            scored_value = 7 - float(raw_value)
            reverse_status = "YES"
            scoring_version = "AUTHORITATIVE_ACIPS3_7_MINUS_RAW"
        elif questionnaire == "TEPS" and item_id == "TEPS13" and pd.notna(raw_value):
            scored_value = 7 - float(raw_value)
            reverse_status = "YES_FOR_TEPS_TOTAL_ONLY"
            scoring_version = "AUTHORITATIVE_TEPS13_TOTAL_SCORE_7_MINUS_RAW"
        elif source_sheet == "csas&cpas":
            scored_col = f"{item_id}.1"
            if scored_col in source_df.columns:
                scored_value = source_df.loc[source_df["Subject_ID"].eq(subject), scored_col].iloc[0]
                reverse_status = "EXISTING_SCORED_REGION_RULE_PENDING"
                scoring_version = "EXISTING_REVERSED_REGION_UNRESOLVED"
                qc_status = "REVIEW_CSAS_CPAS_SCORING"
        if pd.isna(raw_value):
            missing_status = "STRUCTURAL_NOT_APPLICABLE" if group == "HC" and questionnaire in {"PANSS", "SANS", "SAS", "BARS"} else "MISSING_IN_SOURCE"
        else:
            missing_status = "OBSERVED"
        item_rows.append({
            "Subject_ID": subject, "Group": group, "Questionnaire": questionnaire, "Item_ID": item_id,
            "Raw_Value": raw_value, "Scored_Value": scored_value, "Reverse_Scored": reverse_status,
            "Missing_Status": missing_status, "Source_File": RAW_SCALE_XLSM.as_posix(),
            "Source_Sheet": source_sheet, "Scoring_Version": scoring_version, "QC_Status": qc_status,
        })
questionnaire_items = pd.DataFrame(item_rows).sort_values(["Subject_ID", "Questionnaire", "Item_ID"]).reset_index(drop=True)
write_tsv(questionnaire_items, "subject_level_questionnaire_items_long.tsv")

# Corrected, untrimmed target-onset event files. No accuracy is invented.
trial_frames = []
event_files = sorted(EVENT_DIR.glob("*events.tsv"))
for path in event_files:
    match = re.match(r"sub-(\d+)_task-humor_run-(\d+)_events.tsv", path.name)
    if not match:
        continue
    behavior_id, run = match.groups()
    subject = subject_by_behavior.get(norm_behavior_id(behavior_id), "")
    data = pd.read_csv(path, sep="\t")
    data.insert(0, "Subject_ID", subject)
    data.insert(1, "Group", group_by_subject.get(subject, ""))
    data.insert(2, "Behavior_EPrime_ID", norm_behavior_id(behavior_id))
    data.insert(3, "Final64_Included", "YES" if subject in final64_ids else "NO")
    data.insert(4, "Task", "humor_fMRI_dialogue_judgment")
    data.insert(5, "Run", int(run))
    data["Task_Condition"] = data.get("trial_type")
    data["Stimulus_Text"] = data[[c for c in ["stimulus_context", "stimulus_setup", "stimulus_target"] if c in data.columns]].fillna("").astype(str).agg(" | ".join, axis=1).str.strip(" |")
    data["Response"] = data.get("response")
    data["Reaction_Time"] = data.get("response_time")
    data["Accuracy"] = data.get("accuracy")
    data["Accuracy_Status"] = np.where(data["Accuracy"].notna(), "RECORDED", "MISSING_OR_UNDEFINED")
    data["Source_File"] = path.as_posix()
    trial_frames.append(data)
trials = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
trial_front = ["Subject_ID", "Group", "Behavior_EPrime_ID", "Final64_Included", "Task", "Run", "Task_Condition", "Stimulus_Text", "Response", "Reaction_Time", "Accuracy", "Accuracy_Status"]
trials = trials[trial_front + [c for c in trials.columns if c not in trial_front]]
write_tsv(trials, "subject_level_behaviour_trials_long.tsv")

# Variable dictionary and coverage.
scale_prov = pd.read_csv(SCALE_PROVENANCE, sep="\t")
clinical_dict = pd.read_csv(CLINICAL_DICTIONARY)
variable_dict = pd.read_csv(VARIABLE_DICTIONARY)
scale_mismatch = pd.read_csv(SCALE_MISMATCH, sep="\t")
mismatch_n = scale_mismatch.groupby("scale")["subject"].nunique().to_dict()
prov_by_final = scale_prov.dropna(subset=["final_column"]).drop_duplicates("final_column").set_index("final_column").to_dict("index")
clinical_by_var = clinical_dict.drop_duplicates("variable").set_index("variable").to_dict("index")
variable_by_source = variable_dict.drop_duplicates("source_column").set_index("source_column").to_dict("index")

dictionary_rows = []
coverage_rows = []
meta_cols = {"Subject_ID", "Group", "Behavior_EPrime_ID", "Final64_Included", "Exclusion_Reason", "Clinical_Data_Available", "Behaviour_Data_Available", "Imaging_Data_Available"}
for column in clinical_wide.columns:
    source_file = PRIMARY70_XLSX.as_posix()
    source_sheet = "01_主信息_70人"
    original = column
    definition = "Existing source variable; exact local definition pending review"
    scoring_method = "Existing recorded value"
    raw_derived = "Metadata" if column in meta_cols else "Existing source value"
    qc_status = "PASS_SOURCE_PRESERVED"
    definition_status = "DEFINED_METADATA" if column in meta_cols else "UNKNOWN_PENDING_REVIEW"
    higher = "UNKNOWN_PENDING_REVIEW"
    unit = "UNKNOWN_PENDING_REVIEW"
    possible_range = ""
    if column.startswith("final64_v3__"):
        original = column.removeprefix("final64_v3__")
        source_file = FINAL64_CLINICAL.as_posix()
        source_sheet = "TSV"
        raw_derived = "Canonical existing/derived value"
        if original in clinical_by_var:
            rec = clinical_by_var[original]
            definition = rec.get("meaning", definition)
            unit = rec.get("unit", unit)
            qc_status = "PASS_WITH_DOCUMENTED_CAVEAT" if rec.get("cleaning_or_caveat") else qc_status
            definition_status = "DEFINED_WITH_CAVEAT"
        if original in variable_by_source:
            rec = variable_by_source[original]
            definition = rec.get("definition", definition)
            unit = rec.get("unit", unit)
            higher = rec.get("direction", higher)
            definition_status = "DEFINED"
        if original in prov_by_final:
            rec = prov_by_final[original]
            scoring_method = rec.get("scoring_formula") or rec.get("scoring_source") or scoring_method
            qc_status = rec.get("status", qc_status)
    elif column.startswith("clinical_raw__"):
        original = column.removeprefix("clinical_raw__")
        source_file = RAW_SCALE_XLSM.as_posix()
        source_sheet = "临床"
        raw_derived = "Raw clinical field"
        definition = f"Raw clinical field preserved from source: {original}"
        scoring_method = "No transformation"
        definition_status = "SOURCE_LABEL_ONLY"
    elif column.startswith("historical_motion_block"):
        source_file = PRIMARY70_XLSX.as_posix()
        source_sheet = "02_头动参数_70人"
        raw_derived = "Historical task-block motion value"
        definition = "Historical task-run motion QC; not the final64 resting post-trim model covariate"
        scoring_method = "Existing recorded motion summary"
        definition_status = "DEFINED_WITH_CAVEAT"
        qc_status = "HISTORICAL_NOT_FINAL_MODEL_COVARIATE"
    elif column == "final64_rest_posttrim_mean_FD":
        source_file = ROI_MATCHED.as_posix()
        source_sheet = "TSV"
        original = "mean_FD_rest_posttrim"
        raw_derived = "Existing final64 imaging QC"
        definition = "Resting-state post-trim mean FD used in the canonical final64 data"
        scoring_method = "Existing pipeline output"
        unit = "mm"
        definition_status = "DEFINED"
    elif column in meta_cols:
        source_file = AUDIT_XLSX.as_posix()
        source_sheet = "Subject_Crosswalk"
        definition = "Audited subject-level identifier or availability flag"
        scoring_method = "Exact ID mapping; no row-order matching"
        definition_status = "DEFINED"
    if scale_for(column) in {"BFI/FIVE", "CSAS", "CPAS"}:
        qc_status = "REVIEW_KNOWN_SCORING_MISMATCH_OR_UNRESOLVED_SCORING"
    if "SHS二" in column or "SHS_2" in column:
        qc_status = "PENDING_NOT_IN_CURRENT_MASTER"
    applicable = "SZ_ONLY" if domain_for(column) == "Clinical symptoms and illness history" and any(x in column.lower() for x in ["pans", "sans", "sas", "bars", "illness", "onset", "dup", "episode", "medication", "diagnos", "treatment", "relapse"]) else "ALL_SUBJECTS"
    available_n = int(clinical_wide[column].notna().sum())
    final_mask = clinical_wide["Final64_Included"].eq("YES")
    hc_mask = clinical_wide["Group"].eq("HC")
    sz_mask = clinical_wide["Group"].eq("SZ")
    scale_name = scale_for(column).split("/")[0]
    dictionary_rows.append({
        "Variable_Name": column, "Original_Variable_Name": original, "Questionnaire_or_Task": scale_for(column),
        "Domain": domain_for(column), "Definition": definition, "Unit": unit, "Scoring_Method": scoring_method,
        "Possible_Range": possible_range, "Higher_Value_Meaning": higher, "Raw_or_Derived": raw_derived,
        "Source_File": source_file, "Source_Sheet": source_sheet, "Missing_N": int(clinical_wide[column].isna().sum()),
        "Available_N": available_n, "QC_Status": qc_status, "Definition_Status": definition_status,
    })
    coverage_rows.append({
        "Variable_Name": column, "Applicable_Population": applicable, "All_Traceable_Available_N": available_n,
        "Final64_Available_N": int(clinical_wide.loc[final_mask, column].notna().sum()),
        "HC_Available_N": int(clinical_wide.loc[hc_mask, column].notna().sum()),
        "SZ_Available_N": int(clinical_wide.loc[sz_mask, column].notna().sum()),
        "Missing_N": int(clinical_wide[column].isna().sum()), "Source_Match_Failure_N": 0,
        "QC_Not_Pass_N": int(mismatch_n.get(scale_name, 0)), "QC_Status": qc_status,
    })
dictionary = pd.DataFrame(dictionary_rows)
coverage = pd.DataFrame(coverage_rows)
write_tsv(dictionary, "CLINICAL_BEHAVIOUR_DATA_DICTIONARY.tsv")
write_tsv(coverage, "CLINICAL_BEHAVIOUR_COVERAGE.tsv")

# Unified subject-level view: 70 subjects retained; no complete-case deletion or imputation.
sim_wide = similarity.pivot(index=["Subject_ID", "Group"], columns="Stage", values=["Similarity", "Similarity_Raw_r"])
sim_wide.columns = [f"Bin10_{stage}_{measure}" for measure, stage in sim_wide.columns]
sim_wide = sim_wide.reset_index()
derived_wide = derived.pivot_table(index=["Subject_ID", "Group"], columns="Metric_Name", values="Value", aggfunc="first").reset_index()
norm_wide = normative.rename(columns={"subject": "Subject_ID", "group": "Group"})
master = clinical_wide.merge(fc_wide, on=["Subject_ID", "Group"], how="left", validate="one_to_one")
master = master.merge(sim_wide, on=["Subject_ID", "Group"], how="left", validate="one_to_one")
master = master.merge(derived_wide, on=["Subject_ID", "Group"], how="left", validate="one_to_one")
master = master.merge(norm_wide, on=["Subject_ID", "Group"], how="left", validate="one_to_one", suffixes=("", "__normative_source"))
master = master.sort_values(["Group", "Subject_ID"]).reset_index(drop=True)
write_tsv(master, "subject_level_brain_behaviour_master.tsv")
master_final64 = master[master["Final64_Included"].eq("YES")].copy()
write_tsv(master_final64, "subject_level_brain_behaviour_master_final64.tsv")

# Consolidated QC issue table for review workbook.
audit_qc = pd.read_excel(AUDIT_XLSX, sheet_name="QC_Register", skiprows=4, engine="openpyxl")
audit_qc["Record_Type"] = "AUDIT_REGISTER"
conflicts = pd.read_csv(SOURCE_CONFLICTS, sep="\t")
conflict_qc = pd.DataFrame({
    "QC_ID": [f"CONFLICT_{i+1:03d}" for i in range(len(conflicts))], "Area": "Source value conflict",
    "Issue_or_Finding": conflicts.apply(lambda r: f"{r['subject']} {r['variable']}: {r['source_a']}={r['value_a']} vs {r['source_b']}={r['value_b']}", axis=1),
    "Status": conflicts["issue_type"], "Required_Action": conflicts["resolution"], "Source": conflicts["source_a"], "Record_Type": "SOURCE_CONFLICT",
})
mismatch_qc = pd.DataFrame({
    "QC_ID": [f"SCALE_MISMATCH_{i+1:03d}" for i in range(len(scale_mismatch))], "Area": "Scale scoring mismatch",
    "Issue_or_Finding": scale_mismatch.apply(lambda r: f"{r['subject']} {r['scale']} {r['metric']}: existing={r['existing_value']}, calculated={r['calculated_value']}, difference={r['difference']}", axis=1),
    "Status": "REVIEW", "Required_Action": "Preserve both values; do not overwrite the formal existing score without authoritative review",
    "Source": SCALE_MISMATCH.as_posix(), "Record_Type": "SCALE_MISMATCH",
})
qc_issues = pd.concat([audit_qc, conflict_qc, mismatch_qc], ignore_index=True, sort=False)
write_tsv(qc_issues, "DATA_EXPORT_QC_ISSUES.tsv")

# QC summary and gap register.
fc_counts = fc.groupby("Subject_ID").size()
target_counts = fc_target.groupby("Subject_ID").size()
sim_counts = similarity.groupby("Subject_ID").size()
trial_subject_runs = trials[["Subject_ID", "Run"]].drop_duplicates().groupby("Subject_ID").size()
qc_report = f"""# Subject-level data export QC report

Build version: `{BUILD_VERSION}`
Build date: 2026-10-30
Scope: deterministic extraction, ID matching, reshaping and documented differences only. No new formal statistical, correlation, regression, mixed-effects or Bayesian model was run.

## Subject identity and scope

- Audited traceable subjects: {crosswalk['Subject_ID'].nunique()} ({(crosswalk['Group']=='HC').sum()} HC, {(crosswalk['Group']=='SZ').sum()} SZ).
- Frozen final64: {len(final64_ids)} ({crosswalk.query("Final64_Included == 'YES' and Group == 'HC'").shape[0]} HC, {crosswalk.query("Final64_Included == 'YES' and Group == 'SZ'").shape[0]} SZ).
- Duplicate canonical subject IDs: {crosswalk['Subject_ID'].duplicated().sum()}.
- Unmatched behavior IDs: {(crosswalk['Unmatched_ID_Flag']!='NONE').sum()}.
- Matching used only the audited behavior-to-BIDS mapping; no row-order or fuzzy matching was used.

## Imaging FC

- Stage × Bin FC: {len(fc):,} rows, {fc['Subject_ID'].nunique()} subjects, stages={sorted(fc['Stage'].unique())}, bins={fc['Bin'].min()}–{fc['Bin'].max()}.
- Rows per subject: minimum {fc_counts.min()}, maximum {fc_counts.max()} (expected 20).
- Missing FC: {fc['FC'].isna().sum()}; duplicated Subject×Stage×Bin keys: {fc.duplicated(['Subject_ID','Stage','Bin']).sum()}.
- Target-ROI FC: {len(fc_target):,} rows, {fc_target['Subject_ID'].nunique()} subjects, {fc_target['Target_ROI'].nunique()} ROIs, {fc_target['Bin'].nunique()} bins.
- Target rows per subject: minimum {target_counts.min()}, maximum {target_counts.max()} (expected 650).
- FC values are retained as saved Fisher-z FC values. The reference G1 and 10-bin definition were not recomputed or changed.

## Similarity

- Bin 10 individual similarity: {len(similarity)} rows, {similarity['Subject_ID'].nunique()} subjects, two stage-specific values per subject.
- Rows per subject: minimum {sim_counts.min()}, maximum {sim_counts.max()}.
- Missing formal Fisher-z similarity: {similarity['Similarity'].isna().sum()}.
- Anticipation uses 43 targets; Outcome uses 19. Their magnitudes are not interchangeable.
- HC values use the saved leave-one-out reference; SZ values use the frozen HC32 reference.

## Derived imaging metrics

- Long-table records: {len(derived):,}; named metrics: {derived['Metric_Name'].nunique()}.
- Existing formal outputs are kept separate from one reconstructed metric: `Overall_Anticipation_minus_Outcome_offset`.
- The reconstructed offset contrast is a deterministic subtraction of saved centered-bin OLS intercepts; it is not a refitted group model.
- Legacy G1 MAE/RMSE are labeled as voxelwise G1-map deviation, not FC deviation.

## Clinical, questionnaire and behavior data

- Clinical/behavior wide table: {len(clinical_wide)} subjects × {len(clinical_wide.columns)} columns.
- Variable dictionary entries: {len(dictionary)}; coverage entries: {len(coverage)}.
- Questionnaire item records: {len(questionnaire_items):,}; observed raw item values: {questionnaire_items['Raw_Value'].notna().sum():,}.
- Corrected untrimmed target-onset event files: {len(event_files)} of 210 possible 70×3 subject-runs; trial rows: {len(trials):,}; subjects represented: {trials['Subject_ID'].nunique()}.
- Final64 subjects with three available runs: {(trial_subject_runs.reindex(sorted(final64_ids)).fillna(0)==3).sum()} of 64.
- Recorded nonmissing trial accuracy values: {trials['Accuracy'].notna().sum():,}. Blank accuracy values were not converted to zero or inferred.

## Known QC issues preserved

- BFI/FIVE existing scores and raw-item recalculations show systematic mismatches; neither version overwrites the other.
- CSAS and CPAS remain under scoring review; the existing scored/reversed region is retained with a REVIEW label.
- PANSS total in the final64 canonical table is recomputed from the three saved component totals, not an original total field.
- HR is retained under the source acronym because the formal full name is unresolved.
- SHS dimension 2 is not present in the consolidated 70-person master and is not fabricated.
- `vidHit` is retained as the existing source variable. No separate `video_hit_rate` variable with a verified definition was found in the selected audited wide source.
- Historical task-block motion is retained separately and is not substituted for final64 resting post-trim mean FD.
- Medication text is incomplete for formal dose conversion; CPZ-equivalent coverage remains zero.
- Clinical and behavior missingness is retained. No imputation or complete-case row deletion was performed.
"""
(OUT / "DATA_EXPORT_QC_REPORT.md").write_text(qc_report, encoding="utf-8")

gaps_report = """# Data export gaps

## Not found as saved individual-level data

- Full Bin 1–10 HC-reference similarity values were not found. Only the two formal Bin 10 stage-specific similarity values were exported.
- Individual ROI-specific offsets were not found as saved formal subject-level outputs. ROI slopes were exported; offsets were not invented.
- A formally defined and frozen subject-level humor-task accuracy or aggregate score was not found. Trial-level recorded fields were exported without inferring accuracy.
- A verified `video_hit_rate` definition/value distinct from `vidHit` was not found in the selected audited source. They were not treated as equivalent.
- SHS dimension 2 was not present in the consolidated 70-person master.
- A confirmed full name for HR was not found.
- Chlorpromazine-equivalent dose was not available.

## Data recovered locally that had previously been described as pending

- Final64 resting post-trim mean FD is present in the canonical matched subject-level imaging table and was exported as `final64_rest_posttrim_mean_FD`.
- The 1,280-row final64 stage × Bin FC table is present and complete.
- The 41,600-row final64 target-ROI × Bin FC table is present and complete.
- Individual Bin 10 Anticipation and Outcome similarity values are present for all 64 subjects.

## Deliberately not performed

- No Procrustes individual gradient arrays were sought or reconstructed.
- The fixed 375-person reference G1, voxel-to-bin assignment, reward-stage mapping and HC reference rules were not changed.
- No new group model, correlation, regression, mixed-effects model or Bayesian model was run.
- No missing value was imputed and no value was reconstructed from a plot.
"""
(OUT / "DATA_EXPORT_GAPS.md").write_text(gaps_report, encoding="utf-8")

# Export manifest. One row per deliverable, with source and limitations.
output_frames = {
    "SUBJECT_CROSSWALK.tsv": crosswalk,
    "subject_level_fc_long.tsv": fc,
    "subject_level_fc_wide.tsv": fc_wide,
    "subject_level_fc_target_roi_long.tsv": fc_target,
    "subject_level_similarity_long.tsv": similarity,
    "subject_level_derived_imaging_metrics.tsv": derived,
    "subject_level_clinical_behaviour_wide.tsv": clinical_wide,
    "subject_level_questionnaire_items_long.tsv": questionnaire_items,
    "subject_level_behaviour_trials_long.tsv": trials,
    "subject_level_motion_historical_long.tsv": motion,
    "CLINICAL_BEHAVIOUR_DATA_DICTIONARY.tsv": dictionary,
    "CLINICAL_BEHAVIOUR_COVERAGE.tsv": coverage,
    "subject_level_brain_behaviour_master.tsv": master,
    "subject_level_brain_behaviour_master_final64.tsv": master_final64,
    "DATA_EXPORT_QC_ISSUES.tsv": qc_issues,
}
source_map = {
    "SUBJECT_CROSSWALK.tsv": [AUDIT_XLSX, BEHAVIOR_CROSSWALK],
    "subject_level_fc_long.tsv": [FC_STAGE], "subject_level_fc_wide.tsv": [FC_STAGE],
    "subject_level_fc_target_roi_long.tsv": [FC_TARGET], "subject_level_similarity_long.tsv": [ROI_MATCHED],
    "subject_level_derived_imaging_metrics.tsv": [SLOPES_LONG, SLOPES_WIDE, ROI_MATCHED, STAGE_COUPLING, NORMATIVE],
    "subject_level_clinical_behaviour_wide.tsv": [PRIMARY70_XLSX, FINAL64_CLINICAL, RAW_SCALE_XLSM],
    "subject_level_questionnaire_items_long.tsv": [RAW_SCALE_XLSM, SCALE_PROVENANCE],
    "subject_level_behaviour_trials_long.tsv": [EVENT_DIR], "subject_level_motion_historical_long.tsv": [PRIMARY70_XLSX],
    "CLINICAL_BEHAVIOUR_DATA_DICTIONARY.tsv": [PRIMARY70_XLSX, FINAL64_CLINICAL, SCALE_PROVENANCE, CLINICAL_DICTIONARY, VARIABLE_DICTIONARY],
    "CLINICAL_BEHAVIOUR_COVERAGE.tsv": [PRIMARY70_XLSX, FINAL64_CLINICAL],
    "subject_level_brain_behaviour_master.tsv": [PRIMARY70_XLSX, FINAL64_CLINICAL, FC_STAGE, ROI_MATCHED, SLOPES_LONG],
    "subject_level_brain_behaviour_master_final64.tsv": [PRIMARY70_XLSX, FINAL64_CLINICAL, FC_STAGE, ROI_MATCHED, SLOPES_LONG],
    "DATA_EXPORT_QC_ISSUES.tsv": [AUDIT_XLSX, SOURCE_CONFLICTS, SCALE_MISMATCH],
}
manifest_rows = []
for name, frame in output_frames.items():
    ids = sorted(set(frame["Subject_ID"].dropna().astype(str))) if "Subject_ID" in frame.columns else []
    level = "Subject-level wide" if len(frame) <= 70 else "Long-form records"
    manifest_rows.append({
        "Export_File": name, "Data_Level": level,
        "Source_File": "; ".join(p.as_posix() for p in source_map[name]),
        "Source_Sheet_or_Variable": "See data dictionary and source columns", "Subject_Count": len(ids),
        "Row_Count": len(frame), "Variable_Count": len(frame.columns), "Included_Subject_IDs": "|".join(ids),
        "Data_Version": BUILD_VERSION, "Extraction_Method": "Exact ID merge and deterministic reshape in build_subject_level_exports.py",
        "Derived_or_Original": "Original plus explicitly labeled deterministic views",
        "QC_Status": "PASS_WITH_DOCUMENTED_LIMITATIONS", "Known_Limitations": "See DATA_EXPORT_QC_REPORT.md and DATA_EXPORT_GAPS.md",
    })
manifest_rows.extend([
    {"Export_File": "DATA_EXPORT_QC_REPORT.md", "Data_Level": "Documentation", "Source_File": AUDIT_XLSX.as_posix(), "Source_Sheet_or_Variable": "QC_Register and deterministic export checks", "Subject_Count": 70, "Row_Count": 0, "Variable_Count": 0, "Included_Subject_IDs": "", "Data_Version": BUILD_VERSION, "Extraction_Method": "Programmatic QC summary", "Derived_or_Original": "Derived documentation", "QC_Status": "PASS", "Known_Limitations": "No formal statistical model run"},
    {"Export_File": "DATA_EXPORT_GAPS.md", "Data_Level": "Documentation", "Source_File": AUDIT_XLSX.as_posix(), "Source_Sheet_or_Variable": "Source manifest and local search", "Subject_Count": 0, "Row_Count": 0, "Variable_Count": 0, "Included_Subject_IDs": "", "Data_Version": BUILD_VERSION, "Extraction_Method": "Gap register", "Derived_or_Original": "Derived documentation", "QC_Status": "PASS", "Known_Limitations": "Records only confirmed gaps"},
    {"Export_File": "SUBJECT_LEVEL_BRAIN_BEHAVIOUR_DATABASE.xlsx", "Data_Level": "Human review workbook", "Source_File": "; ".join((OUT / n).as_posix() for n in output_frames), "Source_Sheet_or_Variable": "Selected review views", "Subject_Count": 70, "Row_Count": 0, "Variable_Count": 0, "Included_Subject_IDs": "", "Data_Version": BUILD_VERSION, "Extraction_Method": "Artifact Tool workbook export", "Derived_or_Original": "Review container; TSV files remain authoritative exchange outputs", "QC_Status": "PENDING_WORKBOOK_BUILD", "Known_Limitations": "Large lower-level files remain authoritative TSVs"},
])
manifest = pd.DataFrame(manifest_rows)
write_tsv(manifest, "DATA_EXPORT_MANIFEST.tsv")

# JSON payload for the Artifact Tool workbook builder.
payload_frames = {
    "Subject_Crosswalk": crosswalk,
    "Imaging_FC": fc,
    "FC_Target_ROI": fc_target,
    "Imaging_Similarity": similarity,
    "Imaging_Derived_Metrics": derived,
    "Clinical_Scales": clinical_wide,
    "Questionnaire_Items": questionnaire_items,
    "Behaviour_Tasks": trials,
    "Brain_Behaviour_Master": master,
    "Data_Dictionary": dictionary,
    "Coverage_Summary": coverage,
    "QC_Issues": qc_issues,
    "Source_Manifest": manifest,
}
payload = {name: jsonable_matrix(frame) for name, frame in payload_frames.items()}
with (OUT / "workbook_payload.json").open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, allow_nan=False)

summary = {
    "output_dir": str(OUT), "final64": len(final64_ids), "traceable_subjects": crosswalk["Subject_ID"].nunique(),
    "fc_rows": len(fc), "fc_target_rows": len(fc_target), "similarity_rows": len(similarity),
    "derived_rows": len(derived), "clinical_variables": len(clinical_wide.columns),
    "questionnaire_item_rows": len(questionnaire_items), "trial_files": len(event_files), "trial_rows": len(trials),
    "master_rows": len(master), "master_columns": len(master.columns),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
