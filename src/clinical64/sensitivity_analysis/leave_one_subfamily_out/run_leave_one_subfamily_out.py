#!/usr/bin/env python3
"""Leave-one-assigned-subfamily-out Pearson contribution sensitivity.

Each deletion condition restarts from the same frozen 65-ROI Bin-10 profiles.
The three Not_assigned ROIs remain in every condition. The script reproduces
the Full and Minus-Insula checkpoints before running the other deletions.
"""

from __future__ import annotations

import os
import hashlib
import json
import platform
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm


PROJECT = Path(os.environ["STRIATAL_PROJECT_ROOT"])
REPO = Path(os.environ["STRIATAL_RELEASE_ROOT"])
PREVIOUS = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP3_LocalContribution_InsulaSensitivity'))
OUT = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP3_LocalContribution_AllSubfamilies'))

COV_INPUT = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "covariates" / "current64_covariates.tsv"
ROI_META = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "reward_stage_mapping" / "roi_metadata_stage_subfamily_v2.tsv"
FC_ROOT = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "fc_fingerprints_2608x65"
MAPPING = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "voxel_order" / "row_mapping_2847_to_2608.npy"
CURRENT_BINS = Path(os.environ["STRIATAL_CURRENT_BIN_TABLE"])
HISTORICAL_RESULTS = PROJECT / "FINAL_64SUBJECT_RESULTS_PACKAGE" / "tables" / "25_03_Bin10_subfamily_drivers_FINAL.tsv"
HISTORICAL_SCRIPT = REPO / "src" / "clinical64" / "sma_ifg" / "run_bin10_subfamily.sh"
PREVIOUS_MINUS_RESULTS = PREVIOUS / "minus_insula_53roi_subfamily_results.tsv"

EXPECTED_CURRENT_BINS_SHA256 = "26d268283702db209940198e76b8630585e54259cd742016951aaa7d8427292a"
EXPECTED_MAPPING_SHA256 = "bb12bfa947c09778d3d353a765a397c3a5bb17880bde98f6c147d7379289a268"
EXPECTED = json.loads(os.environ["STRIATAL_EXPECTED_CONTRIBUTIONS_JSON"])
FULL_LABEL = "Full"
ALPHA = 0.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path, na_rep: str = "") -> None:
    frame.to_csv(path, sep="\t", index=False, float_format="%.17g", na_rep=na_rep, lineterminator="\n")


def write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("BH input must be finite and between 0 and 1")
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    adjusted = p[order] * n / np.arange(1, n + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(n, dtype=float)
    result[order] = adjusted
    return result


@dataclass
class OLSHC3:
    beta: np.ndarray
    se: np.ndarray
    p: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    residual_df: int


def fit_ols_hc3(y: np.ndarray, x: np.ndarray) -> OLSHC3:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    if np.any(1.0 - leverage <= 0):
        raise RuntimeError("HC3 leverage reached one")
    scaled = residual / (1.0 - leverage)
    covariance = xtx_inv @ (x.T @ ((scaled**2)[:, None] * x)) @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z = beta / se
    p = 2.0 * norm.sf(np.abs(z))
    return OLSHC3(beta, se, p, beta - 1.96 * se, beta + 1.96 * se, int(len(y) - x.shape[1]))


def prepare_covariates() -> pd.DataFrame:
    cov = pd.read_csv(COV_INPUT, sep="\t").rename(
        columns={"Subject": "subject", "Group": "group", "Age": "age", "Sex": "sex", "MeanFD": "meanFD"}
    )
    cov["subject"] = cov["subject"].astype(str)
    cov["group"] = cov["group"].astype(str).str.upper()
    cov["age"] = pd.to_numeric(cov["age"], errors="raise")
    cov["meanFD"] = pd.to_numeric(cov["meanFD"], errors="raise")
    cov["sex_num"] = cov["sex"].astype(str).str.lower().map({"female": 0.0, "f": 0.0, "male": 1.0, "m": 1.0})
    if cov["sex_num"].isna().any():
        raise RuntimeError("Unrecognized sex coding")
    cov["group_sz"] = (cov["group"] == "SZ").astype(int)
    cov["age_c"] = cov["age"] - cov["age"].mean()
    cov["sex_c"] = cov["sex_num"] - cov["sex_num"].mean()
    cov["meanFD_c"] = cov["meanFD"] - cov["meanFD"].mean()
    if len(cov) != 64 or cov["subject"].nunique() != 64 or cov["group"].value_counts().to_dict() != {"SZ": 32, "HC": 32}:
        raise RuntimeError("Expected 64 unique subjects split 32 SZ and 32 HC")
    return cov.reset_index(drop=True)


def prepare_metadata() -> tuple[pd.DataFrame, list[str]]:
    meta = pd.read_csv(ROI_META, sep="\t")
    meta["roi_index0"] = pd.to_numeric(meta["roi_index0"], errors="raise").astype(int)
    meta = meta.sort_values("roi_index0").reset_index(drop=True)
    if len(meta) != 65 or meta["roi_index0"].tolist() != list(range(65)):
        raise RuntimeError("Expected the exact 65-ROI order")
    if int((meta["stage"] != "Not_assigned").sum()) != 62 or int((meta["stage"] == "Not_assigned").sum()) != 3:
        raise RuntimeError("Expected 62 assigned and 3 unassigned ROIs")
    assigned = meta.loc[meta["stage"] != "Not_assigned", "subfamily"].drop_duplicates().tolist()
    if len(assigned) != 13:
        raise RuntimeError(f"Expected 13 assigned subfamilies; got {len(assigned)}")
    if meta.loc[meta["stage"] == "Not_assigned", "subfamily"].unique().tolist() != ["Midbrain_pallidal"]:
        raise RuntimeError("Unassigned ROI definition changed")
    return meta, assigned


def load_fingerprints(cov: pd.DataFrame) -> tuple[np.ndarray, dict, pd.DataFrame]:
    if sha256(CURRENT_BINS) != EXPECTED_CURRENT_BINS_SHA256:
        raise RuntimeError("Current-bin input hash changed")
    if sha256(MAPPING) != EXPECTED_MAPPING_SHA256:
        raise RuntimeError("Voxel mapping hash changed")
    bins = pd.read_csv(CURRENT_BINS, sep="\t")["G1_bin_current"].astype(int).to_numpy()
    mapping = np.asarray(np.load(MAPPING)).reshape(-1).astype(int)
    common_bins = bins[mapping]
    mask = common_bins == 10
    if len(bins) != 2847 or len(mapping) != 2608 or int(mask.sum()) != 285:
        raise RuntimeError("Fixed current Bin-10 definition changed")
    profiles = np.empty((64, 65), dtype=float)
    hashes = []
    for i, subject in enumerate(cov["subject"]):
        path = FC_ROOT / f"{subject}_fingerprint_z_2608x65.npy"
        array = np.asarray(np.load(path), dtype=float)
        if array.shape == (65, 2608):
            array = array.T
        if array.shape != (2608, 65) or not np.isfinite(array).all():
            raise RuntimeError(f"Bad FC array: {path}")
        profiles[i] = array[mask].mean(axis=0)
        hashes.append({"input_type": "subject_fc", "subject": subject, "path": str(path), "sha256": sha256(path)})
    qc = {
        "bin10_voxels": int(mask.sum()),
        "profile_shape": list(profiles.shape),
        "common_bin_counts": {str(k): int(v) for k, v in pd.Series(common_bins).value_counts().sort_index().items()},
    }
    return profiles, qc, pd.DataFrame(hashes)


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(frame)),
            frame["group_sz"].to_numpy(float),
            frame["age_c"].to_numpy(float),
            frame["sex_c"].to_numpy(float),
            frame["meanFD_c"].to_numpy(float),
        ]
    )


def decompose(values: np.ndarray, template: np.ndarray) -> tuple[float, np.ndarray, float]:
    k = len(values)
    zx = (values - values.mean()) / values.std(ddof=1)
    zt = (template - template.mean()) / template.std(ddof=1)
    contribution = zx * zt / float(k - 1)
    direct = float(np.corrcoef(values, template)[0, 1])
    error = float(abs(contribution.sum() - direct))
    if error > 1e-12:
        raise RuntimeError(f"Pearson identity failed: {error}")
    return direct, contribution, error


def analyze_condition(
    deleted: str,
    profiles: np.ndarray,
    cov: pd.DataFrame,
    meta: pd.DataFrame,
    assigned_families: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if deleted == FULL_LABEL:
        keep = np.ones(65, dtype=bool)
        deleted_roi_count = 0
    else:
        if deleted not in assigned_families:
            raise ValueError(deleted)
        keep = meta["subfamily"].to_numpy() != deleted
        deleted_roi_count = int((~keep).sum())
    kept_meta = meta.loc[keep].reset_index(drop=True)
    selected = profiles[:, keep]
    if int((kept_meta["stage"] == "Not_assigned").sum()) != 3:
        raise RuntimeError(f"{deleted}: unassigned ROIs were not all retained")
    if selected.shape[1] != 65 - deleted_roi_count:
        raise RuntimeError("ROI count mismatch")

    hc_idx = np.flatnonzero(cov["group"].to_numpy() == "HC")
    hc_mean = selected[hc_idx].mean(axis=0)
    roi_rows = []
    subject_rows = []
    identity_rows = []
    for i, row in cov.iterrows():
        if row["group"] == "HC":
            template = selected[hc_idx[hc_idx != i]].mean(axis=0)
            reference = "HC_LOO31"
        else:
            template = hc_mean
            reference = "HC32"
        direct, contribution, error = decompose(selected[i], template)
        subject_rows.append(
            {
                "subject": row["subject"],
                "group": row["group"],
                "group_sz": row["group_sz"],
                "age_c": row["age_c"],
                "sex_c": row["sex_c"],
                "meanFD_c": row["meanFD_c"],
                "profile_r": direct,
            }
        )
        identity_rows.append(
            {
                "deleted_subfamily": deleted,
                "subject": row["subject"],
                "group": row["group"],
                "remaining_roi_count": selected.shape[1],
                "reference": reference,
                "profile_r": direct,
                "contribution_sum": float(contribution.sum()),
                "absolute_error": error,
            }
        )
        for j, value in enumerate(contribution):
            mr = kept_meta.iloc[j]
            roi_rows.append(
                {
                    "subject": row["subject"],
                    "group": row["group"],
                    "group_sz": row["group_sz"],
                    "age_c": row["age_c"],
                    "sex_c": row["sex_c"],
                    "meanFD_c": row["meanFD_c"],
                    "subfamily": mr["subfamily"],
                    "stage": mr["stage"],
                    "roi_index0": int(mr["roi_index0"]),
                    "r_contribution": float(value),
                }
            )

    subject = pd.DataFrame(subject_rows)
    roi = pd.DataFrame(roi_rows)
    aggregate = (
        roi.groupby(
            ["subject", "group", "group_sz", "age_c", "sex_c", "meanFD_c", "subfamily", "stage"],
            as_index=False,
            sort=True,
        )
        .agg(contribution_sum=("r_contribution", "sum"), roi_count=("roi_index0", "nunique"))
    )
    records = []
    for (tested, stage), subset in aggregate.groupby(["subfamily", "stage"], sort=True):
        subset = subset.sort_values("subject").reset_index(drop=True)
        fit = fit_ols_hc3(subset["contribution_sum"].to_numpy(float), design_matrix(subset))
        beta = float(fit.beta[1])
        records.append(
            {
                "deleted_subfamily": deleted,
                "deleted_roi_count": deleted_roi_count,
                "remaining_roi_count": int(selected.shape[1]),
                "tested_subfamily": tested,
                "stage": stage,
                "roi_count": int(subset["roi_count"].iloc[0]),
                "beta_SZ_minus_HC": beta,
                "SE_HC3": float(fit.se[1]),
                "CI95_low": float(fit.ci_low[1]),
                "CI95_high": float(fit.ci_high[1]),
                "p_raw": float(fit.p[1]),
                "q_FDR_13_fixed": np.nan,
                "residual_df": fit.residual_df,
            }
        )
    results = pd.DataFrame(records)
    p_family = []
    for family in assigned_families:
        if family == deleted:
            p_family.append(1.0)
        else:
            hit = results.loc[results["tested_subfamily"] == family, "p_raw"]
            if len(hit) != 1:
                raise RuntimeError(f"{deleted}: expected one p-value for {family}")
            p_family.append(float(hit.iloc[0]))
    q_family = bh_adjust(np.asarray(p_family))
    for family, q in zip(assigned_families, q_family):
        if family != deleted:
            results.loc[results["tested_subfamily"] == family, "q_FDR_13_fixed"] = q

    total_fit = fit_ols_hc3(subject["profile_r"].to_numpy(float), design_matrix(subject))
    total_beta = float(total_fit.beta[1])
    beta_sum = float(results["beta_SZ_minus_HC"].sum())
    if abs(total_beta - beta_sum) > 3e-12:
        raise RuntimeError(f"{deleted}: subfamily beta additivity failed")
    qc = {
        "deleted_subfamily": deleted,
        "deleted_roi_count": deleted_roi_count,
        "remaining_roi_count": int(selected.shape[1]),
        "pearson_denominator": int(selected.shape[1] - 1),
        "max_identity_error": float(pd.DataFrame(identity_rows)["absolute_error"].max()),
        "whole_profile_beta": total_beta,
        "sum_subfamily_betas": beta_sum,
        "beta_additivity_error": abs(total_beta - beta_sum),
        "tested_retained_subfamilies": int(len(results)),
        "unassigned_roi_count_retained": int((kept_meta["stage"] == "Not_assigned").sum()),
    }
    return results, pd.DataFrame(identity_rows), qc


def validate_full(full: pd.DataFrame) -> pd.DataFrame:
    historical = pd.read_csv(HISTORICAL_RESULTS, sep="\t")
    merged = historical.merge(
        full,
        left_on=["subfamily", "stage"],
        right_on=["tested_subfamily", "stage"],
        how="outer",
        suffixes=("_historical", "_reproduced"),
        indicator=True,
        validate="one_to_one",
    )
    if len(merged) != 14 or not (merged["_merge"] == "both").all():
        raise RuntimeError("Full historical subfamily keys did not reproduce")
    rows = []
    fields = [
        ("N_ROIs", "roi_count", 0.0),
        ("beta_SZ_minus_HC", "beta_SZ_minus_HC", 2e-12),
        ("SE_HC3", "SE_HC3", 2e-12),
        ("CI95_low", "CI95_low", 3e-12),
        ("CI95_high", "CI95_high", 3e-12),
        ("p_raw", "p_raw", 2e-12),
    ]
    for _, row in merged.iterrows():
        for old, new, tolerance in fields:
            old_col = old if old not in full.columns else f"{old}_historical"
            new_col = new if new not in historical.columns else f"{new}_reproduced"
            a = float(row[old_col])
            b = float(row[new_col])
            error = abs(a - b)
            rows.append(
                {
                    "checkpoint": "Full",
                    "subfamily": row["subfamily"],
                    "field": old,
                    "previous": a,
                    "reproduced": b,
                    "absolute_error": error,
                    "tolerance": tolerance,
                    "pass": error <= tolerance,
                }
            )
        if row["stage"] != "Not_assigned":
            a = float(row["q_FDR_13_assigned"])
            b = float(row["q_FDR_13_fixed"])
            error = abs(a - b)
            rows.append(
                {
                    "checkpoint": "Full",
                    "subfamily": row["subfamily"],
                    "field": "q_FDR_13",
                    "previous": a,
                    "reproduced": b,
                    "absolute_error": error,
                    "tolerance": 2e-12,
                    "pass": error <= 2e-12,
                }
            )
    audit = pd.DataFrame(rows)
    if not audit["pass"].all():
        raise RuntimeError("Full reproduction gate failed")
    return audit


def validate_minus_insula(insula: pd.DataFrame) -> pd.DataFrame:
    previous = pd.read_csv(PREVIOUS_MINUS_RESULTS, sep="\t")
    merged = previous.merge(
        insula,
        left_on=["subfamily", "stage"],
        right_on=["tested_subfamily", "stage"],
        suffixes=("_previous", "_reproduced"),
        validate="one_to_one",
    )
    if len(merged) != 13:
        raise RuntimeError("Minus Insula retained subfamily count changed")
    rows = []
    fields = ["beta_SZ_minus_HC", "SE_HC3", "CI95_low", "CI95_high", "p_raw"]
    for _, row in merged.iterrows():
        for field in fields:
            a = float(row[f"{field}_previous"])
            b = float(row[f"{field}_reproduced"])
            error = abs(a - b)
            rows.append(
                {
                    "checkpoint": "Minus Insula",
                    "subfamily": row["subfamily"],
                    "field": field,
                    "previous": a,
                    "reproduced": b,
                    "absolute_error": error,
                    "tolerance": 3e-12,
                    "pass": error <= 3e-12,
                }
            )
        if row["stage"] != "Not_assigned":
            a = float(row["q_FDR_13_prespecified"])
            b = float(row["q_FDR_13_fixed"])
            error = abs(a - b)
            rows.append(
                {
                    "checkpoint": "Minus Insula",
                    "subfamily": row["subfamily"],
                    "field": "q_FDR_13",
                    "previous": a,
                    "reproduced": b,
                    "absolute_error": error,
                    "tolerance": 3e-12,
                    "pass": error <= 3e-12,
                }
            )
    audit = pd.DataFrame(rows)
    if not audit["pass"].all():
        raise RuntimeError("Minus Insula reproduction gate failed")
    return audit


def validate_target_checkpoints(full: pd.DataFrame, insula: pd.DataFrame) -> None:
    for condition, frame in [("Full", full), ("Insula", insula)]:
        for target in ("SMA", "IFG"):
            row = frame.loc[frame["tested_subfamily"] == target].iloc[0]
            beta_expected, q_expected = EXPECTED[condition][target]
            if abs(float(row["beta_SZ_minus_HC"]) - beta_expected) > 3e-12:
                raise RuntimeError(f"{condition} {target} beta gate failed")
            if abs(float(row["q_FDR_13_fixed"]) - q_expected) > 3e-12:
                raise RuntimeError(f"{condition} {target} q gate failed")


def make_table1(all_results: pd.DataFrame, assigned_families: list[str], meta: pd.DataFrame) -> pd.DataFrame:
    full_beta = all_results.loc[all_results["deleted_subfamily"] == FULL_LABEL].set_index("tested_subfamily")["beta_SZ_minus_HC"]
    rows = []
    for deleted in [FULL_LABEL] + assigned_families:
        condition = all_results[all_results["deleted_subfamily"] == deleted]
        deleted_count = 0 if deleted == FULL_LABEL else int((meta["subfamily"] == deleted).sum())
        row = {
            "deleted_subfamily": deleted,
            "deleted_roi_count": deleted_count,
            "remaining_roi_count": 65 - deleted_count,
        }
        for target in ("SMA", "IFG"):
            hit = condition[condition["tested_subfamily"] == target]
            if hit.empty:
                row.update(
                    {
                        f"{target}_beta": np.nan,
                        f"{target}_raw_p": np.nan,
                        f"{target}_FDR_q_13_fixed": np.nan,
                        f"{target}_effect_direction": "N/A",
                        f"{target}_delta_beta_vs_full": np.nan,
                        f"{target}_q_lt_0_05": "N/A",
                    }
                )
            else:
                rr = hit.iloc[0]
                beta = float(rr["beta_SZ_minus_HC"])
                direction = "Negative" if beta < 0 else ("Positive" if beta > 0 else "Zero")
                row.update(
                    {
                        f"{target}_beta": beta,
                        f"{target}_raw_p": float(rr["p_raw"]),
                        f"{target}_FDR_q_13_fixed": float(rr["q_FDR_13_fixed"]),
                        f"{target}_effect_direction": direction,
                        f"{target}_delta_beta_vs_full": beta - float(full_beta[target]),
                        f"{target}_q_lt_0_05": "YES" if float(rr["q_FDR_13_fixed"]) < ALPHA else "NO",
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def make_table2(all_results: pd.DataFrame) -> pd.DataFrame:
    full = all_results.loc[all_results["deleted_subfamily"] == FULL_LABEL].set_index("tested_subfamily")
    table = all_results.copy()
    table["delta_beta_vs_full"] = table.apply(
        lambda row: float(row["beta_SZ_minus_HC"] - full.loc[row["tested_subfamily"], "beta_SZ_minus_HC"]), axis=1
    )
    table["sign_consistency"] = table.apply(
        lambda row: "YES"
        if np.sign(row["beta_SZ_minus_HC"]) == np.sign(full.loc[row["tested_subfamily"], "beta_SZ_minus_HC"])
        else "NO",
        axis=1,
    )
    table["FDR_family_member"] = np.where(table["stage"] == "Not_assigned", "NO", "YES")
    table["q_lt_0_05"] = np.where(
        table["q_FDR_13_fixed"].isna(), "N/A", np.where(table["q_FDR_13_fixed"] < ALPHA, "YES", "NO")
    )
    cols = [
        "deleted_subfamily",
        "deleted_roi_count",
        "remaining_roi_count",
        "tested_subfamily",
        "stage",
        "roi_count",
        "beta_SZ_minus_HC",
        "SE_HC3",
        "CI95_low",
        "CI95_high",
        "p_raw",
        "q_FDR_13_fixed",
        "delta_beta_vs_full",
        "sign_consistency",
        "FDR_family_member",
        "q_lt_0_05",
    ]
    return table[cols]


def make_table3(table2: pd.DataFrame, assigned_families: list[str]) -> pd.DataFrame:
    full = table2[table2["deleted_subfamily"] == FULL_LABEL].set_index("tested_subfamily")
    rows = []
    for tested in full.index:
        sensitivity = table2[(table2["deleted_subfamily"] != FULL_LABEL) & (table2["tested_subfamily"] == tested)].copy()
        deltas = sensitivity["delta_beta_vs_full"].to_numpy(float)
        impact_i = int(np.argmax(np.abs(deltas)))
        family_member = tested in assigned_families
        rows.append(
            {
                "subfamily": tested,
                "stage": full.loc[tested, "stage"],
                "full_beta": float(full.loc[tested, "beta_SZ_minus_HC"]),
                "full_FDR_q_13": float(full.loc[tested, "q_FDR_13_fixed"]) if family_member else np.nan,
                "evaluable_deletion_conditions": int(len(sensitivity)),
                "direction_preserved_count": int((sensitivity["sign_consistency"] == "YES").sum()),
                "direction_preserved_proportion": float((sensitivity["sign_consistency"] == "YES").mean()),
                "FDR_q_lt_0_05_count": int((sensitivity["q_lt_0_05"] == "YES").sum()) if family_member else np.nan,
                "FDR_q_lt_0_05_proportion": float((sensitivity["q_lt_0_05"] == "YES").mean()) if family_member else np.nan,
                "beta_min_across_deletions": float(sensitivity["beta_SZ_minus_HC"].min()),
                "beta_max_across_deletions": float(sensitivity["beta_SZ_minus_HC"].max()),
                "delta_beta_min": float(deltas.min()),
                "delta_beta_max": float(deltas.max()),
                "max_absolute_delta_beta": float(abs(deltas[impact_i])),
                "deleted_subfamily_max_impact": sensitivity.iloc[impact_i]["deleted_subfamily"],
                "FDR_family_member": "YES" if family_member else "NO",
                "interpretation_note": "Significance frequency is descriptive and not the sole robustness criterion",
            }
        )
    return pd.DataFrame(rows).sort_values("max_absolute_delta_beta", ascending=False).reset_index(drop=True)


def input_provenance(fc_hashes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, path in [
        ("covariates", COV_INPUT),
        ("roi_metadata", ROI_META),
        ("voxel_mapping", MAPPING),
        ("fixed_current_bins", CURRENT_BINS),
        ("historical_results", HISTORICAL_RESULTS),
        ("historical_script", HISTORICAL_SCRIPT),
        ("previous_minus_insula_results", PREVIOUS_MINUS_RESULTS),
    ]:
        rows.append({"input_type": kind, "subject": "", "path": str(path), "sha256": sha256(path)})
    return pd.concat([pd.DataFrame(rows), fc_hashes], ignore_index=True)


def write_blocking_report(message: str) -> None:
    text = f"""# Leave-One-Subfamily-Out 阻塞报告\n\n状态：**STOPPED_AT_REPRODUCTION_GATE**\n\nFull 或 Minus Insula 复现未通过，因此没有执行其余 Subfamily 删除。\n\n```text\n{message}\n```\n"""
    (OUT / "BLOCKING_REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("STEP 3 - Leave-One-Subfamily-Out Pearson Contribution Sensitivity")
    print("Start UTC:", datetime.now(timezone.utc).isoformat())
    cov = prepare_covariates()
    meta, assigned_families = prepare_metadata()
    profiles, bin_qc, fc_hashes = load_fingerprints(cov)
    print("Data audit: 65 ROIs, 62 assigned, 13 assigned subfamilies, 3 unassigned ROIs")

    full, full_identity, full_qc = analyze_condition(FULL_LABEL, profiles, cov, meta, assigned_families)
    minus_insula, insula_identity, insula_qc = analyze_condition("Insula", profiles, cov, meta, assigned_families)
    gate_full = validate_full(full)
    gate_insula = validate_minus_insula(minus_insula)
    validate_target_checkpoints(full, minus_insula)
    gate = pd.concat([gate_full, gate_insula], ignore_index=True)
    print("Full and Minus Insula reproduction gates: PASS")

    all_results = [full, minus_insula]
    all_identity = [full_identity, insula_identity]
    condition_qc = [full_qc, insula_qc]
    for deleted in assigned_families:
        if deleted == "Insula":
            continue
        print("Analyzing deletion:", deleted)
        result, identity, qc = analyze_condition(deleted, profiles, cov, meta, assigned_families)
        all_results.append(result)
        all_identity.append(identity)
        condition_qc.append(qc)

    all_results = pd.concat(all_results, ignore_index=True)
    order = {name: i for i, name in enumerate([FULL_LABEL] + assigned_families)}
    family_order = {name: i for i, name in enumerate(meta["subfamily"].drop_duplicates().tolist())}
    all_results["_condition_order"] = all_results["deleted_subfamily"].map(order)
    all_results["_family_order"] = all_results["tested_subfamily"].map(family_order)
    all_results = all_results.sort_values(["_condition_order", "_family_order"]).drop(columns=["_condition_order", "_family_order"]).reset_index(drop=True)
    identity = pd.concat(all_identity, ignore_index=True)
    qc_frame = pd.DataFrame(condition_qc).sort_values("deleted_subfamily", key=lambda s: s.map(order)).reset_index(drop=True)

    if len(qc_frame) != 14 or len(identity) != 14 * 64 or float(identity["absolute_error"].max()) > 1e-12:
        raise RuntimeError("Global condition or Pearson identity QC failed")
    if not (qc_frame["unassigned_roi_count_retained"] == 3).all():
        raise RuntimeError("At least one deletion lost an unassigned ROI")

    table1 = make_table1(all_results, assigned_families, meta)
    table2 = make_table2(all_results)
    table3 = make_table3(table2, assigned_families)
    roi_audit = meta.copy()
    roi_audit["assignment_status"] = np.where(
        roi_audit["stage"].eq("Not_assigned"), "unassigned_retained", "assigned"
    )
    subfamily_audit = (
        roi_audit.groupby(["subfamily", "stage", "assignment_status"], as_index=False)
        .agg(roi_count=("roi_index0", "nunique"))
        .sort_values(["assignment_status", "subfamily"])
        .reset_index(drop=True)
    )

    write_tsv(table1, OUT / "table1_sma_ifg_stability_overview.tsv", na_rep="N/A")
    write_tsv(table2, OUT / "table2_complete_contribution_results.tsv", na_rep="N/A")
    write_tsv(table3, OUT / "table3_robustness_summary.tsv", na_rep="N/A")
    write_tsv(roi_audit, OUT / "roi_assignment_audit.tsv", na_rep="N/A")
    write_tsv(subfamily_audit, OUT / "subfamily_roi_count_audit.tsv", na_rep="N/A")
    write_tsv(identity, OUT / "pearson_contribution_identity_qc.tsv")
    write_tsv(qc_frame, OUT / "condition_level_qc.tsv")
    write_tsv(gate, OUT / "reproduction_gate_audit.tsv")
    write_tsv(input_provenance(fc_hashes), OUT / "input_provenance_sha256.tsv")

    qc = {
        "status": "SUCCESS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_project_modified": False,
        "sample": {"n": 64, "HC": 32, "SZ": 32},
        "roi_audit": {
            "total_rois": 65,
            "assigned_rois": 62,
            "assigned_subfamilies": 13,
            "unassigned_rois_retained_every_condition": 3,
            "unassigned_subfamily": "Midbrain_pallidal",
        },
        "bin10": bin_qc,
        "conditions": [FULL_LABEL] + assigned_families,
        "fdr_rule": "BH across fixed family of 13 assigned subfamilies; deleted subfamily enters calculation as p=1 and is displayed N/A; Midbrain_pallidal is excluded",
        "pearson_identity_max_error": float(identity["absolute_error"].max()),
        "beta_additivity_max_error": float(qc_frame["beta_additivity_error"].max()),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    write_json(qc, OUT / "analysis_qc.json")
    (OUT / "BLOCKING_REPORT.md").unlink(missing_ok=True)

    print("All 14 conditions completed")
    print("Maximum |sum(contributions)-r|:", f"{identity['absolute_error'].max():.3e}")
    print("Table 1 target overview:")
    print(table1.to_string(index=False))
    print("Completed UTC:", datetime.now(timezone.utc).isoformat())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        write_blocking_report(message)
        print(message, file=sys.stderr)
        raise SystemExit(1)
