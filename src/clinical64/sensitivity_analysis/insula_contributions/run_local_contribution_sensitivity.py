#!/usr/bin/env python3
"""STEP 3: Bin-10 local Pearson-contribution sensitivity to Insula removal.

This script reads the frozen 64-subject FC arrays and fixed current-bin map,
reproduces the historical 65-ROI contribution analysis, and only after that
gate passes recomputes Pearson normalization on the 53 non-Insula ROIs.

It never modifies the source project. All outputs are written beside this file.
"""

from __future__ import annotations

import os
import hashlib
import json
import math
import platform
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, ttest_1samp


PROJECT = Path(os.environ["STRIATAL_PROJECT_ROOT"])
REPO = Path(os.environ["STRIATAL_RELEASE_ROOT"])
OUT = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP3_LocalContribution_InsulaSensitivity'))

COV_INPUT = (
    PROJECT
    / "CURRENT64_CANONICAL_RERUN"
    / "01_FROZEN_INPUTS"
    / "covariates"
    / "current64_covariates.tsv"
)
ROI_META = (
    PROJECT
    / "CURRENT64_CANONICAL_RERUN"
    / "01_FROZEN_INPUTS"
    / "reward_stage_mapping"
    / "roi_metadata_stage_subfamily_v2.tsv"
)
FC_ROOT = (
    PROJECT
    / "CURRENT64_CANONICAL_RERUN"
    / "01_FROZEN_INPUTS"
    / "fc_fingerprints_2608x65"
)
MAPPING = (
    PROJECT
    / "CURRENT64_CANONICAL_RERUN"
    / "01_FROZEN_INPUTS"
    / "voxel_order"
    / "row_mapping_2847_to_2608.npy"
)
CURRENT_BINS = Path(
    os.environ["STRIATAL_CURRENT_BIN_TABLE"]
)
HISTORICAL_RESULTS = (
    PROJECT
    / "FINAL_64SUBJECT_RESULTS_PACKAGE"
    / "tables"
    / "25_03_Bin10_subfamily_drivers_FINAL.tsv"
)
HISTORICAL_SCRIPT = (
    REPO / "src" / "clinical64" / "sma_ifg" / "run_bin10_subfamily.sh"
)

EXPECTED_CURRENT_BINS_SHA256 = "26d268283702db209940198e76b8630585e54259cd742016951aaa7d8427292a"
EXPECTED_MAPPING_SHA256 = "bb12bfa947c09778d3d353a765a397c3a5bb17880bde98f6c147d7379289a268"
EXPECTED_SMA_BETA = float(os.environ["STRIATAL_EXPECTED_SMA_BETA"])
EXPECTED_SMA_Q = float(os.environ["STRIATAL_EXPECTED_SMA_Q"])
EXPECTED_IFG_BETA = float(os.environ["STRIATAL_EXPECTED_IFG_BETA"])
EXPECTED_IFG_Q = float(os.environ["STRIATAL_EXPECTED_IFG_Q"])

FULL_SET = "Full_65ROI"
MINUS_SET = "Minus_Insula_53ROI"
ALPHA = 0.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, float_format="%.17g", lineterminator="\n")


def write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, matching statsmodels/R p.adjust."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("BH input must be a finite vector in [0, 1]")
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * n / np.arange(1, n + 1, dtype=float)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.minimum(ranked, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = ranked
    return out


@dataclass
class OLSHC3:
    beta: np.ndarray
    se: np.ndarray
    p: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    residual_df: int
    max_leverage: float


def fit_ols_hc3(y: np.ndarray, x: np.ndarray) -> OLSHC3:
    """OLS with HC3 covariance and normal-reference Wald inference."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("Bad OLS shapes")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Non-finite OLS input")
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    if np.any(1.0 - leverage <= 0):
        raise RuntimeError("HC3 leverage is not strictly below 1")
    scaled = resid / (1.0 - leverage)
    meat = x.T @ ((scaled**2)[:, None] * x)
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z = beta / se
    p = 2.0 * norm.sf(np.abs(z))
    return OLSHC3(
        beta=beta,
        se=se,
        p=p,
        ci_low=beta - 1.96 * se,
        ci_high=beta + 1.96 * se,
        residual_df=int(y.size - x.shape[1]),
        max_leverage=float(leverage.max()),
    )


def prepare_covariates() -> pd.DataFrame:
    cov = pd.read_csv(COV_INPUT, sep="\t").rename(
        columns={"Subject": "subject", "Group": "group", "Age": "age", "Sex": "sex", "MeanFD": "meanFD"}
    )
    required = {"subject", "group", "age", "sex", "meanFD"}
    if not required.issubset(cov.columns):
        raise RuntimeError(f"Missing covariate columns: {sorted(required - set(cov.columns))}")
    cov["subject"] = cov["subject"].astype(str)
    cov["group"] = cov["group"].astype(str).str.upper()
    cov["age"] = pd.to_numeric(cov["age"], errors="raise")
    cov["meanFD"] = pd.to_numeric(cov["meanFD"], errors="raise")
    cov["sex_num"] = cov["sex"].astype(str).str.lower().map(
        {"female": 0.0, "f": 0.0, "male": 1.0, "m": 1.0}
    )
    if cov["sex_num"].isna().any():
        raise RuntimeError("Unrecognized sex coding")
    cov["group_sz"] = (cov["group"] == "SZ").astype(int)
    cov["age_c"] = cov["age"] - cov["age"].mean()
    cov["sex_c"] = cov["sex_num"] - cov["sex_num"].mean()
    cov["meanFD_c"] = cov["meanFD"] - cov["meanFD"].mean()
    if len(cov) != 64 or cov["subject"].nunique() != 64:
        raise RuntimeError("Expected 64 unique subjects")
    if cov["group"].value_counts().to_dict() != {"SZ": 32, "HC": 32}:
        raise RuntimeError("Expected 32 SZ and 32 HC")
    return cov.reset_index(drop=True)


def prepare_metadata() -> pd.DataFrame:
    meta = pd.read_csv(ROI_META, sep="\t")
    required = {"roi_index0", "brainnetome_label", "subfamily", "stage"}
    if not required.issubset(meta.columns):
        raise RuntimeError(f"Missing ROI metadata columns: {sorted(required - set(meta.columns))}")
    meta["roi_index0"] = pd.to_numeric(meta["roi_index0"], errors="raise").astype(int)
    meta = meta.sort_values("roi_index0").reset_index(drop=True)
    if len(meta) != 65 or meta["roi_index0"].tolist() != list(range(65)):
        raise RuntimeError("ROI metadata is not exactly indices 0..64")
    if meta["stage"].value_counts().to_dict() != {
        "Anticipation": 43,
        "Outcome": 19,
        "Not_assigned": 3,
    }:
        raise RuntimeError("ROI stage mapping changed")
    if int((meta["subfamily"] == "Insula").sum()) != 12:
        raise RuntimeError("Expected exactly 12 Insula ROIs")
    assigned = meta.loc[meta["stage"] != "Not_assigned", "subfamily"].nunique()
    if assigned != 13:
        raise RuntimeError(f"Expected 13 assigned subfamilies, got {assigned}")
    return meta


def load_bin10_fingerprints(cov: pd.DataFrame) -> tuple[np.ndarray, dict, pd.DataFrame]:
    if sha256(CURRENT_BINS) != EXPECTED_CURRENT_BINS_SHA256:
        raise RuntimeError("Fixed current-bin input SHA256 changed")
    if sha256(MAPPING) != EXPECTED_MAPPING_SHA256:
        raise RuntimeError("2847-to-2608 mapping SHA256 changed")
    current_table = pd.read_csv(CURRENT_BINS, sep="\t")
    if "G1_bin_current" not in current_table.columns:
        raise RuntimeError("Current-bin table lacks G1_bin_current")
    full_bins = pd.to_numeric(current_table["G1_bin_current"], errors="raise").astype(int).to_numpy()
    mapping = np.asarray(np.load(MAPPING)).reshape(-1).astype(int)
    if len(full_bins) != 2847 or len(mapping) != 2608 or np.unique(mapping).size != 2608:
        raise RuntimeError("Voxel mapping dimensions changed")
    common_bins = full_bins[mapping]
    bin10_mask = common_bins == 10
    if int(bin10_mask.sum()) != 285:
        raise RuntimeError(f"Expected 285 current Bin-10 voxels, got {int(bin10_mask.sum())}")

    fingerprints = np.empty((len(cov), 65), dtype=float)
    hashes = []
    for i, subject in enumerate(cov["subject"]):
        path = FC_ROOT / f"{subject}_fingerprint_z_2608x65.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        array = np.asarray(np.load(path), dtype=float)
        if array.shape == (65, 2608):
            array = array.T
        if array.shape != (2608, 65) or not np.isfinite(array).all():
            raise RuntimeError(f"Bad FC array: {path} shape={array.shape}")
        fingerprints[i] = array[bin10_mask, :].mean(axis=0)
        hashes.append(
            {
                "input_type": "subject_fc_array",
                "subject": subject,
                "path": str(path),
                "sha256": sha256(path),
                "shape_or_rows": "2608x65",
            }
        )

    qc = {
        "full_current_bin_counts": {
            str(k): int(v) for k, v in pd.Series(full_bins).value_counts().sort_index().items()
        },
        "common_current_bin_counts": {
            str(k): int(v) for k, v in pd.Series(common_bins).value_counts().sort_index().items()
        },
        "bin10_voxels": int(bin10_mask.sum()),
        "fingerprint_shape": list(fingerprints.shape),
    }
    return fingerprints, qc, pd.DataFrame(hashes)


def contribution_decomposition(values: np.ndarray, template: np.ndarray) -> tuple[float, np.ndarray, dict]:
    values = np.asarray(values, dtype=float)
    template = np.asarray(template, dtype=float)
    if values.shape != template.shape or values.ndim != 1:
        raise ValueError("Pearson inputs must be matching 1D vectors")
    k = values.size
    if k < 3:
        raise ValueError("Need at least 3 ROIs")
    values_mean = float(values.mean())
    template_mean = float(template.mean())
    values_sd = float(values.std(ddof=1))
    template_sd = float(template.std(ddof=1))
    if values_sd <= 0 or template_sd <= 0:
        raise RuntimeError("Zero variance in Pearson input")
    z_values = (values - values_mean) / values_sd
    z_template = (template - template_mean) / template_sd
    contributions = z_values * z_template / float(k - 1)
    direct_r = float(np.corrcoef(values, template)[0, 1])
    error = float(abs(contributions.sum() - direct_r))
    if error > 1e-12:
        raise RuntimeError(f"Pearson contribution identity failed: {error}")
    stats = {
        "n_rois_profile": k,
        "subject_profile_mean": values_mean,
        "subject_profile_sd_sample": values_sd,
        "template_profile_mean": template_mean,
        "template_profile_sd_sample": template_sd,
        "identity_abs_error": error,
    }
    return direct_r, contributions, stats


def decompose_roi_set(
    roi_set: str,
    keep_indices: np.ndarray,
    fingerprints: np.ndarray,
    cov: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep_indices = np.asarray(keep_indices, dtype=int)
    if np.unique(keep_indices).size != keep_indices.size:
        raise ValueError("Duplicate ROI indices")
    selected = fingerprints[:, keep_indices]
    selected_meta = meta.iloc[keep_indices].reset_index(drop=True)
    hc_idx = np.flatnonzero(cov["group"].to_numpy() == "HC")
    hc_mean = selected[hc_idx].mean(axis=0)

    subject_rows = []
    roi_rows = []
    identity_rows = []
    for i, row in cov.iterrows():
        if row["group"] == "HC":
            template = selected[hc_idx[hc_idx != i]].mean(axis=0)
            reference = "HC_LOO31"
            reference_n = 31
        else:
            template = hc_mean
            reference = "HC32"
            reference_n = 32
        direct_r, contributions, stats = contribution_decomposition(selected[i], template)
        subject_rows.append(
            {
                "subject": row["subject"],
                "group": row["group"],
                "group_sz": row["group_sz"],
                "age": row["age"],
                "sex": row["sex"],
                "meanFD": row["meanFD"],
                "age_c": row["age_c"],
                "sex_c": row["sex_c"],
                "meanFD_c": row["meanFD_c"],
                "roi_set": roi_set,
                "n_rois_profile": len(keep_indices),
                "reference": reference,
                "reference_n": reference_n,
                "profile_r": direct_r,
                **stats,
            }
        )
        identity_rows.append(
            {
                "subject": row["subject"],
                "group": row["group"],
                "roi_set": roi_set,
                "n_rois": len(keep_indices),
                "profile_r": direct_r,
                "contribution_sum": float(contributions.sum()),
                "abs_error": stats["identity_abs_error"],
            }
        )
        for local_i, contribution in enumerate(contributions):
            meta_row = selected_meta.iloc[local_i]
            roi_rows.append(
                {
                    "subject": row["subject"],
                    "group": row["group"],
                    "group_sz": row["group_sz"],
                    "age_c": row["age_c"],
                    "sex_c": row["sex_c"],
                    "meanFD_c": row["meanFD_c"],
                    "roi_set": roi_set,
                    "n_rois_profile": len(keep_indices),
                    "reference": reference,
                    "profile_r": direct_r,
                    "roi_index0": int(meta_row["roi_index0"]),
                    "brainnetome_label": meta_row["brainnetome_label"],
                    "subfamily": meta_row["subfamily"],
                    "stage": meta_row["stage"],
                    "r_contribution": float(contribution),
                }
            )
    return pd.DataFrame(subject_rows), pd.DataFrame(roi_rows), pd.DataFrame(identity_rows)


def design_matrix(cov_frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(cov_frame), dtype=float),
            cov_frame["group_sz"].to_numpy(dtype=float),
            cov_frame["age_c"].to_numpy(dtype=float),
            cov_frame["sex_c"].to_numpy(dtype=float),
            cov_frame["meanFD_c"].to_numpy(dtype=float),
        ]
    )


def fit_subfamilies(
    roi_set: str,
    subject: pd.DataFrame,
    roi_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "subject",
        "group",
        "group_sz",
        "age_c",
        "sex_c",
        "meanFD_c",
        "roi_set",
        "subfamily",
        "stage",
    ]
    aggregate = (
        roi_long.groupby(keys, as_index=False, sort=True)
        .agg(
            contribution_sum=("r_contribution", "sum"),
            contribution_mean_per_roi=("r_contribution", "mean"),
            n_rois_subfamily=("roi_index0", "nunique"),
        )
        .sort_values(["subfamily", "subject"])
        .reset_index(drop=True)
    )
    subject_lookup = subject[["subject", "roi_set", "profile_r"]]
    aggregate = aggregate.merge(subject_lookup, on=["subject", "roi_set"], validate="many_to_one")

    total_x = design_matrix(subject)
    total_fit = fit_ols_hc3(subject["profile_r"].to_numpy(float), total_x)
    total_record = pd.DataFrame(
        [
            {
                "roi_set": roi_set,
                "n_subjects": len(subject),
                "n_rois_profile": int(subject["n_rois_profile"].iloc[0]),
                "beta_SZ_minus_HC_profile_r": total_fit.beta[1],
                "SE_HC3": total_fit.se[1],
                "CI95_low": total_fit.ci_low[1],
                "CI95_high": total_fit.ci_high[1],
                "p_raw": total_fit.p[1],
                "residual_df": total_fit.residual_df,
                "max_leverage": total_fit.max_leverage,
            }
        ]
    )

    records = []
    for (subfamily, stage), subset in aggregate.groupby(["subfamily", "stage"], sort=True):
        subset = subset.sort_values("subject").reset_index(drop=True)
        fit = fit_ols_hc3(subset["contribution_sum"].to_numpy(float), design_matrix(subset))
        beta = float(fit.beta[1])
        records.append(
            {
                "roi_set": roi_set,
                "subfamily": subfamily,
                "stage": stage,
                "N_ROIs": int(subset["n_rois_subfamily"].iloc[0]),
                "n_subjects": len(subset),
                "HC_mean_contribution": subset.loc[subset["group"] == "HC", "contribution_sum"].mean(),
                "SZ_mean_contribution": subset.loc[subset["group"] == "SZ", "contribution_sum"].mean(),
                "beta_SZ_minus_HC": beta,
                "beta_per_ROI": beta / int(subset["n_rois_subfamily"].iloc[0]),
                "SE_HC3": float(fit.se[1]),
                "CI95_low": float(fit.ci_low[1]),
                "CI95_high": float(fit.ci_high[1]),
                "p_raw": float(fit.p[1]),
                "q_FDR_13_prespecified": np.nan,
                "q_FDR_available_only_descriptive": np.nan,
                "FDR_family": "13 assigned subfamilies; Midbrain_pallidal excluded",
                "test_status": "TESTED",
                "residual_df": fit.residual_df,
                "max_leverage": fit.max_leverage,
            }
        )
    results = pd.DataFrame(records)
    assigned_mask = results["stage"] != "Not_assigned"
    assigned = results.loc[assigned_mask]
    if roi_set == FULL_SET:
        if len(assigned) != 13:
            raise RuntimeError("Full analysis did not yield 13 assigned subfamily tests")
        q = bh_adjust(assigned["p_raw"].to_numpy(float))
        results.loc[assigned.index, "q_FDR_13_prespecified"] = q
        results.loc[assigned.index, "q_FDR_available_only_descriptive"] = q
    elif roi_set == MINUS_SET:
        if len(assigned) != 12 or "Insula" in set(assigned["subfamily"]):
            raise RuntimeError("Minus-Insula analysis did not yield 12 testable assigned subfamilies")
        p_with_placeholder = np.concatenate([assigned["p_raw"].to_numpy(float), np.array([1.0])])
        q_with_placeholder = bh_adjust(p_with_placeholder)
        results.loc[assigned.index, "q_FDR_13_prespecified"] = q_with_placeholder[:-1]
        results.loc[assigned.index, "q_FDR_available_only_descriptive"] = bh_adjust(
            assigned["p_raw"].to_numpy(float)
        )
        results["FDR_family"] = np.where(
            assigned_mask,
            "Prespecified 13: 12 tested p-values plus removed Insula placeholder p=1; BH",
            "Not in FDR family",
        )
    else:
        raise ValueError(roi_set)

    total_beta = float(total_record["beta_SZ_minus_HC_profile_r"].iloc[0])
    results["signed_share_of_total_pct"] = 100.0 * results["beta_SZ_minus_HC"] / total_beta
    return results.sort_values("beta_SZ_minus_HC").reset_index(drop=True), total_record


def validate_full_reproduction(full_results: pd.DataFrame) -> pd.DataFrame:
    historical = pd.read_csv(HISTORICAL_RESULTS, sep="\t")
    merged = historical.merge(
        full_results,
        on=["subfamily", "stage"],
        how="outer",
        suffixes=("_historical", "_reproduced"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all() or len(merged) != 14:
        raise RuntimeError("Historical and reproduced subfamily keys differ")
    comparisons = {
        "N_ROIs": 0.0,
        "beta_SZ_minus_HC": 2e-12,
        "SE_HC3": 2e-12,
        "p_raw": 2e-12,
        "CI95_low": 3e-12,
        "CI95_high": 3e-12,
    }
    audit_rows = []
    for _, row in merged.iterrows():
        for field, tolerance in comparisons.items():
            h = float(row[f"{field}_historical"])
            r = float(row[f"{field}_reproduced"])
            error = abs(h - r)
            audit_rows.append(
                {
                    "subfamily": row["subfamily"],
                    "stage": row["stage"],
                    "field": field,
                    "historical": h,
                    "reproduced": r,
                    "absolute_error": error,
                    "tolerance": tolerance,
                    "pass": error <= tolerance,
                }
            )
    q_hist_col = "q_FDR_13_assigned"
    assigned_rows = merged[merged["stage"] != "Not_assigned"]
    for _, row in assigned_rows.iterrows():
        h = float(row[q_hist_col])
        r = float(row["q_FDR_13_prespecified"])
        error = abs(h - r)
        audit_rows.append(
            {
                "subfamily": row["subfamily"],
                "stage": row["stage"],
                "field": "q_FDR_13_assigned",
                "historical": h,
                "reproduced": r,
                "absolute_error": error,
                "tolerance": 2e-12,
                "pass": error <= 2e-12,
            }
        )
    audit = pd.DataFrame(audit_rows)
    if not audit["pass"].all():
        failed = audit.loc[~audit["pass"]]
        raise RuntimeError(f"Full historical table did not reproduce:\n{failed.to_string(index=False)}")

    for subfamily, beta_expected, q_expected in [
        ("SMA", EXPECTED_SMA_BETA, EXPECTED_SMA_Q),
        ("IFG", EXPECTED_IFG_BETA, EXPECTED_IFG_Q),
    ]:
        row = full_results.loc[full_results["subfamily"] == subfamily].iloc[0]
        if abs(float(row["beta_SZ_minus_HC"]) - beta_expected) > 2e-12:
            raise RuntimeError(f"{subfamily} beta reproduction gate failed")
        if abs(float(row["q_FDR_13_prespecified"]) - q_expected) > 2e-12:
            raise RuntimeError(f"{subfamily} q reproduction gate failed")
    return audit


def make_comparison(full: pd.DataFrame, minus: pd.DataFrame) -> pd.DataFrame:
    full2 = full.copy().add_prefix("full_").rename(
        columns={"full_subfamily": "subfamily", "full_stage": "stage"}
    )
    minus2 = minus.copy().add_prefix("minus_insula_").rename(
        columns={"minus_insula_subfamily": "subfamily", "minus_insula_stage": "stage"}
    )
    out = full2.merge(minus2, on=["subfamily", "stage"], how="left", validate="one_to_one")
    out["minus_insula_test_status"] = np.where(
        out["subfamily"] == "Insula", "UNAVAILABLE_ROI_REMOVED", out["minus_insula_test_status"]
    )
    out["minus_insula_fdr_placeholder_p"] = np.where(out["subfamily"] == "Insula", 1.0, np.nan)
    out["delta_beta_minus_insula_minus_full"] = (
        out["minus_insula_beta_SZ_minus_HC"] - out["full_beta_SZ_minus_HC"]
    )
    out["absolute_beta_change"] = out["delta_beta_minus_insula_minus_full"].abs()
    out["percent_change_in_absolute_beta"] = 100.0 * (
        out["minus_insula_beta_SZ_minus_HC"].abs() - out["full_beta_SZ_minus_HC"].abs()
    ) / out["full_beta_SZ_minus_HC"].abs()
    out["direction_preserved"] = np.where(
        out["minus_insula_beta_SZ_minus_HC"].isna(),
        "NOT_TESTABLE",
        np.where(
            np.sign(out["minus_insula_beta_SZ_minus_HC"]) == np.sign(out["full_beta_SZ_minus_HC"]),
            "YES",
            "NO",
        ),
    )
    out["full_FDR_lt_0_05"] = out["full_q_FDR_13_prespecified"] < ALPHA
    out["minus_insula_FDR13_lt_0_05"] = out["minus_insula_q_FDR_13_prespecified"] < ALPHA
    out["analysis_role"] = np.where(
        out["subfamily"] == "Insula",
        "Removed hypothesis retained as p=1 placeholder for the 13-item BH family",
        "Post hoc ROI-composition sensitivity analysis",
    )
    preferred = [
        "subfamily",
        "stage",
        "full_N_ROIs",
        "minus_insula_N_ROIs",
        "full_HC_mean_contribution",
        "full_SZ_mean_contribution",
        "minus_insula_HC_mean_contribution",
        "minus_insula_SZ_mean_contribution",
        "full_beta_SZ_minus_HC",
        "full_SE_HC3",
        "full_CI95_low",
        "full_CI95_high",
        "full_p_raw",
        "full_q_FDR_13_prespecified",
        "minus_insula_beta_SZ_minus_HC",
        "minus_insula_SE_HC3",
        "minus_insula_CI95_low",
        "minus_insula_CI95_high",
        "minus_insula_p_raw",
        "minus_insula_q_FDR_13_prespecified",
        "minus_insula_q_FDR_available_only_descriptive",
        "delta_beta_minus_insula_minus_full",
        "absolute_beta_change",
        "percent_change_in_absolute_beta",
        "direction_preserved",
        "full_FDR_lt_0_05",
        "minus_insula_FDR13_lt_0_05",
        "minus_insula_test_status",
        "minus_insula_fdr_placeholder_p",
        "analysis_role",
    ]
    return out[preferred]


def make_subject_level_contributions(
    full_subject: pd.DataFrame,
    full_aggregate: pd.DataFrame,
    minus_subject: pd.DataFrame,
    minus_aggregate: pd.DataFrame,
) -> pd.DataFrame:
    full_long = full_aggregate.merge(
        full_subject[
            [
                "subject",
                "roi_set",
                "reference",
                "reference_n",
                "n_rois_profile",
                "profile_r",
                "subject_profile_mean",
                "subject_profile_sd_sample",
                "template_profile_mean",
                "template_profile_sd_sample",
            ]
        ],
        on=["subject", "roi_set", "profile_r"],
        validate="many_to_one",
    )
    minus_long = minus_aggregate.merge(
        minus_subject[
            [
                "subject",
                "roi_set",
                "reference",
                "reference_n",
                "n_rois_profile",
                "profile_r",
                "subject_profile_mean",
                "subject_profile_sd_sample",
                "template_profile_mean",
                "template_profile_sd_sample",
            ]
        ],
        on=["subject", "roi_set", "profile_r"],
        validate="many_to_one",
    )
    return pd.concat([full_long, minus_long], ignore_index=True).sort_values(
        ["subfamily", "subject", "roi_set"]
    )


def subject_change_summary(subject_contributions: pd.DataFrame) -> pd.DataFrame:
    pivot = subject_contributions.pivot_table(
        index=["subject", "group", "subfamily", "stage"],
        columns="roi_set",
        values="contribution_sum",
        aggfunc="first",
    ).reset_index()
    pivot = pivot[pivot[MINUS_SET].notna()].copy()
    pivot["delta_minus_full"] = pivot[MINUS_SET] - pivot[FULL_SET]
    records = []
    for (subfamily, stage), subset in pivot.groupby(["subfamily", "stage"], sort=True):
        delta = subset["delta_minus_full"].to_numpy(float)
        test = ttest_1samp(delta, 0.0)
        hc = subset.loc[subset["group"] == "HC", "delta_minus_full"]
        sz = subset.loc[subset["group"] == "SZ", "delta_minus_full"]
        records.append(
            {
                "subfamily": subfamily,
                "stage": stage,
                "n_subjects": len(delta),
                "mean_delta_minus_full": float(delta.mean()),
                "sd_delta_minus_full": float(delta.std(ddof=1)),
                "median_delta_minus_full": float(np.median(delta)),
                "min_delta_minus_full": float(delta.min()),
                "max_delta_minus_full": float(delta.max()),
                "proportion_delta_positive": float(np.mean(delta > 0)),
                "proportion_delta_negative": float(np.mean(delta < 0)),
                "HC_mean_delta": float(hc.mean()),
                "SZ_mean_delta": float(sz.mean()),
                "paired_t_p_descriptive": float(test.pvalue),
                "inference_note": "Descriptive paired change; not part of the historical confirmatory family",
            }
        )
    return pd.DataFrame(records).sort_values("mean_delta_minus_full").reset_index(drop=True)


def input_provenance(fc_hashes: pd.DataFrame) -> pd.DataFrame:
    core = []
    for kind, path, shape in [
        ("covariates", COV_INPUT, "64 rows"),
        ("roi_metadata", ROI_META, "65 rows"),
        ("voxel_mapping", MAPPING, "2608 indices"),
        ("fixed_current_bins", CURRENT_BINS, "2847 rows"),
        ("historical_results", HISTORICAL_RESULTS, "14 subfamilies"),
        ("historical_script", HISTORICAL_SCRIPT, "embedded Python workflow"),
    ]:
        core.append(
            {
                "input_type": kind,
                "subject": "",
                "path": str(path),
                "sha256": sha256(path),
                "shape_or_rows": shape,
            }
        )
    return pd.concat([pd.DataFrame(core), fc_hashes], ignore_index=True)


def write_blocking_report(message: str) -> None:
    content = f"""# STEP 3 阻塞报告\n\n状态：**STOPPED_BEFORE_INSULA_SENSITIVITY**\n\n时间：{datetime.now(timezone.utc).isoformat()}\n\n原因：原始 65-ROI 历史分析未通过复现门槛，因此按照预注册式要求未执行删除 Insula 后的推断。\n\n错误：\n\n```text\n{message}\n```\n"""
    (OUT / "BLOCKING_REPORT.md").write_text(content, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("STEP 3 - SMA/IFG Local Contribution Sensitivity After Insula Removal")
    print(f"Start UTC: {datetime.now(timezone.utc).isoformat()}")
    print("Phase 1: reproduce historical 65-ROI analysis from frozen arrays")

    cov = prepare_covariates()
    meta = prepare_metadata()
    fingerprints, bin_qc, fc_hashes = load_bin10_fingerprints(cov)

    full_indices = np.arange(65, dtype=int)
    minus_indices = meta.loc[meta["subfamily"] != "Insula", "roi_index0"].to_numpy(int)
    if len(minus_indices) != 53:
        raise RuntimeError(f"Expected 53 non-Insula ROIs, got {len(minus_indices)}")

    full_subject, full_roi, full_identity = decompose_roi_set(
        FULL_SET, full_indices, fingerprints, cov, meta
    )
    full_results, full_total = fit_subfamilies(FULL_SET, full_subject, full_roi)
    full_audit = validate_full_reproduction(full_results)
    print("Historical reproduction gate: PASS")
    print(
        full_results.loc[
            full_results["subfamily"].isin(["SMA", "IFG"]),
            ["subfamily", "beta_SZ_minus_HC", "SE_HC3", "p_raw", "q_FDR_13_prespecified"],
        ].to_string(index=False)
    )

    print("Phase 2: recompute Pearson normalization after removing all 12 Insula ROIs")
    minus_subject, minus_roi, minus_identity = decompose_roi_set(
        MINUS_SET, minus_indices, fingerprints, cov, meta
    )
    minus_results, minus_total = fit_subfamilies(MINUS_SET, minus_subject, minus_roi)

    full_aggregate = (
        full_roi.groupby(
            [
                "subject",
                "group",
                "group_sz",
                "age_c",
                "sex_c",
                "meanFD_c",
                "roi_set",
                "subfamily",
                "stage",
            ],
            as_index=False,
        )
        .agg(
            contribution_sum=("r_contribution", "sum"),
            contribution_mean_per_roi=("r_contribution", "mean"),
            n_rois_subfamily=("roi_index0", "nunique"),
        )
        .merge(full_subject[["subject", "roi_set", "profile_r"]], on=["subject", "roi_set"], validate="many_to_one")
    )
    minus_aggregate = (
        minus_roi.groupby(
            [
                "subject",
                "group",
                "group_sz",
                "age_c",
                "sex_c",
                "meanFD_c",
                "roi_set",
                "subfamily",
                "stage",
            ],
            as_index=False,
        )
        .agg(
            contribution_sum=("r_contribution", "sum"),
            contribution_mean_per_roi=("r_contribution", "mean"),
            n_rois_subfamily=("roi_index0", "nunique"),
        )
        .merge(minus_subject[["subject", "roi_set", "profile_r"]], on=["subject", "roi_set"], validate="many_to_one")
    )

    comparison = make_comparison(full_results, minus_results)
    subject_contributions = make_subject_level_contributions(
        full_subject, full_aggregate, minus_subject, minus_aggregate
    )
    change_summary = subject_change_summary(subject_contributions)
    identity = pd.concat([full_identity, minus_identity], ignore_index=True)
    total_effects = pd.concat([full_total, minus_total], ignore_index=True)
    provenance = input_provenance(fc_hashes)

    max_identity_full = float(full_identity["abs_error"].max())
    max_identity_minus = float(minus_identity["abs_error"].max())
    if max_identity_full > 1e-12 or max_identity_minus > 1e-12:
        raise RuntimeError("Contribution identity QC failed")

    full_sum = float(full_results["beta_SZ_minus_HC"].sum())
    minus_sum = float(minus_results["beta_SZ_minus_HC"].sum())
    full_total_beta = float(full_total["beta_SZ_minus_HC_profile_r"].iloc[0])
    minus_total_beta = float(minus_total["beta_SZ_minus_HC_profile_r"].iloc[0])
    if abs(full_sum - full_total_beta) > 2e-12 or abs(minus_sum - minus_total_beta) > 2e-12:
        raise RuntimeError("Subfamily beta sum did not equal whole-profile beta")

    write_tsv(comparison, OUT / "local_effects_full_vs_minus_insula.tsv")
    write_tsv(subject_contributions, OUT / "subject_level_contributions.tsv")
    write_tsv(change_summary, OUT / "subject_change_summary.tsv")
    write_tsv(identity, OUT / "pearson_contribution_identity_qc.tsv")
    write_tsv(full_audit, OUT / "historical_full_reproduction_audit.tsv")
    write_tsv(full_results, OUT / "full_65roi_subfamily_results.tsv")
    write_tsv(minus_results, OUT / "minus_insula_53roi_subfamily_results.tsv")
    write_tsv(total_effects, OUT / "whole_profile_r_effects.tsv")
    write_tsv(provenance, OUT / "input_provenance_sha256.tsv")

    subject_profiles = full_subject.merge(
        minus_subject,
        on=["subject", "group", "group_sz", "age", "sex", "meanFD", "age_c", "sex_c", "meanFD_c"],
        suffixes=("_full", "_minus_insula"),
        validate="one_to_one",
    )
    subject_profiles["delta_profile_r_minus_insula_minus_full"] = (
        subject_profiles["profile_r_minus_insula"] - subject_profiles["profile_r_full"]
    )
    write_tsv(subject_profiles, OUT / "subject_level_profile_r.tsv")

    qc = {
        "status": "SUCCESS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "historical_reproduction_gate": "PASS",
        "source_project_modified": False,
        "sample": {
            "n": int(len(cov)),
            "group_counts": {k: int(v) for k, v in cov["group"].value_counts().items()},
        },
        "bin10": bin_qc,
        "roi_sets": {
            FULL_SET: {"n_rois": 65, "pearson_denominator": 64},
            MINUS_SET: {"n_rois": 53, "pearson_denominator": 52, "removed_subfamily": "Insula", "removed_rois": 12},
        },
        "pearson_identity_max_abs_error": {
            FULL_SET: max_identity_full,
            MINUS_SET: max_identity_minus,
        },
        "beta_additivity": {
            FULL_SET: {
                "whole_profile_beta": full_total_beta,
                "sum_subfamily_betas": full_sum,
                "abs_error": abs(full_sum - full_total_beta),
            },
            MINUS_SET: {
                "whole_profile_beta": minus_total_beta,
                "sum_subfamily_betas": minus_sum,
                "abs_error": abs(minus_sum - minus_total_beta),
            },
        },
        "fdr": {
            "full": "BH across 13 assigned subfamilies",
            "minus_insula_primary": "BH across prespecified family size 13 using 12 observed p-values plus Insula placeholder p=1",
            "minus_insula_descriptive": "BH across 12 available tests, reported separately and not used for direct q comparison",
            "excluded_from_family": "Midbrain_pallidal (Not_assigned)",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    write_json(qc, OUT / "analysis_qc.json")

    print("Contribution identity QC: PASS")
    print(f"  Full max |sum(c)-r|  = {max_identity_full:.3e}")
    print(f"  Minus max |sum(c)-r| = {max_identity_minus:.3e}")
    print("Post-removal target results:")
    print(
        comparison.loc[
            comparison["subfamily"].isin(["SMA", "IFG"]),
            [
                "subfamily",
                "minus_insula_beta_SZ_minus_HC",
                "minus_insula_SE_HC3",
                "minus_insula_CI95_low",
                "minus_insula_CI95_high",
                "minus_insula_p_raw",
                "minus_insula_q_FDR_13_prespecified",
                "delta_beta_minus_insula_minus_full",
                "direction_preserved",
            ],
        ].to_string(index=False)
    )
    print(f"Completed UTC: {datetime.now(timezone.utc).isoformat()}")
    (OUT / "BLOCKING_REPORT.md").unlink(missing_ok=True)
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
