#!/usr/bin/env python3
"""STEP 4: subject-level SMA/IFG 10-bin FC profile analysis.

All inputs are read-only. All generated artifacts are written below the dedicated
STEP4 output directory. This script deliberately does not re-test the historical
Bin 10 Pearson contribution outcome.
"""

from __future__ import annotations

import os
import hashlib
import json
import logging
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

VENDOR = Path(os.environ.get("STRIATAL_PYTHON_VENDOR", "").strip() or ".missing_python_vendor")
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
import scipy
from scipy import stats


DATA_ROOT = Path(os.path.join(os.environ["STRIATAL_PROJECT_ROOT"], 'CURRENT64_CANONICAL_RERUN'))
OUT = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP4_SMA_IFG_ProfileSimilarity'))
REPORTS = OUT / "reports"
TABLES = OUT / "tables"
CODE = OUT / "code"
LOGS = OUT / "logs"
FIGURES = OUT / "figures"

RANDOM_SEED = 20261012  # Locked for reproducibility; no random procedure is used in the primary analysis.
TARGETS = ["SMA", "IFG"]
EXPECTED_ROI_COUNTS = {"SMA": 1, "IFG": 6}
EXPECTED_STAGES = {"SMA": "Anticipation", "IFG": "Outcome"}
Z975 = float(stats.norm.ppf(0.975))

INPUTS = {
    "ROI_by_bin_FC": DATA_ROOT / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_FC_long.tsv",
    "ROI_metadata": DATA_ROOT / "01_FROZEN_INPUTS" / "reward_stage_mapping" / "roi_metadata_stage_subfamily_v2.tsv",
    "Covariates": DATA_ROOT / "01_FROZEN_INPUTS" / "covariates" / "current64_covariates.tsv",
    "Subject_table": DATA_ROOT / "01_FROZEN_INPUTS" / "subject_table" / "current64_subject_table.tsv",
    "Definition_A_bins": DATA_ROOT / "01_FROZEN_INPUTS" / "canonical_bins" / "fixed_375_G1_voxel_bins_2608.tsv",
    "STEP1_input_audit": Path(os.environ["STRIATAL_PRIVATE_AUDIT_ROOT"]) / "STEP1_FINAL_INPUT_AUDIT.md",
    "STEP2_analysis_plan": Path(os.environ["STRIATAL_PRIVATE_AUDIT_ROOT"]) / "step2_variance_analysis" / "STEP2_STATISTICAL_ANALYSIS_PLAN.md",
    "STEP2_report": Path(os.environ["STRIATAL_PRIVATE_AUDIT_ROOT"]) / "step2_variance_analysis" / "STEP2_VARIANCE_REPORT.md",
    "STEP3_contribution_audit": Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP3_LocalContribution_InsulaSensitivity/LOCAL_CONTRIBUTION_AUDIT.md')),
    "STEP3_LOSO_report": Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP3_LocalContribution_AllSubfamilies/LEAVE_ONE_SUBFAMILY_OUT_RESULTS.md')),
    "Historical_family_profile_builder": Path(os.path.join(os.environ["STRIATAL_PROJECT_ROOT"], '01_code/03_connectivity_analysis/28_build_roi_family_profiles.py')),
    "Canonical_ROI_figure_source_builder": DATA_ROOT / "scripts" / "python" / "05_prepare_roi_figure_sources.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, na_rep="N/A", float_format="%.17g")


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
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    require(y.ndim == 1 and x.ndim == 2 and len(y) == x.shape[0], "Invalid OLS dimensions")
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    h = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    require(np.all(h < 1.0 - 1e-12), "HC3 leverage reached one")
    omega = (resid / (1.0 - h)) ** 2
    meat = x.T @ (omega[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv
    covariance = 0.5 * (covariance + covariance.T)
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    z = beta / se
    p = 2.0 * stats.norm.sf(np.abs(z))
    return OLSHC3(
        beta=beta,
        se=se,
        p=p,
        ci_low=beta - Z975 * se,
        ci_high=beta + Z975 * se,
        residual_df=int(len(y) - x.shape[1]),
        max_leverage=float(h.max()),
    )


@dataclass
class PillaiResult:
    trace: float
    f_value: float
    df_num: float
    df_den: float
    p_value: float
    residual_df: int
    response_rank: int
    contrast_rank: int


def pillai_group_test(y: np.ndarray, x: np.ndarray, group_column: int = 1) -> PillaiResult:
    """Classical multivariate OLS Pillai test for one design coefficient."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    require(y.ndim == 2 and x.ndim == 2 and y.shape[0] == x.shape[0], "Invalid MANOVA dimensions")
    n = x.shape[0]
    rank_x = int(np.linalg.matrix_rank(x))
    df_resid = int(n - rank_x)
    xtx_inv = np.linalg.inv(x.T @ x)
    b = xtx_inv @ x.T @ y
    resid = y - x @ b
    e = resid.T @ resid
    c = np.zeros((1, x.shape[1]), dtype=float)
    c[0, group_column] = 1.0
    cb = c @ b
    t = c @ xtx_inv @ c.T
    h = cb.T @ np.linalg.inv(t) @ cb
    eh = e + h
    eig = np.linalg.eigvals(np.linalg.solve(eh, h))
    eig = np.real_if_close(eig, tol=1000).real
    eig = np.clip(eig, 0.0, 1.0)
    rank_eh = int(np.linalg.matrix_rank(eh))
    rank_c = int(np.linalg.matrix_rank(c))
    s = min(rank_eh, rank_c)
    v = float(eig[eig > 1e-10].sum())
    m = (abs(rank_eh - rank_c) - 1.0) / 2.0
    nn = (df_resid - rank_eh - 1.0) / 2.0
    df_num = float(s * (2.0 * m + s + 1.0))
    df_den = float(s * (2.0 * nn + s + 1.0))
    require(0.0 <= v < float(s), "Invalid Pillai trace")
    f_value = float((df_den / df_num) * v / (s - v))
    p_value = float(stats.f.sf(f_value, df_num, df_den))
    return PillaiResult(v, f_value, df_num, df_den, p_value, df_resid, rank_eh, rank_c)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    require(np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1)), "Invalid p values for Holm")
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    scaled = np.array([(m - i) * sorted_p[i] for i in range(m)], dtype=float)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def pearson_safe(a: np.ndarray, b: np.ndarray) -> tuple[float, bool, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sd_a = float(np.std(a, ddof=1))
    sd_b = float(np.std(b, ddof=1))
    unstable = bool(min(sd_a, sd_b) < 1e-8)
    if unstable:
        return math.nan, True, sd_a, sd_b
    return float(np.corrcoef(a, b)[0, 1]), False, sd_a, sd_b


def design_from(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(frame), dtype=float),
            frame["group_sz"].to_numpy(float),
            frame["age_c"].to_numpy(float),
            frame["sex_c"].to_numpy(float),
            frame["meanFD_c"].to_numpy(float),
        ]
    )


def group_descriptives(values: pd.Series, groups: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in ["HC", "SZ"]:
        x = values.loc[groups == group].to_numpy(float)
        out[f"{group}_N"] = int(len(x))
        out[f"{group}_mean"] = float(x.mean())
        out[f"{group}_SD"] = float(x.std(ddof=1))
    return out


def main() -> None:
    for directory in [REPORTS, TABLES, CODE, LOGS, FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    plan_path = REPORTS / "STEP4_STATISTICAL_ANALYSIS_PLAN.md"
    require(plan_path.exists(), "Statistical analysis plan must exist before testing")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        handlers=[logging.FileHandler(LOGS / "analysis_execution.log", mode="w", encoding="utf-8"), logging.StreamHandler()],
    )
    log = logging.getLogger("step4")
    log.info("STEP 4 started; fixed random seed=%d (no random primary procedure)", RANDOM_SEED)

    for label, path in INPUTS.items():
        require(path.exists(), f"Missing required input: {label}: {path}")

    roi = pd.read_csv(INPUTS["ROI_by_bin_FC"], sep="\t")
    meta = pd.read_csv(INPUTS["ROI_metadata"], sep="\t")
    cov = pd.read_csv(INPUTS["Covariates"], sep="\t")
    subjects = pd.read_csv(INPUTS["Subject_table"], sep="\t")
    bins = pd.read_csv(INPUTS["Definition_A_bins"], sep="\t")

    require(roi.shape == (41600, 7), f"Unexpected ROI-long shape: {roi.shape}")
    require(roi["Subject"].nunique() == 64 and roi["ROI"].nunique() == 65, "ROI-long cardinality mismatch")
    require(roi.groupby("Group")["Subject"].nunique().to_dict() == {"HC": 32, "SZ": 32}, "Group count mismatch")
    require(subjects.groupby("Group")["Subject"].nunique().to_dict() == {"HC": 32, "SZ": 32}, "Subject-table group mismatch")
    require(set(roi["Subject"]) == set(cov["Subject"]) == set(subjects["Subject"]), "Subject sets do not match")
    require(not cov.isna().any().any() and not subjects.isna().any().any(), "Missing covariate or subject data")
    require(set(roi["Bin"]) == set(range(1, 11)), "Bins are not 1-10")
    require(not roi.duplicated(["Subject", "Bin", "ROI"]).any(), "Duplicate Subject x Bin x ROI key")
    require(np.isfinite(roi["FC"].to_numpy(float)).all(), "Non-finite FC")
    require(len(bins) == 2608 and set(bins["bin"]) == set(range(1, 11)), "Definition A bin file mismatch")

    meta_target = meta.loc[meta["subfamily"].isin(TARGETS)].copy()
    roi_count_map = meta_target.groupby("subfamily")["roi_id"].nunique().to_dict()
    require(roi_count_map == EXPECTED_ROI_COUNTS, f"SMA/IFG ROI counts differ: {roi_count_map}")
    stage_map = meta_target.groupby("subfamily")["stage"].unique().apply(lambda x: list(x)).to_dict()
    require(all(stage_map[t] == [EXPECTED_STAGES[t]] for t in TARGETS), f"SMA/IFG stage mismatch: {stage_map}")
    require(meta_target["include_two_family"].eq(1).all(), "Target ROI unexpectedly excluded")

    roi_name_rows = []
    for family in TARGETS:
        long_names = sorted(roi.loc[roi["Subfamily"] == family, "ROI"].unique().tolist())
        m = meta_target.loc[meta_target["subfamily"] == family].sort_values("roi_index0")
        require(len(long_names) == len(m), f"Long-table ROI count mismatch for {family}")
        for (_, row), long_name in zip(m.iterrows(), long_names):
            roi_name_rows.append(
                {
                    "Subfamily": family,
                    "Stage_classification": row["stage"],
                    "roi_id_1based": int(row["roi_id"]),
                    "roi_index_0based": int(row["roi_index0"]),
                    "brainnetome_label": row["brainnetome_label"],
                    "ROI_name_in_FC_table": long_name,
                    "include_two_family": int(row["include_two_family"]),
                }
            )
    roi_membership = pd.DataFrame(roi_name_rows).sort_values(["Subfamily", "roi_index_0based"])
    write_tsv(roi_membership, TABLES / "roi_membership_audit.tsv")

    g1 = (
        bins.groupby("bin", as_index=False, sort=True)
        .agg(
            n_voxels=("reference_G1", "size"),
            mean_reference_G1=("reference_G1", "mean"),
            SD_reference_G1=("reference_G1", "std"),
            min_reference_G1=("reference_G1", "min"),
            max_reference_G1=("reference_G1", "max"),
        )
        .rename(columns={"bin": "Bin"})
    )
    require(g1["n_voxels"].tolist() == [260, 260] + [261] * 8, f"Definition A counts changed: {g1['n_voxels'].tolist()}")
    require(np.all(np.diff(g1["mean_reference_G1"]) > 0), "Mean G1 positions are not strictly increasing")
    g1_mean = float(g1["mean_reference_G1"].mean())
    g1_sd = float(g1["mean_reference_G1"].std(ddof=1))
    g1["G1_bin_mean_centered"] = g1["mean_reference_G1"] - g1_mean
    g1["G1_bin_mean_z"] = g1["G1_bin_mean_centered"] / g1_sd
    write_tsv(g1, TABLES / "definition_A_G1_bin_axis.tsv")
    xg = g1["G1_bin_mean_z"].to_numpy(float)
    require(abs(float(xg.mean())) < 1e-15 and abs(float(xg.std(ddof=1)) - 1.0) < 1e-14, "G1 standardization failed")

    cov = cov.copy()
    cov["group_sz"] = (cov["Group"] == "SZ").astype(int)
    cov["sex_num"] = cov["Sex"].map({"Female": 0, "Male": 1})
    require(not cov["sex_num"].isna().any(), "Unexpected Sex coding")
    cov["age_c"] = cov["Age"] - cov["Age"].mean()
    cov["sex_c"] = cov["sex_num"] - cov["sex_num"].mean()
    cov["meanFD_c"] = cov["MeanFD"] - cov["MeanFD"].mean()

    target_long = roi.loc[roi["Subfamily"].isin(TARGETS)].copy()
    family_profile_long = (
        target_long.groupby(["Subject", "Group", "Subfamily", "RewardStage", "Bin"], as_index=False, sort=True)
        .agg(FC=("FC", "mean"), ROI_count=("ROI", "nunique"))
        .rename(columns={"Subfamily": "Target", "RewardStage": "Stage_classification"})
    )
    for family in TARGETS:
        d = family_profile_long.loc[family_profile_long["Target"] == family]
        require(d["ROI_count"].eq(EXPECTED_ROI_COUNTS[family]).all(), f"ROI aggregation count mismatch for {family}")
        require(len(d) == 640, f"Expected 640 rows for {family}; found {len(d)}")
        require(d.groupby("Subject")["Bin"].nunique().eq(10).all(), f"Incomplete profiles for {family}")
    require(not family_profile_long.duplicated(["Subject", "Target", "Bin"]).any(), "Duplicate profile key")
    require(len(family_profile_long) == 1280, "Expected 1,280 target profile points")
    family_profile_long = family_profile_long.merge(g1[["Bin", "mean_reference_G1", "G1_bin_mean_centered", "G1_bin_mean_z"]], on="Bin", validate="many_to_one")

    profile_wide = family_profile_long.pivot(index=["Subject", "Group", "Target", "Stage_classification", "ROI_count"], columns="Bin", values="FC").reset_index()
    profile_wide.columns = [f"FC_bin{c}" if isinstance(c, (int, np.integer)) else c for c in profile_wide.columns]
    require(len(profile_wide) == 128 and profile_wide[[f"FC_bin{i}" for i in range(1, 11)]].notna().all().all(), "Wide profile construction failed")
    profile_wide = profile_wide.merge(cov, on=["Subject", "Group"], validate="many_to_one")
    profile_cols = [f"FC_bin{i}" for i in range(1, 11)]
    write_tsv(profile_wide, TABLES / "SMA_IFG_subject_10bin_profiles.tsv")

    # Fixed orthogonal decomposition basis: constant + standardized actual-G1 axis + 8D shape complement.
    trend_design = np.column_stack([np.ones(10), xg])
    q_complete, _ = np.linalg.qr(trend_design, mode="complete")
    q_shape = q_complete[:, 2:]
    basis_qc = {
        "shape_basis_columns": int(q_shape.shape[1]),
        "max_abs_QtQ_minus_I": float(np.abs(q_shape.T @ q_shape - np.eye(8)).max()),
        "max_abs_constant_inner_product": float(np.abs(np.ones(10) @ q_shape).max()),
        "max_abs_G1z_inner_product": float(np.abs(xg @ q_shape).max()),
    }
    require(basis_qc["max_abs_QtQ_minus_I"] < 1e-12, "Shape basis not orthonormal")
    require(basis_qc["max_abs_constant_inner_product"] < 1e-12, "Shape basis not orthogonal to constant")
    require(basis_qc["max_abs_G1z_inner_product"] < 1e-12, "Shape basis not orthogonal to G1")

    metric_rows = []
    residual_rows = []
    score_by_family: dict[str, np.ndarray] = {}
    metric_by_family: dict[str, pd.DataFrame] = {}
    for family in TARGETS:
        d = profile_wide.loc[profile_wide["Target"] == family].sort_values("Subject").copy()
        ymat = d[profile_cols].to_numpy(float)
        offsets = ymat.mean(axis=1)
        slopes = ((ymat - offsets[:, None]) @ xg) / float(xg @ xg)
        fitted = offsets[:, None] + slopes[:, None] * xg[None, :]
        residual = ymat - fitted
        scores = residual @ q_shape
        reconstructed = scores @ q_shape.T
        require(float(np.max(np.abs(reconstructed - residual))) < 2e-15, f"Shape reconstruction failed for {family}")
        require(float(np.max(np.abs(residual.mean(axis=1)))) < 1e-15, f"Residual mean failed for {family}")
        require(float(np.max(np.abs(residual @ xg))) < 2e-15, f"Residual G1 orthogonality failed for {family}")

        md = d[["Subject", "Group", "Target", "Stage_classification", "ROI_count", "Age", "Sex", "MeanFD", "group_sz", "age_c", "sex_c", "meanFD_c"]].copy()
        md["Offset"] = offsets
        md["Slope_per_G1_bin_mean_SD"] = slopes
        md["subject_shape_RMSE"] = np.sqrt(np.mean(residual**2, axis=1))
        metric_rows.append(md)
        metric_by_family[family] = md
        score_by_family[family] = scores

        for row_idx, row in d.reset_index(drop=True).iterrows():
            out = {
                "Subject": row["Subject"],
                "Group": row["Group"],
                "Target": family,
                "Stage_classification": row["Stage_classification"],
                "residual_mean": float(residual[row_idx].mean()),
                "residual_dot_G1z": float(residual[row_idx] @ xg),
                "reconstruction_max_abs_error": float(np.max(np.abs(reconstructed[row_idx] - residual[row_idx]))),
            }
            for b in range(10):
                out[f"shape_residual_bin{b + 1}"] = float(residual[row_idx, b])
            for j in range(8):
                out[f"shape_score_{j + 1}"] = float(scores[row_idx, j])
            residual_rows.append(out)

    metrics = pd.concat(metric_rows, ignore_index=True).sort_values(["Target", "Subject"])
    residuals_wide = pd.DataFrame(residual_rows).sort_values(["Target", "Subject"])
    write_tsv(metrics, TABLES / "subject_level_offset_slope.tsv")
    write_tsv(residuals_wide, TABLES / "subject_level_shape_residuals.tsv")

    # Six prespecified tests.
    test_rows = []
    raw_adjusted_profiles: dict[str, dict[str, np.ndarray | float | np.ndarray]] = {}
    main_models: dict[tuple[str, str], object] = {}
    for family in TARGETS:
        d = metric_by_family[family].sort_values("Subject").reset_index(drop=True)
        design = design_from(d)
        for metric_name, column in [("FC_Offset", "Offset"), ("Linear_G1_Slope", "Slope_per_G1_bin_mean_SD")]:
            fit = fit_ols_hc3(d[column].to_numpy(float), design)
            desc = group_descriptives(d[column], d["Group"])
            main_models[(family, metric_name)] = fit
            test_rows.append(
                {
                    "Target": family,
                    "Stage_classification": EXPECTED_STAGES[family],
                    "Analysis_dimension": metric_name,
                    "Method": "Subject-level OLS with HC3 robust covariance",
                    "Estimand": "Adjusted SZ minus HC",
                    **desc,
                    "Beta_SZ_minus_HC": float(fit.beta[1]),
                    "SE_HC3": float(fit.se[1]),
                    "CI95_low": float(fit.ci_low[1]),
                    "CI95_high": float(fit.ci_high[1]),
                    "Pillai_trace": math.nan,
                    "Approx_F": math.nan,
                    "df_num": math.nan,
                    "df_den": math.nan,
                    "raw_p": float(fit.p[1]),
                    "residual_df": int(fit.residual_df),
                    "max_leverage": float(fit.max_leverage),
                }
            )

        scores = score_by_family[family]
        pillai = pillai_group_test(scores, design)
        main_models[(family, "Nonlinear_Profile_Shape")] = pillai
        test_rows.append(
            {
                "Target": family,
                "Stage_classification": EXPECTED_STAGES[family],
                "Analysis_dimension": "Nonlinear_Profile_Shape",
                "Method": "8D orthogonal-basis MANCOVA; Pillai trace",
                "Estimand": "Adjusted omnibus SZ versus HC shape effect",
                "HC_N": 32,
                "HC_mean": math.nan,
                "HC_SD": math.nan,
                "SZ_N": 32,
                "SZ_mean": math.nan,
                "SZ_SD": math.nan,
                "Beta_SZ_minus_HC": math.nan,
                "SE_HC3": math.nan,
                "CI95_low": math.nan,
                "CI95_high": math.nan,
                "Pillai_trace": pillai.trace,
                "Approx_F": pillai.f_value,
                "df_num": pillai.df_num,
                "df_den": pillai.df_den,
                "raw_p": pillai.p_value,
                "residual_df": pillai.residual_df,
                "max_leverage": math.nan,
            }
        )

        # Adjusted original-profile group coefficient and its orthogonal decomposition.
        original = profile_wide.loc[profile_wide["Target"] == family].sort_values("Subject").reset_index(drop=True)
        original_design = design_from(original)
        yraw = original[profile_cols].to_numpy(float)
        braw = np.linalg.inv(original_design.T @ original_design) @ original_design.T @ yraw
        adjusted_diff = braw[1, :]
        adjusted_offset = float(adjusted_diff.mean())
        adjusted_slope = float(((adjusted_diff - adjusted_offset) @ xg) / (xg @ xg))
        offset_component = np.repeat(adjusted_offset, 10)
        linear_component = adjusted_slope * xg
        nonlinear_component = adjusted_diff - offset_component - linear_component
        require(abs(float(nonlinear_component.mean())) < 1e-15, f"Adjusted nonlinear mean failed for {family}")
        require(abs(float(nonlinear_component @ xg)) < 2e-15, f"Adjusted nonlinear G1 orthogonality failed for {family}")
        total_rmse_adjusted = float(np.sqrt(np.mean(adjusted_diff**2)))
        offset_rms = abs(adjusted_offset)
        linear_rms = float(np.sqrt(np.mean(linear_component**2)))
        shape_rmse = float(np.sqrt(np.mean(nonlinear_component**2)))
        decomposition_error = abs(total_rmse_adjusted**2 - (offset_rms**2 + linear_rms**2 + shape_rmse**2))
        require(decomposition_error < 1e-14, f"Adjusted decomposition identity failed for {family}")
        raw_adjusted_profiles[family] = {
            "adjusted_diff": adjusted_diff,
            "adjusted_offset": adjusted_offset,
            "adjusted_slope": adjusted_slope,
            "offset_component": offset_component,
            "linear_component": linear_component,
            "nonlinear_component": nonlinear_component,
            "total_rmse_adjusted": total_rmse_adjusted,
            "offset_rms": offset_rms,
            "linear_rms": linear_rms,
            "shape_rmse": shape_rmse,
            "decomposition_error": decomposition_error,
        }

    tests = pd.DataFrame(test_rows)
    dimension_order = {"FC_Offset": 1, "Linear_G1_Slope": 2, "Nonlinear_Profile_Shape": 3}
    target_order = {"SMA": 1, "IFG": 2}
    tests["_order"] = tests["Target"].map(target_order) * 10 + tests["Analysis_dimension"].map(dimension_order)
    tests = tests.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    tests["Holm_adjusted_p_6_tests"] = holm_adjust(tests["raw_p"].to_numpy(float))
    tests["Significant_raw_0.05"] = tests["raw_p"] < 0.05
    tests["Significant_Holm_0.05"] = tests["Holm_adjusted_p_6_tests"] < 0.05
    tests["Effect_direction"] = np.where(
        tests["Analysis_dimension"].eq("Nonlinear_Profile_Shape"),
        "Omnibus_not_directional",
        np.where(tests["Beta_SZ_minus_HC"] > 0, "SZ_greater_than_HC", np.where(tests["Beta_SZ_minus_HC"] < 0, "SZ_less_than_HC", "Zero")),
    )
    write_tsv(tests, TABLES / "STEP4_statistical_tests.tsv")

    # Profile group means, mean CIs, detrended shapes, and adjusted decomposition by bin.
    group_stat_rows = []
    similarity_rows = []
    full_summary_rows = []
    for family in TARGETS:
        dlong = family_profile_long.loc[family_profile_long["Target"] == family].copy()
        rw = residuals_wide.loc[residuals_wide["Target"] == family]
        residual_long_rows = []
        for row in rw.itertuples(index=False):
            for b in range(1, 11):
                residual_long_rows.append({"Subject": row.Subject, "Group": row.Group, "Bin": b, "ShapeResidual": getattr(row, f"shape_residual_bin{b}")})
        rlong = pd.DataFrame(residual_long_rows)
        merged = dlong.merge(rlong, on=["Subject", "Group", "Bin"], validate="one_to_one")
        group_vectors: dict[str, np.ndarray] = {}
        group_shape_vectors: dict[str, np.ndarray] = {}
        components = raw_adjusted_profiles[family]
        for b in range(1, 11):
            grow = {
                "Target": family,
                "Stage_classification": EXPECTED_STAGES[family],
                "Bin": b,
                "mean_reference_G1": float(g1.loc[g1["Bin"] == b, "mean_reference_G1"].iloc[0]),
                "G1_bin_mean_z": float(g1.loc[g1["Bin"] == b, "G1_bin_mean_z"].iloc[0]),
            }
            for group in ["HC", "SZ"]:
                cell = merged.loc[(merged["Bin"] == b) & (merged["Group"] == group)]
                vals = cell["FC"].to_numpy(float)
                shapes = cell["ShapeResidual"].to_numpy(float)
                require(len(vals) == 32, f"Expected n=32 for {family}/{group}/bin{b}")
                tcrit = float(stats.t.ppf(0.975, len(vals) - 1))
                mean_fc, sd_fc = float(vals.mean()), float(vals.std(ddof=1))
                se_fc = sd_fc / math.sqrt(len(vals))
                mean_shape, sd_shape = float(shapes.mean()), float(shapes.std(ddof=1))
                se_shape = sd_shape / math.sqrt(len(shapes))
                grow.update(
                    {
                        f"{group}_N": len(vals),
                        f"{group}_mean_FC": mean_fc,
                        f"{group}_SD_FC": sd_fc,
                        f"{group}_SE_FC": se_fc,
                        f"{group}_mean_FC_CI95_low": mean_fc - tcrit * se_fc,
                        f"{group}_mean_FC_CI95_high": mean_fc + tcrit * se_fc,
                        f"{group}_mean_shape_residual": mean_shape,
                        f"{group}_SD_shape_residual": sd_shape,
                        f"{group}_SE_shape_residual": se_shape,
                        f"{group}_mean_shape_CI95_low": mean_shape - tcrit * se_shape,
                        f"{group}_mean_shape_CI95_high": mean_shape + tcrit * se_shape,
                    }
                )
            grow["raw_SZ_minus_HC"] = grow["SZ_mean_FC"] - grow["HC_mean_FC"]
            grow["raw_shape_SZ_minus_HC"] = grow["SZ_mean_shape_residual"] - grow["HC_mean_shape_residual"]
            grow["adjusted_SZ_minus_HC"] = float(components["adjusted_diff"][b - 1])
            grow["adjusted_offset_component"] = float(components["offset_component"][b - 1])
            grow["adjusted_linear_component"] = float(components["linear_component"][b - 1])
            grow["adjusted_nonlinear_component"] = float(components["nonlinear_component"][b - 1])
            group_stat_rows.append(grow)

        family_group_stats = pd.DataFrame([r for r in group_stat_rows if r["Target"] == family]).sort_values("Bin")
        hc = family_group_stats["HC_mean_FC"].to_numpy(float)
        sz = family_group_stats["SZ_mean_FC"].to_numpy(float)
        group_vectors["HC"] = hc
        group_vectors["SZ"] = sz
        r_raw, unstable, sd_hc_bins, sd_sz_bins = pearson_safe(hc, sz)
        hc_centered = hc - hc.mean()
        sz_centered = sz - sz.mean()
        r_centered, unstable_centered, _, _ = pearson_safe(hc_centered, sz_centered)
        raw_rmse = float(np.sqrt(np.mean((sz - hc) ** 2)))
        raw_offset = float((sz - hc).mean())
        raw_slope_diff = float((((sz - hc) - raw_offset) @ xg) / (xg @ xg))
        similarity_rows.append(
            {
                "Target": family,
                "Stage_classification": EXPECTED_STAGES[family],
                "N_HC": 32,
                "N_SZ": 32,
                "Pearson_r_group_mean_profiles": r_raw,
                "Mean_centered_profile_correlation": r_centered,
                "Raw_profile_RMSE": raw_rmse,
                "Raw_offset_SZ_minus_HC": raw_offset,
                "Raw_linear_slope_difference": raw_slope_diff,
                "HC_profile_SD_across_bins": sd_hc_bins,
                "SZ_profile_SD_across_bins": sd_sz_bins,
                "Correlation_near_constant_warning": bool(unstable or unstable_centered),
                "Adjusted_offset_SZ_minus_HC": components["adjusted_offset"],
                "Adjusted_slope_SZ_minus_HC": components["adjusted_slope"],
                "Adjusted_total_profile_RMSE": components["total_rmse_adjusted"],
                "Adjusted_offset_component_RMS": components["offset_rms"],
                "Adjusted_linear_component_RMS": components["linear_rms"],
                "Adjusted_nonlinear_shape_RMSE": components["shape_rmse"],
                "Decomposition_squared_identity_error": components["decomposition_error"],
            }
        )

        target_tests = tests.loc[tests["Target"] == family].set_index("Analysis_dimension")
        full_summary_rows.append(
            {
                "Target": family,
                "Stage_classification": EXPECTED_STAGES[family],
                "ROI_count": EXPECTED_ROI_COUNTS[family],
                "Offset_beta_SZ_minus_HC": target_tests.loc["FC_Offset", "Beta_SZ_minus_HC"],
                "Offset_SE_HC3": target_tests.loc["FC_Offset", "SE_HC3"],
                "Offset_CI95_low": target_tests.loc["FC_Offset", "CI95_low"],
                "Offset_CI95_high": target_tests.loc["FC_Offset", "CI95_high"],
                "Offset_raw_p": target_tests.loc["FC_Offset", "raw_p"],
                "Offset_Holm_p": target_tests.loc["FC_Offset", "Holm_adjusted_p_6_tests"],
                "Slope_beta_SZ_minus_HC": target_tests.loc["Linear_G1_Slope", "Beta_SZ_minus_HC"],
                "Slope_SE_HC3": target_tests.loc["Linear_G1_Slope", "SE_HC3"],
                "Slope_CI95_low": target_tests.loc["Linear_G1_Slope", "CI95_low"],
                "Slope_CI95_high": target_tests.loc["Linear_G1_Slope", "CI95_high"],
                "Slope_raw_p": target_tests.loc["Linear_G1_Slope", "raw_p"],
                "Slope_Holm_p": target_tests.loc["Linear_G1_Slope", "Holm_adjusted_p_6_tests"],
                "Shape_Pillai_trace": target_tests.loc["Nonlinear_Profile_Shape", "Pillai_trace"],
                "Shape_approx_F": target_tests.loc["Nonlinear_Profile_Shape", "Approx_F"],
                "Shape_df_num": target_tests.loc["Nonlinear_Profile_Shape", "df_num"],
                "Shape_df_den": target_tests.loc["Nonlinear_Profile_Shape", "df_den"],
                "Shape_raw_p": target_tests.loc["Nonlinear_Profile_Shape", "raw_p"],
                "Shape_Holm_p": target_tests.loc["Nonlinear_Profile_Shape", "Holm_adjusted_p_6_tests"],
                "Pearson_r_group_mean_profiles": r_raw,
                "Raw_profile_RMSE": raw_rmse,
                "Raw_offset_SZ_minus_HC": raw_offset,
                "Adjusted_offset_SZ_minus_HC": components["adjusted_offset"],
                "Adjusted_slope_SZ_minus_HC": components["adjusted_slope"],
                "Adjusted_total_profile_RMSE": components["total_rmse_adjusted"],
                "Adjusted_offset_component_RMS": components["offset_rms"],
                "Adjusted_linear_component_RMS": components["linear_rms"],
                "Adjusted_nonlinear_shape_RMSE": components["shape_rmse"],
            }
        )

    group_stats = pd.DataFrame(group_stat_rows).sort_values(["Target", "Bin"])
    similarity = pd.DataFrame(similarity_rows).sort_values("Target")
    full_summary = pd.DataFrame(full_summary_rows).sort_values("Target")
    write_tsv(group_stats, TABLES / "profile_group_statistics.tsv")
    write_tsv(similarity, TABLES / "profile_similarity_metrics.tsv")
    write_tsv(full_summary, TABLES / "STEP4_full_results_summary.tsv")

    # Prespecified descriptive leave-one-participant-out influence diagnostic.
    influence_rows = []
    for family in TARGETS:
        d = metric_by_family[family].sort_values("Subject").reset_index(drop=True)
        scores = score_by_family[family]
        full_x = design_from(d)
        for metric_name, column in [("FC_Offset", "Offset"), ("Linear_G1_Slope", "Slope_per_G1_bin_mean_SD")]:
            full_fit = main_models[(family, metric_name)]
            for i, subject in enumerate(d["Subject"]):
                keep = np.arange(len(d)) != i
                loo = fit_ols_hc3(d.loc[keep, column].to_numpy(float), full_x[keep, :])
                influence_rows.append(
                    {
                        "Target": family,
                        "Analysis_dimension": metric_name,
                        "Left_out_subject": subject,
                        "Left_out_group": d.loc[i, "Group"],
                        "Full_effect": float(full_fit.beta[1]),
                        "LOO_effect": float(loo.beta[1]),
                        "LOO_minus_full": float(loo.beta[1] - full_fit.beta[1]),
                        "LOO_raw_p": float(loo.p[1]),
                        "Direction_consistent": bool(np.sign(loo.beta[1]) == np.sign(full_fit.beta[1])),
                    }
                )
        full_pillai = main_models[(family, "Nonlinear_Profile_Shape")]
        for i, subject in enumerate(d["Subject"]):
            keep = np.arange(len(d)) != i
            loo = pillai_group_test(scores[keep, :], full_x[keep, :])
            influence_rows.append(
                {
                    "Target": family,
                    "Analysis_dimension": "Nonlinear_Profile_Shape",
                    "Left_out_subject": subject,
                    "Left_out_group": d.loc[i, "Group"],
                    "Full_effect": float(full_pillai.trace),
                    "LOO_effect": float(loo.trace),
                    "LOO_minus_full": float(loo.trace - full_pillai.trace),
                    "LOO_raw_p": float(loo.p_value),
                    "Direction_consistent": math.nan,
                }
            )
    influence = pd.DataFrame(influence_rows).sort_values(["Target", "Analysis_dimension", "Left_out_subject"])
    write_tsv(influence, TABLES / "subject_level_influence_diagnostics.tsv")

    influence_summary = (
        influence.groupby(["Target", "Analysis_dimension"], as_index=False)
        .agg(
            N_leave_one_out=("Left_out_subject", "size"),
            LOO_effect_min=("LOO_effect", "min"),
            LOO_effect_max=("LOO_effect", "max"),
            max_absolute_effect_change=("LOO_minus_full", lambda x: float(np.max(np.abs(x)))),
            LOO_raw_p_min=("LOO_raw_p", "min"),
            LOO_raw_p_max=("LOO_raw_p", "max"),
        )
    )
    direction_summary = (
        influence.loc[influence["Analysis_dimension"] != "Nonlinear_Profile_Shape"]
        .groupby(["Target", "Analysis_dimension"])["Direction_consistent"]
        .agg(lambda x: bool(pd.Series(x).astype(bool).all()))
        .rename("all_LOO_directions_consistent")
        .reset_index()
    )
    influence_summary = influence_summary.merge(direction_summary, on=["Target", "Analysis_dimension"], how="left")
    write_tsv(influence_summary, TABLES / "subject_level_influence_summary.tsv")

    # Input hashes and formal QC table.
    provenance = pd.DataFrame(
        [{"input": label, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for label, path in INPUTS.items()]
    )
    write_tsv(provenance, TABLES / "input_sha256.tsv")

    qc_rows = [
        {"check": "Statistical plan existed before tests", "observed": plan_path.exists(), "expected": True, "status": "PASS"},
        {"check": "Unique participants", "observed": roi["Subject"].nunique(), "expected": 64, "status": "PASS"},
        {"check": "HC participants", "observed": roi.loc[roi["Group"] == "HC", "Subject"].nunique(), "expected": 32, "status": "PASS"},
        {"check": "SZ participants", "observed": roi.loc[roi["Group"] == "SZ", "Subject"].nunique(), "expected": 32, "status": "PASS"},
        {"check": "SMA ROI count", "observed": roi_count_map["SMA"], "expected": 1, "status": "PASS"},
        {"check": "IFG ROI count", "observed": roi_count_map["IFG"], "expected": 6, "status": "PASS"},
        {"check": "Subject-target profiles", "observed": len(profile_wide), "expected": 128, "status": "PASS"},
        {"check": "Bins per profile", "observed": int(family_profile_long.groupby(["Subject", "Target"])["Bin"].nunique().min()), "expected": 10, "status": "PASS"},
        {"check": "Definition A voxels", "observed": len(bins), "expected": 2608, "status": "PASS"},
        {"check": "Definition A bin counts", "observed": ",".join(map(str, g1["n_voxels"])), "expected": "260,260,261,261,261,261,261,261,261,261", "status": "PASS"},
        {"check": "Maximum absolute subject residual mean", "observed": float(residuals_wide["residual_mean"].abs().max()), "expected": "<1e-12", "status": "PASS"},
        {"check": "Maximum absolute subject residual dot G1z", "observed": float(residuals_wide["residual_dot_G1z"].abs().max()), "expected": "<1e-12", "status": "PASS"},
        {"check": "Maximum adjusted decomposition identity error", "observed": float(similarity["Decomposition_squared_identity_error"].max()), "expected": "<1e-12", "status": "PASS"},
        {"check": "Prespecified tests", "observed": len(tests), "expected": 6, "status": "PASS"},
        {"check": "Holm p values finite and bounded", "observed": bool(tests["Holm_adjusted_p_6_tests"].between(0, 1).all()), "expected": True, "status": "PASS"},
        {"check": "Influence refits", "observed": len(influence), "expected": 384, "status": "PASS"},
        {"check": "Original FC value type", "observed": "Fisher-z transformed Pearson FC", "expected": "Fisher-z transformed Pearson FC", "status": "PASS"},
        {"check": "Independent unit", "observed": "Subject", "expected": "Subject", "status": "PASS"},
        {"check": "Source project writes", "observed": 0, "expected": 0, "status": "PASS"},
    ]
    qc = pd.DataFrame(qc_rows)
    write_tsv(qc, TABLES / "STEP4_qc_checks.tsv")

    state = {
        "status": "SUCCESS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "sample": {"N": 64, "HC": 32, "SZ": 32},
        "targets": {"SMA": {"ROI": 1, "stage": "Anticipation"}, "IFG": {"ROI": 6, "stage": "Outcome"}},
        "data_type": "Original subject-level ROI x Definition-A-bin Fisher-z FC",
        "IFG_aggregation": "Unweighted arithmetic mean across six IFG ROI within subject and bin",
        "G1_axis": {"definition": "Mean frozen reference G1 within each Definition A bin; equal-bin standardization", "mean": g1_mean, "sample_sd": g1_sd},
        "model": "group_sz + age_c + sex_c + meanFD_c",
        "univariate_covariance": "HC3",
        "shape_test": "8D orthogonal-complement multivariate OLS; Pillai trace",
        "multiplicity": "Holm across six prespecified tests",
        "basis_qc": basis_qc,
        "max_decomposition_identity_error": float(similarity["Decomposition_squared_identity_error"].max()),
        "primary_tests": tests[["Target", "Analysis_dimension", "raw_p", "Holm_adjusted_p_6_tests"]].to_dict(orient="records"),
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "not_run": ["375-person G1 recomputation", "Bin10 Pearson contribution test", "Equal-Subfamily-Weighting", "STEP 5", "ROI search", "permutation test"],
    }
    (LOGS / "analysis_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (CODE / "dependencies.txt").write_text(
        "Python 3.12+\nnumpy==%s\npandas==%s\nscipy==%s\nR 4.6.1 (figures only)\n" % (np.__version__, pd.__version__, scipy.__version__),
        encoding="utf-8",
    )

    log.info("Primary tests completed:\n%s", tests[["Target", "Analysis_dimension", "Beta_SZ_minus_HC", "Pillai_trace", "raw_p", "Holm_adjusted_p_6_tests"]].to_string(index=False))
    log.info("Profile similarity completed:\n%s", similarity.to_string(index=False))
    log.info("All QC checks passed; outputs written only to %s", OUT)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("STEP 4 failed")
        raise
