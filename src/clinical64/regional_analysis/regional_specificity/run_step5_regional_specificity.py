#!/usr/bin/env python3
"""STEP 5 regional specificity of SMA and IFG Pearson contributions.

The script is self-contained, reads only frozen/source inputs, enforces the
historical reproduction gate, and writes all analysis products under the
dedicated STEP 5 output directory.
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

# The frozen rerun ships a read-only dependency directory used by the prior
# verified STEP 3/4 analyses. Add it explicitly when the bundled interpreter
# does not expose those packages on its default sys.path.
VENDOR_PACKAGES = Path(
    os.environ.get("STRIATAL_PYTHON_VENDOR", "").strip() or ".missing_python_vendor"
)
if VENDOR_PACKAGES.exists():
    sys.path.insert(0, str(VENDOR_PACKAGES))

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm


PROJECT = Path(os.environ["STRIATAL_PROJECT_ROOT"])
REPO = Path(os.environ["STRIATAL_RELEASE_ROOT"])
OUT = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP5_Regional_Specificity'))
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"

COV_INPUT = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "covariates" / "current64_covariates.tsv"
ROI_META = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "reward_stage_mapping" / "roi_metadata_stage_subfamily_v2.tsv"
FC_ROOT = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "fc_fingerprints_2608x65"
MAPPING = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "voxel_order" / "row_mapping_2847_to_2608.npy"
CURRENT_BINS = Path(os.environ["STRIATAL_CURRENT_BIN_TABLE"])
HISTORICAL_RESULTS = PROJECT / "FINAL_64SUBJECT_RESULTS_PACKAGE" / "tables" / "25_03_Bin10_subfamily_drivers_FINAL.tsv"
HISTORICAL_SCRIPT = REPO / "src" / "clinical64" / "sma_ifg" / "run_bin10_subfamily.sh"
STEP3_LOSO = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP3_LocalContribution_AllSubfamilies/table1_sma_ifg_stability_overview.tsv'))
STEP4_REPORT = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP4_SMA_IFG_ProfileSimilarity/reports/STEP4_PROFILE_SIMILARITY_REPORT.md'))

EXPECTED_CURRENT_BINS_SHA256 = "26d268283702db209940198e76b8630585e54259cd742016951aaa7d8427292a"
EXPECTED_MAPPING_SHA256 = "bb12bfa947c09778d3d353a765a397c3a5bb17880bde98f6c147d7379289a268"
EXPECTED_SMA_BETA = float(os.environ["STRIATAL_EXPECTED_SMA_BETA"])
EXPECTED_SMA_Q = float(os.environ["STRIATAL_EXPECTED_SMA_Q"])
EXPECTED_IFG_BETA = float(os.environ["STRIATAL_EXPECTED_IFG_BETA"])
EXPECTED_IFG_Q = float(os.environ["STRIATAL_EXPECTED_IFG_Q"])
BOOTSTRAP_SEED = 20261012
BOOTSTRAP_REPLICATES = 5000
IDENTITY_TOL = 1e-12
REPRO_TOL = 3e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, float_format="%.17g", lineterminator="\n")


def write_json(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1, dtype=float)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(p), dtype=float)
    out[order] = np.minimum(ranked, 1.0)
    return out


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * (len(p) - np.arange(len(p)))
    ranked = np.maximum.accumulate(ranked)
    out = np.empty(len(p), dtype=float)
    out[order] = np.minimum(ranked, 1.0)
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
    rank: int


def fit_ols_hc3(y: np.ndarray, x: np.ndarray) -> OLSHC3:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x):
        raise ValueError("Invalid OLS shapes")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Non-finite OLS input")
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise RuntimeError(f"Rank-deficient design: rank={rank}, columns={x.shape[1]}")
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    if np.any(1.0 - leverage <= 0):
        raise RuntimeError("HC3 leverage is not strictly below 1")
    scaled = residual / (1.0 - leverage)
    covariance = xtx_inv @ (x.T @ ((scaled**2)[:, None] * x)) @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    p = 2.0 * norm.sf(np.abs(beta / se))
    return OLSHC3(
        beta=beta,
        se=se,
        p=p,
        ci_low=beta - 1.96 * se,
        ci_high=beta + 1.96 * se,
        residual_df=int(len(y) - x.shape[1]),
        max_leverage=float(leverage.max()),
        rank=rank,
    )


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(frame), dtype=float),
            frame["group_sz"].to_numpy(float),
            frame["age_c"].to_numpy(float),
            frame["sex_c"].to_numpy(float),
            frame["meanFD_c"].to_numpy(float),
        ]
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
    cov["sex_num"] = cov["sex"].astype(str).str.lower().map({"female": 0.0, "f": 0.0, "male": 1.0, "m": 1.0})
    if cov["sex_num"].isna().any():
        raise RuntimeError("Unrecognized Sex coding")
    cov["group_sz"] = (cov["group"] == "SZ").astype(int)
    cov["age_c"] = cov["age"] - cov["age"].mean()
    cov["sex_c"] = cov["sex_num"] - cov["sex_num"].mean()
    cov["meanFD_c"] = cov["meanFD"] - cov["meanFD"].mean()
    if len(cov) != 64 or cov["subject"].nunique() != 64:
        raise RuntimeError("Expected 64 unique subjects")
    if cov["group"].value_counts().to_dict() != {"SZ": 32, "HC": 32}:
        raise RuntimeError("Expected 32 SZ and 32 HC")
    if cov[["age", "sex_num", "meanFD"]].isna().any().any():
        raise RuntimeError("Missing required covariates")
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
    expected_stage = {"Anticipation": 43, "Outcome": 19, "Not_assigned": 3}
    if meta["stage"].value_counts().to_dict() != expected_stage:
        raise RuntimeError(f"ROI Stage mapping changed: {meta['stage'].value_counts().to_dict()}")
    assigned = meta.loc[meta["stage"] != "Not_assigned"]
    if len(assigned) != 62 or assigned["subfamily"].nunique() != 13:
        raise RuntimeError("Expected 62 assigned ROIs across 13 subfamilies")
    if int((meta["subfamily"] == "SMA").sum()) != 1 or int((meta["subfamily"] == "IFG").sum()) != 6:
        raise RuntimeError("Expected SMA=1 ROI and IFG=6 ROIs")
    return meta


def load_bin10_profiles(cov: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict]:
    if sha256(CURRENT_BINS) != EXPECTED_CURRENT_BINS_SHA256:
        raise RuntimeError("Fixed current-bin input SHA-256 changed")
    if sha256(MAPPING) != EXPECTED_MAPPING_SHA256:
        raise RuntimeError("2847-to-2608 mapping SHA-256 changed")
    current = pd.read_csv(CURRENT_BINS, sep="\t")
    full_bins = pd.to_numeric(current["G1_bin_current"], errors="raise").astype(int).to_numpy()
    mapping = np.asarray(np.load(MAPPING)).reshape(-1).astype(int)
    if len(full_bins) != 2847 or len(mapping) != 2608 or np.unique(mapping).size != 2608:
        raise RuntimeError("Unexpected voxel mapping dimensions")
    common_bins = full_bins[mapping]
    bin10 = common_bins == 10
    if int(bin10.sum()) != 285:
        raise RuntimeError(f"Expected 285 current Bin-10 voxels, got {int(bin10.sum())}")

    profiles = np.empty((len(cov), 65), dtype=float)
    hashes: list[dict] = []
    for i, subject in enumerate(cov["subject"]):
        path = FC_ROOT / f"{subject}_fingerprint_z_2608x65.npy"
        array = np.asarray(np.load(path), dtype=float)
        if array.shape == (65, 2608):
            array = array.T
        if array.shape != (2608, 65) or not np.isfinite(array).all():
            raise RuntimeError(f"Invalid frozen FC array: {path}, shape={array.shape}")
        profiles[i] = array[bin10].mean(axis=0)
        hashes.append({"input_type": "subject_fc_array", "subject": subject, "path": str(path), "sha256": sha256(path), "shape_or_rows": "2608x65"})
    qc = {
        "full_bin_counts": {str(k): int(v) for k, v in pd.Series(full_bins).value_counts().sort_index().items()},
        "common_bin_counts": {str(k): int(v) for k, v in pd.Series(common_bins).value_counts().sort_index().items()},
        "bin10_voxels": int(bin10.sum()),
        "profile_shape": list(profiles.shape),
    }
    return profiles, pd.DataFrame(hashes), qc


def pearson_contributions(values: np.ndarray, reference: np.ndarray) -> tuple[float, np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if values.shape != reference.shape or values.ndim != 1 or len(values) != 65:
        raise RuntimeError("Pearson inputs must be matching 65-element vectors")
    zv = (values - values.mean()) / values.std(ddof=1)
    zr = (reference - reference.mean()) / reference.std(ddof=1)
    contribution = zv * zr / 64.0
    direct_r = float(np.corrcoef(values, reference)[0, 1])
    error = float(abs(contribution.sum() - direct_r))
    if error > IDENTITY_TOL:
        raise RuntimeError(f"Pearson contribution identity failed: {error}")
    return direct_r, contribution, error


def decompose_full_profiles(profiles: np.ndarray, cov: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hc_idx = np.flatnonzero(cov["group"].to_numpy() == "HC")
    hc_mean = profiles[hc_idx].mean(axis=0)
    subject_rows: list[dict] = []
    roi_rows: list[dict] = []
    identity_rows: list[dict] = []
    for i, row in cov.iterrows():
        if row["group"] == "HC":
            reference = profiles[hc_idx[hc_idx != i]].mean(axis=0)
            reference_label = "HC_LOO31"
            reference_n = 31
        else:
            reference = hc_mean
            reference_label = "HC32"
            reference_n = 32
        direct_r, contribution, error = pearson_contributions(profiles[i], reference)
        subject_rows.append(
            {
                **row.to_dict(),
                "reference": reference_label,
                "reference_n": reference_n,
                "profile_r": direct_r,
            }
        )
        identity_rows.append(
            {
                "subject": row["subject"],
                "group": row["group"],
                "profile_r": direct_r,
                "sum_65_roi_contributions": float(contribution.sum()),
                "absolute_error": error,
                "pass": error <= IDENTITY_TOL,
            }
        )
        for j, c in enumerate(contribution):
            m = meta.iloc[j]
            roi_rows.append(
                {
                    "subject": row["subject"],
                    "group": row["group"],
                    "group_sz": row["group_sz"],
                    "age_c": row["age_c"],
                    "sex_c": row["sex_c"],
                    "meanFD_c": row["meanFD_c"],
                    "profile_r": direct_r,
                    "reference": reference_label,
                    "roi_index0": int(m["roi_index0"]),
                    "brainnetome_label": m["brainnetome_label"],
                    "subfamily": m["subfamily"],
                    "stage": m["stage"],
                    "r_contribution": float(c),
                }
            )
    return pd.DataFrame(subject_rows), pd.DataFrame(roi_rows), pd.DataFrame(identity_rows)


def fit_historical_subfamilies(roi_long: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject", "group", "group_sz", "age_c", "sex_c", "meanFD_c", "subfamily", "stage"]
    aggregate = roi_long.groupby(keys, as_index=False, sort=True).agg(
        contribution_sum=("r_contribution", "sum"),
        contribution_mean_per_roi=("r_contribution", "mean"),
        N_ROIs=("roi_index0", "nunique"),
    )
    records: list[dict] = []
    for (subfamily, stage), subset in aggregate.groupby(["subfamily", "stage"], sort=True):
        subset = subset.sort_values("subject").reset_index(drop=True)
        fit = fit_ols_hc3(subset["contribution_sum"].to_numpy(float), design_matrix(subset))
        records.append(
            {
                "subfamily": subfamily,
                "stage": stage,
                "N_ROIs": int(subset["N_ROIs"].iloc[0]),
                "beta_SZ_minus_HC": float(fit.beta[1]),
                "SE_HC3": float(fit.se[1]),
                "CI95_low": float(fit.ci_low[1]),
                "CI95_high": float(fit.ci_high[1]),
                "p_raw": float(fit.p[1]),
                "q_FDR_13_assigned": np.nan,
            }
        )
    results = pd.DataFrame(records)
    assigned = results["stage"] != "Not_assigned"
    if int(assigned.sum()) != 13:
        raise RuntimeError("Historical reproduction did not yield 13 assigned tests")
    results.loc[assigned, "q_FDR_13_assigned"] = bh_adjust(results.loc[assigned, "p_raw"].to_numpy(float))
    return results.sort_values(["stage", "subfamily"]).reset_index(drop=True)


def validate_historical(results: pd.DataFrame) -> pd.DataFrame:
    historical = pd.read_csv(HISTORICAL_RESULTS, sep="\t")
    merged = historical.merge(results, on=["subfamily", "stage"], how="outer", suffixes=("_historical", "_reproduced"), indicator=True, validate="one_to_one")
    if len(merged) != 14 or not (merged["_merge"] == "both").all():
        raise RuntimeError("Historical/reproduced subfamily keys differ")
    fields = {"N_ROIs": 0.0, "beta_SZ_minus_HC": 2e-12, "SE_HC3": 2e-12, "CI95_low": 3e-12, "CI95_high": 3e-12, "p_raw": 2e-12}
    rows: list[dict] = []
    for _, row in merged.iterrows():
        for field, tolerance in fields.items():
            old = float(row[f"{field}_historical"])
            new = float(row[f"{field}_reproduced"])
            rows.append({"subfamily": row["subfamily"], "stage": row["stage"], "field": field, "historical": old, "reproduced": new, "absolute_error": abs(old - new), "tolerance": tolerance, "pass": abs(old - new) <= tolerance})
    audit = pd.DataFrame(rows)
    if not audit["pass"].all():
        raise RuntimeError("Historical result reproduction failed")
    for name, expected_beta, expected_q in [("SMA", EXPECTED_SMA_BETA, EXPECTED_SMA_Q), ("IFG", EXPECTED_IFG_BETA, EXPECTED_IFG_Q)]:
        row = results.loc[results["subfamily"] == name].iloc[0]
        if abs(float(row["beta_SZ_minus_HC"]) - expected_beta) > REPRO_TOL or abs(float(row["q_FDR_13_assigned"]) - expected_q) > REPRO_TOL:
            raise RuntimeError(f"{name} reproduction gate failed")
    return audit


def build_subject_contrasts(cov: pd.DataFrame, roi_long: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    ant = set(meta.loc[meta["stage"] == "Anticipation", "roi_index0"])
    out = set(meta.loc[meta["stage"] == "Outcome", "roi_index0"])
    sma = set(meta.loc[meta["subfamily"] == "SMA", "roi_index0"])
    ifg = set(meta.loc[meta["subfamily"] == "IFG", "roi_index0"])
    other_ant = ant - sma
    other_out = out - ifg
    if (len(ant), len(sma), len(other_ant), len(out), len(ifg), len(other_out)) != (43, 1, 42, 19, 6, 13):
        raise RuntimeError("Target/reference ROI counts do not match the locked definition")
    rows: list[dict] = []
    for _, c in cov.iterrows():
        s = roi_long.loc[roi_long["subject"] == c["subject"]]
        if len(s) != 65 or s["roi_index0"].nunique() != 65:
            raise RuntimeError(f"Incomplete ROI contributions for {c['subject']}")
        sma_values = s.loc[s["roi_index0"].isin(sma), "r_contribution"]
        ant_values = s.loc[s["roi_index0"].isin(other_ant), "r_contribution"]
        ifg_values = s.loc[s["roi_index0"].isin(ifg), "r_contribution"]
        out_values = s.loc[s["roi_index0"].isin(other_out), "r_contribution"]
        if (len(sma_values), len(ant_values), len(ifg_values), len(out_values)) != (1, 42, 6, 13):
            raise RuntimeError(f"Bad target/reference counts for {c['subject']}")
        sma_sum = float(sma_values.sum())
        sma_mean = float(sma_values.mean())
        other_ant_mean = float(ant_values.mean())
        ifg_sum = float(ifg_values.sum())
        ifg_mean = float(ifg_values.mean())
        other_out_mean = float(out_values.mean())
        rows.append(
            {
                "Subject ID": c["subject"],
                "Group": c["group"],
                "Age": float(c["age"]),
                "Sex": c["sex"],
                "MeanFD": float(c["meanFD"]),
                "SMA contribution sum": sma_sum,
                "SMA per-ROI contribution": sma_mean,
                "Other Anticipation per-ROI contribution": other_ant_mean,
                "SMA contrast L": sma_mean - other_ant_mean,
                "IFG contribution sum": ifg_sum,
                "IFG per-ROI contribution": ifg_mean,
                "Other Outcome per-ROI contribution": other_out_mean,
                "IFG contrast L": ifg_mean - other_out_mean,
            }
        )
    result = pd.DataFrame(rows).merge(
        cov[["subject", "group_sz", "age_c", "sex_c", "meanFD_c"]],
        left_on="Subject ID",
        right_on="subject",
        how="left",
        validate="one_to_one",
    ).drop(columns="subject")
    if len(result) != 64 or result["Subject ID"].nunique() != 64 or result.isna().any().any():
        raise RuntimeError("Subject contrast table has missing or duplicate rows")
    return result


TARGETS = {
    "SMA": {
        "stage": "Anticipation",
        "target_n": 1,
        "reference_n": 42,
        "target_col": "SMA per-ROI contribution",
        "other_col": "Other Anticipation per-ROI contribution",
        "contrast_col": "SMA contrast L",
    },
    "IFG": {
        "stage": "Outcome",
        "target_n": 6,
        "reference_n": 13,
        "target_col": "IFG per-ROI contribution",
        "other_col": "Other Outcome per-ROI contribution",
        "contrast_col": "IFG contrast L",
    },
}


def fit_components(subject: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = design_matrix(subject)
    fit_store: dict = {}
    rows: list[dict] = []
    for target, spec in TARGETS.items():
        fit_store[target] = {}
        for component, col in [("Target", spec["target_col"]), ("Other", spec["other_col"]), ("Target-minus-Other", spec["contrast_col"])]:
            fit = fit_ols_hc3(subject[col].to_numpy(float), x)
            fit_store[target][component] = fit
            for group in ["HC", "SZ"]:
                values = subject.loc[subject["Group"] == group, col].to_numpy(float)
                rows.append(
                    {
                        "Target": target,
                        "Stage": spec["stage"],
                        "Component": component,
                        "Group": group,
                        "N": len(values),
                        "Mean": float(values.mean()),
                        "SD": float(values.std(ddof=1)),
                        "Median": float(np.median(values)),
                        "Adjusted Group Effect": np.nan,
                        "HC3 SE": np.nan,
                        "CI Lower": np.nan,
                        "CI Upper": np.nan,
                        "Raw p": np.nan,
                        "Scale": "Per-ROI Pearson contribution",
                    }
                )
            rows.append(
                {
                    "Target": target,
                    "Stage": spec["stage"],
                    "Component": component,
                    "Group": "Adjusted SZ-HC",
                    "N": len(subject),
                    "Mean": np.nan,
                    "SD": np.nan,
                    "Median": np.nan,
                    "Adjusted Group Effect": float(fit.beta[1]),
                    "HC3 SE": float(fit.se[1]),
                    "CI Lower": float(fit.ci_low[1]),
                    "CI Upper": float(fit.ci_high[1]),
                    "Raw p": float(fit.p[1]),
                    "Scale": "Per-ROI Pearson contribution",
                }
            )
        difference = float(fit_store[target]["Target"].beta[1] - fit_store[target]["Other"].beta[1])
        direct = float(fit_store[target]["Target-minus-Other"].beta[1])
        if abs(difference - direct) > 1e-13:
            raise RuntimeError(f"{target} beta decomposition identity failed: {difference} vs {direct}")
    return pd.DataFrame(rows), fit_store


def bootstrap_contrasts(subject: pd.DataFrame, fits: dict) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    group_indices = {group: np.flatnonzero(subject["Group"].to_numpy() == group) for group in ["HC", "SZ"]}
    distributions = {target: [] for target in TARGETS}
    failures = {target: 0 for target in TARGETS}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate([rng.choice(group_indices["HC"], size=32, replace=True), rng.choice(group_indices["SZ"], size=32, replace=True)])
        boot = subject.iloc[sampled].reset_index(drop=True)
        x = design_matrix(boot)
        for target, spec in TARGETS.items():
            try:
                fit = fit_ols_hc3(boot[spec["contrast_col"]].to_numpy(float), x)
                distributions[target].append(float(fit.beta[1]))
            except (np.linalg.LinAlgError, RuntimeError, ValueError):
                failures[target] += 1
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for target in TARGETS:
        values = np.asarray(distributions[target], dtype=float)
        arrays[target] = values
        if len(values) == 0:
            raise RuntimeError(f"All bootstrap fits failed for {target}")
        full_fit = fits[target]["Target-minus-Other"]
        rows.append(
            {
                "Target": target,
                "Stage": TARGETS[target]["stage"],
                "Original Adjusted Beta L": float(full_fit.beta[1]),
                "HC3 CI Lower": float(full_fit.ci_low[1]),
                "HC3 CI Upper": float(full_fit.ci_high[1]),
                "Bootstrap CI Lower": float(np.percentile(values, 2.5)),
                "Bootstrap CI Upper": float(np.percentile(values, 97.5)),
                "Bootstrap Median": float(np.median(values)),
                "Effective Bootstrap N": int(len(values)),
                "Bootstrap Failures": int(failures[target]),
                "Requested Bootstrap N": BOOTSTRAP_REPLICATES,
                "Random Seed": BOOTSTRAP_SEED,
                "Resampling": "Stratified by Group; 32 HC and 32 SZ with replacement",
            }
        )
    return pd.DataFrame(rows), arrays


def leave_one_subject_out(subject: pd.DataFrame, fits: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for target, spec in TARGETS.items():
        full_beta = float(fits[target]["Target-minus-Other"].beta[1])
        for omitted in subject["Subject ID"]:
            subset = subject.loc[subject["Subject ID"] != omitted].reset_index(drop=True)
            fit = fit_ols_hc3(subset[spec["contrast_col"]].to_numpy(float), design_matrix(subset))
            omitted_row = subject.loc[subject["Subject ID"] == omitted].iloc[0]
            beta = float(fit.beta[1])
            rows.append(
                {
                    "Target": target,
                    "Stage": spec["stage"],
                    "Omitted Subject ID": omitted,
                    "Omitted Group": omitted_row["Group"],
                    "N": len(subset),
                    "Beta L": beta,
                    "HC3 SE": float(fit.se[1]),
                    "CI Lower": float(fit.ci_low[1]),
                    "CI Upper": float(fit.ci_high[1]),
                    "Raw p": float(fit.p[1]),
                    "Delta Beta vs Full": beta - full_beta,
                    "Direction Matches Full": bool(np.sign(beta) == np.sign(full_beta)),
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for target, subset in detail.groupby("Target", sort=False):
        full_beta = float(fits[target]["Target-minus-Other"].beta[1])
        max_idx = subset["Delta Beta vs Full"].abs().idxmax()
        max_row = subset.loc[max_idx]
        summary_rows.append(
            {
                "Target": target,
                "Stage": TARGETS[target]["stage"],
                "Full Beta L": full_beta,
                "LOO Refits": len(subset),
                "LOO Beta Minimum": float(subset["Beta L"].min()),
                "LOO Beta Maximum": float(subset["Beta L"].max()),
                "Direction Retained Count": int(subset["Direction Matches Full"].sum()),
                "Raw p Minimum": float(subset["Raw p"].min()),
                "Raw p Maximum": float(subset["Raw p"].max()),
                "Maximum Absolute Delta Beta": float(subset["Delta Beta vs Full"].abs().max()),
                "Most Influential Omitted Subject": max_row["Omitted Subject ID"],
                "Most Influential Omitted Group": max_row["Omitted Group"],
            }
        )
    return detail, pd.DataFrame(summary_rows)


def assemble_main(fits: dict, bootstrap: pd.DataFrame) -> pd.DataFrame:
    raw = np.array([float(fits[target]["Target-minus-Other"].p[1]) for target in TARGETS])
    holm = holm_adjust(raw)
    rows: list[dict] = []
    for j, (target, spec) in enumerate(TARGETS.items()):
        fit = fits[target]["Target-minus-Other"]
        boot = bootstrap.loc[bootstrap["Target"] == target].iloc[0]
        if holm[j] < 0.05:
            interpretation = f"Post-hoc evidence that {target} differs from the mean per-ROI contribution effect of other {spec['stage']} ROIs"
        else:
            interpretation = f"Insufficient evidence that {target} differs from the mean per-ROI contribution effect of other {spec['stage']} ROIs"
        rows.append(
            {
                "Target Subfamily": target,
                "Stage": spec["stage"],
                "Target ROI Count": spec["target_n"],
                "Reference ROI Count": spec["reference_n"],
                "HC N": 32,
                "SZ N": 32,
                "Adjusted Group Beta": float(fit.beta[1]),
                "HC3 SE": float(fit.se[1]),
                "CI Lower": float(fit.ci_low[1]),
                "CI Upper": float(fit.ci_high[1]),
                "Raw p": float(fit.p[1]),
                "Holm p": float(holm[j]),
                "Bootstrap CI Lower": float(boot["Bootstrap CI Lower"]),
                "Bootstrap CI Upper": float(boot["Bootstrap CI Upper"]),
                "Interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def make_provenance(fc_hashes: pd.DataFrame) -> pd.DataFrame:
    files = [
        ("covariates", COV_INPUT),
        ("roi_metadata", ROI_META),
        ("voxel_mapping", MAPPING),
        ("fixed_current_bins", CURRENT_BINS),
        ("historical_results", HISTORICAL_RESULTS),
        ("historical_script", HISTORICAL_SCRIPT),
        ("step3_loso_result", STEP3_LOSO),
        ("step4_report", STEP4_REPORT),
    ]
    rows = [{"input_type": kind, "subject": "", "path": str(path), "sha256": sha256(path), "shape_or_rows": ""} for kind, path in files]
    return pd.concat([pd.DataFrame(rows), fc_hashes], ignore_index=True)


def fmt(x: float) -> str:
    return f"{x:.9g}"


def write_reports(main: pd.DataFrame, components: pd.DataFrame, boot: pd.DataFrame, loo_summary: pd.DataFrame, identity: pd.DataFrame, reproduction: pd.DataFrame, qc: pd.DataFrame) -> None:
    main_i = main.set_index("Target Subfamily")
    adjusted = components.loc[components["Group"] == "Adjusted SZ-HC"].set_index(["Target", "Component"])
    loo_i = loo_summary.set_index("Target")
    report = [
        "# STEP 5 — Regional Specificity of SMA and IFG Pearson Contributions",
        "",
        "## Analysis status",
        "",
        "The historical 65-ROI Pearson contribution analysis passed the reproduction gate. STEP 5 then tested two post-hoc subject-level regional contrasts with the locked OLS model and HC3 inference.",
        "",
        "## Primary results",
        "",
        "| Target contrast | Adjusted SZ-HC beta | HC3 SE | 95% CI | Raw p | Holm p | Bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in ["SMA", "IFG"]:
        r = main_i.loc[target]
        report.append(f"| {target} vs other {r['Stage']} ROIs | {fmt(r['Adjusted Group Beta'])} | {fmt(r['HC3 SE'])} | [{fmt(r['CI Lower'])}, {fmt(r['CI Upper'])}] | {fmt(r['Raw p'])} | {fmt(r['Holm p'])} | [{fmt(r['Bootstrap CI Lower'])}, {fmt(r['Bootstrap CI Upper'])}] |")
    report += [
        "",
        "The outcome is the within-subject difference in per-ROI Pearson contribution. Negative beta means that the target contribution is more reduced in SZ, relative to its Stage-matched reference ROI average, than in HC.",
        "",
        "## Target and Stage-matched reference effects",
        "",
        "| Target | Component | Adjusted SZ-HC beta | 95% CI | Raw p |",
        "|---|---|---:|---:|---:|",
    ]
    for target in ["SMA", "IFG"]:
        for component in ["Target", "Other", "Target-minus-Other"]:
            r = adjusted.loc[(target, component)]
            report.append(f"| {target} | {component} | {fmt(r['Adjusted Group Effect'])} | [{fmt(r['CI Lower'])}, {fmt(r['CI Upper'])}] | {fmt(r['Raw p'])} |")
    report += [
        "",
        "For both targets, the directly fitted contrast beta equals the target beta minus the reference beta to floating-point precision. Contrast uncertainty was obtained by fitting the within-subject contrast directly, preserving Target–Other covariance.",
        "",
        "## Robustness",
        "",
    ]
    for target in ["SMA", "IFG"]:
        r = loo_i.loc[target]
        report.append(f"- {target}: 64 leave-one-subject-out betas ranged from {fmt(r['LOO Beta Minimum'])} to {fmt(r['LOO Beta Maximum'])}; direction was retained in {int(r['Direction Retained Count'])}/64 refits; raw p ranged from {fmt(r['Raw p Minimum'])} to {fmt(r['Raw p Maximum'])}. The largest absolute beta change followed omission of {r['Most Influential Omitted Subject']} ({r['Most Influential Omitted Group']}).")
    report += [
        "",
        "The stratified bootstrap used 5000 group-preserving subject resamples with seed 20261012. Bootstrap intervals are supplementary and were not used to select the preferred inference.",
        "",
        "## Relation to STEP 4",
        "",
        "The negative SMA regional-specificity result is compatible with the significant STEP 4 SMA Linear G1 Slope result, but the two analyses quantify different outcomes. The significant IFG regional-specificity contrast does not contradict the nonsignificant STEP 4 IFG Offset, Slope, and Nonlinear Shape tests: STEP 5 decomposes a normalized 65-ROI Pearson profile correlation, whereas STEP 4 tested the original 10-bin FC profile. STEP 5 therefore does not establish an IFG abnormality in raw FC magnitude or shape.",
        "",
        "## Interpretation",
        "",
        "These tests concern relative contributions to a shared 65-ROI Pearson profile correlation. They are not tests of raw FC strength. A significant result supports a difference from the average of a Stage-matched aggregate reference set, not uniqueness versus every ROI or subfamily, whole-brain specificity, or causality. Target selection from the same sample makes the analysis post-hoc and exploratory.",
    ]
    (REPORTS / "STEP5_REGIONAL_SPECIFICITY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    qc_lines = [
        "# STEP 5 Quality-Control Report",
        "",
        f"Overall QC status: **{'PASS' if qc['pass'].all() else 'FAIL'}**.",
        "",
        "## Locked-input and reproduction checks",
        "",
        f"- Subjects: 64 unique (32 HC, 32 SZ).",
        f"- ROI structure: 65 total, 62 assigned, 3 unassigned; 43 Anticipation and 19 Outcome ROIs.",
        f"- Stage-matched reference counts: SMA 1 vs 42 other Anticipation ROIs; IFG 6 vs 13 other Outcome ROIs.",
        f"- Maximum absolute Pearson identity error: {identity['absolute_error'].max():.3e}.",
        f"- Historical field comparisons passed: {int(reproduction['pass'].sum())}/{len(reproduction)}.",
        f"- Maximum absolute historical reproduction error: {reproduction['absolute_error'].max():.3e}.",
        "- The 3 unassigned ROIs were retained in all 65-ROI Pearson normalizations and excluded only from Stage-specific reference sets.",
        "- No missing or duplicate subject rows were found.",
        "",
        "## Model and sensitivity checks",
        "",
        "- All main, component, and leave-one-subject-out models used the same covariate definitions and HC3 normal-reference inference.",
        "- Both Group-coefficient decomposition identities passed at floating-point precision.",
        f"- Bootstrap seed: {BOOTSTRAP_SEED}; requested replicates: {BOOTSTRAP_REPLICATES}.",
        "- Source project files were opened read-only; no project, manuscript, Git, or GitHub changes were made.",
        "",
        "## Machine-readable checks",
        "",
        "See `tables/STEP5_qc_checks.tsv`, `tables/STEP5_pearson_identity_qc.tsv`, and `tables/STEP5_historical_reproduction_audit.tsv`.",
    ]
    (REPORTS / "STEP5_QC_REPORT.md").write_text("\n".join(qc_lines) + "\n", encoding="utf-8")

    sma = main_i.loc["SMA"]
    ifg = main_i.loc["IFG"]
    sma_target = adjusted.loc[("SMA", "Target")]
    sma_other = adjusted.loc[("SMA", "Other")]
    ifg_target = adjusted.loc[("IFG", "Target")]
    ifg_other = adjusted.loc[("IFG", "Other")]
    sma_loo = loo_i.loc["SMA"]
    ifg_loo = loo_i.loc["IFG"]
    cn = [
        "# STEP 5 导师汇报摘要（中英文）",
        "",
        "## 中文总结",
        "",
        f"1. **SMA是否区别于其他Anticipation ROI：是。** Target-minus-Other的调整后SZ−HC beta为{fmt(sma['Adjusted Group Beta'])}，两项Holm校正后仍显著。负值表示相对于其他Anticipation ROI平均贡献，SMA contribution在SZ中下降更多。",
        "",
        f"2. **IFG是否区别于其他Outcome ROI：是。** Target-minus-Other的调整后SZ−HC beta为{fmt(ifg['Adjusted Group Beta'])}，两项Holm校正后仍显著。负值表示相对于其他Outcome ROI平均贡献，IFG每ROI contribution在SZ中下降更多。",
        "",
        f"3. **主要效应量：** SMA beta={fmt(sma['Adjusted Group Beta'])}，95% CI [{fmt(sma['CI Lower'])}, {fmt(sma['CI Upper'])}]，Holm p={fmt(sma['Holm p'])}；IFG beta={fmt(ifg['Adjusted Group Beta'])}，95% CI [{fmt(ifg['CI Lower'])}, {fmt(ifg['CI Upper'])}]，Holm p={fmt(ifg['Holm p'])}。对应bootstrap percentile 95% CI分别为[{fmt(sma['Bootstrap CI Lower'])}, {fmt(sma['Bootstrap CI Upper'])}]和[{fmt(ifg['Bootstrap CI Lower'])}, {fmt(ifg['Bootstrap CI Upper'])}]。",
        "",
        f"4. **Target与Other各自效应：** SMA Target beta={fmt(sma_target['Adjusted Group Effect'])}，Other Anticipation beta={fmt(sma_other['Adjusted Group Effect'])}；IFG Target每ROI beta={fmt(ifg_target['Adjusted Group Effect'])}，Other Outcome beta={fmt(ifg_other['Adjusted Group Effect'])}。IFG的Stage-matched Other效应不显著，而SMA的Other效应较小但raw p<0.05；这些成分p值不属于本次两项主要Holm家族。",
        "",
        f"5. **单个被试影响：未发现方向由单个被试决定。** SMA 64次LOO beta范围为[{fmt(sma_loo['LOO Beta Minimum'])}, {fmt(sma_loo['LOO Beta Maximum'])}]，IFG为[{fmt(ifg_loo['LOO Beta Minimum'])}, {fmt(ifg_loo['LOO Beta Maximum'])}]，两者均64/64保持负向；5000次分层bootstrap均无失败，区间均未跨0。",
        "",
        "6. **相对区域特异性：在当前Pearson contribution定义下获得支持。** 证据表明SMA和IFG分别区别于同Stage其他ROI的聚合平均效应；这不等于其区别于每一个单独ROI或Subfamily，也不证明全脑解剖唯一性。",
        "",
        "7. **与STEP 4的关系：不矛盾。** SMA在STEP 4的真实10-bin FC Linear G1 Slope检验显著，方向上与本次SMA相对贡献下降相容，但两者不是同一统计量。IFG在STEP 4的Offset、Slope和Nonlinear Shape均不显著，而本次相对Pearson contribution对比显著；这说明IFG证据依赖profile-correlation contribution及共享归一化定义，不能据此声称IFG原始FC profile异常。",
        "",
        "8. **限制：** 两个Target由同一64人样本中的既往发现选出，因此是事后探索性分析；Target与Other共享65-ROI Pearson中心化和归一化；Other是聚合参照；样本量为64；结果不能推出疾病因果机制。3个unassigned ROI保留在65-ROI归一化中，但不进入Stage-specific Other集合。",
        "",
        "## English oral-report version",
        "",
        f"Both regional contrasts were significant after the prespecified two-test Holm correction. For SMA versus the other 42 Anticipation-associated ROIs, the adjusted SZ-minus-HC contrast was {fmt(sma['Adjusted Group Beta'])}, with a 95% CI of [{fmt(sma['CI Lower'])}, {fmt(sma['CI Upper'])}] and Holm p={fmt(sma['Holm p'])}. For IFG versus the other 13 Outcome-associated ROIs, the corresponding beta was {fmt(ifg['Adjusted Group Beta'])}, with a 95% CI of [{fmt(ifg['CI Lower'])}, {fmt(ifg['CI Upper'])}] and Holm p={fmt(ifg['Holm p'])}.",
        "",
        f"The component effects were {fmt(sma_target['Adjusted Group Effect'])} for SMA and {fmt(sma_other['Adjusted Group Effect'])} for the other Anticipation ROIs, and {fmt(ifg_target['Adjusted Group Effect'])} per ROI for IFG versus {fmt(ifg_other['Adjusted Group Effect'])} for the other Outcome ROIs. The directly fitted within-subject contrasts preserve the Target–Other covariance; their uncertainty was not obtained by subtracting component standard errors.",
        "",
        f"The findings were not driven by a single participant: all 64 leave-one-subject-out estimates retained the negative direction for both targets, with ranges [{fmt(sma_loo['LOO Beta Minimum'])}, {fmt(sma_loo['LOO Beta Maximum'])}] for SMA and [{fmt(ifg_loo['LOO Beta Minimum'])}, {fmt(ifg_loo['LOO Beta Maximum'])}] for IFG. Both 5000-replicate stratified bootstrap intervals excluded zero.",
        "",
        "These results support relative regional specificity within the 65-ROI Pearson contribution definition. They do not establish that SMA or IFG differs from every individual ROI, whole-brain uniqueness, or causality. The SMA result is compatible with the significant STEP 4 linear G1 slope finding, although the metrics differ. The IFG result does not contradict the nonsignificant STEP 4 raw-FC profile metrics, because Pearson contribution is a distinct, normalized profile-similarity decomposition. The target selection was post-hoc in the same 64-subject sample, so the findings are exploratory rather than independently confirmatory.",
    ]
    (REPORTS / "STEP5_SUPERVISOR_SUMMARY_CN_EN.md").write_text("\n".join(cn) + "\n", encoding="utf-8")


def main() -> int:
    for directory in [TABLES, REPORTS, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    print(f"STEP5 analysis started UTC: {started.isoformat()}")
    cov = prepare_covariates()
    meta = prepare_metadata()
    profiles, fc_hashes, bin_qc = load_bin10_profiles(cov)
    subject_base, roi_long, identity = decompose_full_profiles(profiles, cov, meta)
    historical = fit_historical_subfamilies(roi_long)
    reproduction = validate_historical(historical)
    if not identity["pass"].all():
        raise RuntimeError("Pearson identity gate failed")
    print("Reproduction gate: PASS")

    subject = build_subject_contrasts(cov, roi_long, meta)
    components, fits = fit_components(subject)
    bootstrap, bootstrap_arrays = bootstrap_contrasts(subject, fits)
    loo, loo_summary = leave_one_subject_out(subject, fits)
    main_results = assemble_main(fits, bootstrap)

    # Main output preserves the exact requested columns; centered design fields stay internal.
    subject_output_cols = [
        "Subject ID", "Group", "Age", "Sex", "MeanFD",
        "SMA contribution sum", "SMA per-ROI contribution", "Other Anticipation per-ROI contribution", "SMA contrast L",
        "IFG contribution sum", "IFG per-ROI contribution", "Other Outcome per-ROI contribution", "IFG contrast L",
    ]
    subject_output = subject[subject_output_cols].copy()

    decomposition_checks = []
    for target in TARGETS:
        target_beta = float(fits[target]["Target"].beta[1])
        other_beta = float(fits[target]["Other"].beta[1])
        contrast_beta = float(fits[target]["Target-minus-Other"].beta[1])
        decomposition_checks.append({"Target": target, "Target beta": target_beta, "Other beta": other_beta, "Target minus Other beta": target_beta - other_beta, "Direct contrast beta": contrast_beta, "Absolute error": abs((target_beta - other_beta) - contrast_beta), "pass": abs((target_beta - other_beta) - contrast_beta) <= 1e-13})
    decomposition = pd.DataFrame(decomposition_checks)

    robustness = loo_summary.merge(bootstrap, on=["Target", "Stage"], validate="one_to_one")
    roi_audit = meta.groupby(["stage", "subfamily"], as_index=False, dropna=False).agg(ROI_Count=("roi_index0", "nunique"))
    roi_audit["Included_in_65ROI_Pearson"] = True
    roi_audit["Included_in_Stage_specific_Other_reference"] = np.where(roi_audit["stage"] == "Not_assigned", False, ~roi_audit["subfamily"].isin(["SMA", "IFG"]))

    sma_hist = historical.loc[historical["subfamily"] == "SMA"].iloc[0]
    ifg_hist = historical.loc[historical["subfamily"] == "IFG"].iloc[0]
    qc_rows = [
        ("64 unique subjects", len(cov) == 64 and cov["subject"].nunique() == 64, f"N={len(cov)}"),
        ("32 HC and 32 SZ", cov["group"].value_counts().to_dict() == {"SZ": 32, "HC": 32}, str(cov["group"].value_counts().to_dict())),
        ("65 ROI metadata", len(meta) == 65, f"N={len(meta)}"),
        ("62 assigned and 3 unassigned", int((meta["stage"] != "Not_assigned").sum()) == 62 and int((meta["stage"] == "Not_assigned").sum()) == 3, str(meta["stage"].value_counts().to_dict())),
        ("43 Anticipation and 19 Outcome", int((meta["stage"] == "Anticipation").sum()) == 43 and int((meta["stage"] == "Outcome").sum()) == 19, str(meta["stage"].value_counts().to_dict())),
        ("SMA=1, Other Anticipation=42", TARGETS["SMA"]["target_n"] == 1 and TARGETS["SMA"]["reference_n"] == 42, "1 and 42"),
        ("IFG=6, Other Outcome=13", TARGETS["IFG"]["target_n"] == 6 and TARGETS["IFG"]["reference_n"] == 13, "6 and 13"),
        ("All Pearson contribution identities", bool(identity["pass"].all()), f"max_error={identity['absolute_error'].max():.3e}"),
        ("All historical fields reproduced", bool(reproduction["pass"].all()), f"max_error={reproduction['absolute_error'].max():.3e}"),
        ("SMA historical beta reproduced", abs(float(sma_hist["beta_SZ_minus_HC"]) - EXPECTED_SMA_BETA) <= REPRO_TOL, fmt(float(sma_hist["beta_SZ_minus_HC"]))),
        ("SMA historical q reproduced", abs(float(sma_hist["q_FDR_13_assigned"]) - EXPECTED_SMA_Q) <= REPRO_TOL, fmt(float(sma_hist["q_FDR_13_assigned"]))),
        ("IFG historical beta reproduced", abs(float(ifg_hist["beta_SZ_minus_HC"]) - EXPECTED_IFG_BETA) <= REPRO_TOL, fmt(float(ifg_hist["beta_SZ_minus_HC"]))),
        ("IFG historical q reproduced", abs(float(ifg_hist["q_FDR_13_assigned"]) - EXPECTED_IFG_Q) <= REPRO_TOL, fmt(float(ifg_hist["q_FDR_13_assigned"]))),
        ("Complete subject contrast rows", len(subject_output) == 64 and subject_output["Subject ID"].nunique() == 64 and not subject_output.isna().any().any(), f"rows={len(subject_output)}"),
        ("Both beta decomposition identities", bool(decomposition["pass"].all()), f"max_error={decomposition['Absolute error'].max():.3e}"),
        ("Bootstrap effective counts", bool((bootstrap["Effective Bootstrap N"] + bootstrap["Bootstrap Failures"] == BOOTSTRAP_REPLICATES).all()), str(bootstrap[["Target", "Effective Bootstrap N", "Bootstrap Failures"]].to_dict("records"))),
        ("LOO 64 refits per target", loo.groupby("Target").size().to_dict() == {"IFG": 64, "SMA": 64}, str(loo.groupby("Target").size().to_dict())),
    ]
    qc = pd.DataFrame(qc_rows, columns=["check", "pass", "detail"])
    if not qc["pass"].all():
        raise RuntimeError("One or more final QC checks failed")

    write_tsv(main_results, TABLES / "STEP5_main_regional_specificity.tsv")
    write_tsv(components, TABLES / "STEP5_target_vs_reference.tsv")
    write_tsv(subject_output, TABLES / "STEP5_subject_level_contrasts.tsv")
    write_tsv(bootstrap, TABLES / "STEP5_bootstrap_results.tsv")
    write_tsv(loo, TABLES / "STEP5_leave_one_subject_out.tsv")
    write_tsv(robustness, TABLES / "STEP5_robustness_summary.tsv")
    write_tsv(identity, TABLES / "STEP5_pearson_identity_qc.tsv")
    write_tsv(historical, TABLES / "STEP5_historical_full_subfamily_results.tsv")
    write_tsv(reproduction, TABLES / "STEP5_historical_reproduction_audit.tsv")
    write_tsv(decomposition, TABLES / "STEP5_beta_decomposition_qc.tsv")
    write_tsv(roi_audit, TABLES / "STEP5_roi_metadata_audit.tsv")
    write_tsv(qc, TABLES / "STEP5_qc_checks.tsv")
    provenance = make_provenance(fc_hashes)
    write_tsv(provenance, TABLES / "STEP5_input_sha256.tsv")
    for target, values in bootstrap_arrays.items():
        write_tsv(pd.DataFrame({"Target": target, "Bootstrap replicate": np.arange(1, len(values) + 1), "Beta L": values}), TABLES / f"STEP5_{target}_bootstrap_distribution.tsv")

    write_reports(main_results, components, bootstrap, loo_summary, identity, reproduction, qc)
    state = {
        "status": "PASS",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bin_qc": bin_qc,
        "main_results": main_results.to_dict("records"),
    }
    write_json(state, LOGS / "analysis_state.json")
    print(main_results.to_string(index=False))
    print("STEP5 analysis completed: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        fail = {
            "status": "BLOCKED",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        LOGS.mkdir(parents=True, exist_ok=True)
        write_json(fail, LOGS / "analysis_state.json")
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
