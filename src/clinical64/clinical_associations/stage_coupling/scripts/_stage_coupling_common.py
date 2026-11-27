from __future__ import annotations

import os
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, rankdata, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ["STRIATAL_OUTPUT_ROOT"])
PROJECT_ROOT = Path(os.environ["STRIATAL_PROJECT_ROOT"])

BRAIN_BEHAVIOR = (
    PROJECT_ROOT
    / "CURRENT64_CANONICAL_RERUN"
    / "10_BRAIN_BEHAVIOR"
    / "brain_behavior_subject_level.tsv"
)
INDIVIDUAL_SLOPES = (
    PROJECT_ROOT
    / "CURRENT64_CANONICAL_RERUN"
    / "09_INDIVIDUAL_SLOPES"
    / "subject_stage_slopes_wide.tsv"
)
FROZEN_BEHAVIOR = (
    PROJECT_ROOT
    / "CURRENT64_CANONICAL_RERUN"
    / "01_FROZEN_INPUTS"
    / "behavior"
    / "current64_behavior_final_scores.tsv"
)
BIN10_STAGE = (
    PROJECT_ROOT
    / "journal_figure_scatter_revision_20261028"
    / "source_data"
    / "B05_Bin10_stage_subjects_used.tsv"
)
CLINICAL_IMAGING = (
    PROJECT_ROOT
    / "clinical_extension_analysis"
    / "data_internal"
    / "clinical_imaging_analysis_dataset.tsv"
)
ROI_BEHAVIOR = (
    PROJECT_ROOT
    / "clinical_extension_analysis"
    / "roi_behavior_extension"
    / "data_internal"
    / "roi_behavior_matched_dataset.tsv"
)
ROI_OVERLAP_AUDIT = (
    PROJECT_ROOT
    / "clinical_extension_analysis"
    / "roi_behavior_extension"
    / "sample_overlap_audit.csv"
)
RAW_ITEM_TABLE = Path(
    os.path.join(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"], 'final64_gradient_clinical_stats/03_clinical_master/final64_clinical_for_server_v2.tsv')
)

SUBJECT_DATA = OUT_DIR / "stage_coupling_subject_level_data.csv"
RESULTS = OUT_DIR / "stage_coupling_results.csv"
JOINT_RESULTS = OUT_DIR / "joint_stage_model_results.csv"
BAYES_RESULTS = OUT_DIR / "bayesian_stage_coupling_results.csv"
LOO_RESULTS = OUT_DIR / "leave_one_out_results.csv"
BOOT_RESULTS = OUT_DIR / "bootstrap_results.csv"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261029
BAYES_GRID_POINTS = 40_001

ENDPOINTS = {
    "overall_fc_slope": {
        "label": "Overall FC slope stage contrast",
        "y": "NeuralSlope_stage_contrast",
        "raw_y": "NeuralSlope_raw_difference",
        "ant": "Ant_slope_z",
        "out": "Out_slope_z",
        "raw_ant": "Anticipation_slope",
        "raw_out": "Outcome_slope",
        "hierarchy": "primary",
    },
    "bin10_similarity": {
        "label": "Bin 10 HC-reference similarity stage contrast",
        "y": "Bin10Similarity_stage_contrast",
        "raw_y": "Bin10Similarity_raw_difference",
        "ant": "Bin10_Ant_similarity_z",
        "out": "Bin10_Out_similarity_z",
        "raw_ant": "Bin10_Ant_similarity_fisher_z",
        "raw_out": "Bin10_Out_similarity_fisher_z",
        "hierarchy": "secondary",
    },
}


def ensure_output_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    (OUT_DIR / "logs").mkdir(exist_ok=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> pd.DataFrame:
    require(path.is_file(), f"Required input not found: {path}")
    return pd.read_csv(path, sep="\t")


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    sd = values.std(ddof=1)
    require(np.isfinite(sd) and sd > 0, f"Cannot z-standardize {values.name}: invalid SD")
    return (values - values.mean()) / sd


def spearman_pair(x: Iterable[float], y: Iterable[float]) -> tuple[int, float, float]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    result = spearmanr(frame["x"].to_numpy(), frame["y"].to_numpy())
    return len(frame), float(result.statistic), float(result.pvalue)


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    require(len(x) == len(y) and len(x) >= 4, "Bootstrap inputs are invalid")
    rng = np.random.default_rng(seed)
    n = len(x)
    observed = float(spearmanr(x, y).statistic)
    draws = np.empty(n_resamples, dtype=float)
    draws.fill(np.nan)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        draws[i] = float(spearmanr(x[idx], y[idx]).statistic)
    valid = draws[np.isfinite(draws)]
    require(len(valid) >= int(0.99 * n_resamples), "Too many invalid bootstrap replicates")
    low, high = np.quantile(valid, [0.025, 0.975])
    same_sign = np.mean(valid > 0) if observed >= 0 else np.mean(valid < 0)
    return {
        "observed_rho": observed,
        "bootstrap_median": float(np.median(valid)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_bootstrap_gt_zero": float(np.mean(valid > 0)),
        "p_bootstrap_lt_zero": float(np.mean(valid < 0)),
        "sign_consistency": float(same_sign),
        "requested_resamples": int(n_resamples),
        "valid_resamples": int(len(valid)),
        "seed": int(seed),
    }


def rank_normal_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranks = rankdata(values, method="average")
    u = (ranks - 0.5) / len(values)
    return norm.ppf(u)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: Iterable[float]) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights)
    cdf /= cdf[-1]
    return np.interp(np.asarray(list(probabilities), dtype=float), cdf, sorted_values)


def bayesian_rank_correlation_grid(
    x: np.ndarray,
    y: np.ndarray,
    eta: float = 1.0,
    grid_points: int = BAYES_GRID_POINTS,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Gaussian-copula posterior for a Spearman-type correlation.

    Average ranks are mapped to normal scores.  The latent bivariate-normal
    correlation receives an LKJ(eta) prior.  Deterministic grid quadrature is
    used, and latent correlation is transformed to population Spearman rho.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    require(len(x) == len(y) and len(x) >= 4, "Bayesian inputs are invalid")
    zx = rank_normal_scores(x)
    zy = rank_normal_scores(y)
    latent_rho = np.linspace(-0.9995, 0.9995, int(grid_points))
    one_minus = 1.0 - latent_rho**2
    sxx = float(np.sum(zx**2))
    syy = float(np.sum(zy**2))
    sxy = float(np.sum(zx * zy))
    n = len(zx)
    log_likelihood = (
        -0.5 * n * np.log(one_minus)
        - 0.5 * (sxx + syy - 2.0 * latent_rho * sxy) / one_minus
    )
    log_prior = (float(eta) - 1.0) * np.log(one_minus)
    log_posterior = log_likelihood + log_prior
    weights = np.exp(log_posterior - logsumexp(log_posterior))
    spearman_rho = (6.0 / np.pi) * np.arcsin(latent_rho / 2.0)
    q025, median, q975 = weighted_quantile(
        spearman_rho, weights, [0.025, 0.5, 0.975]
    )
    endpoint_mass = float(weights[:10].sum() + weights[-10:].sum())
    summary = {
        "n": int(n),
        "prior": f"LKJ(eta={eta:g}) on latent Gaussian-copula correlation",
        "eta": float(eta),
        "posterior_median_r": float(median),
        "credible_interval_95_low": float(q025),
        "credible_interval_95_high": float(q975),
        "p_r_gt_zero": float(weights[spearman_rho > 0].sum()),
        "p_r_lt_zero": float(weights[spearman_rho < 0].sum()),
        "p_abs_r_gt_0_10": float(weights[np.abs(spearman_rho) > 0.10].sum()),
        "p_abs_r_gt_0_20": float(weights[np.abs(spearman_rho) > 0.20].sum()),
        "grid_points": int(grid_points),
        "endpoint_mass": endpoint_mass,
        "convergence": "deterministic grid quadrature; posterior normalized to 1",
        "bf10": np.nan,
        "bf_note": "Not reported: no separate point-null model was fit.",
    }
    density_latent = weights / (latent_rho[1] - latent_rho[0])
    derivative = (3.0 / np.pi) / np.sqrt(1.0 - latent_rho**2 / 4.0)
    density_spearman = density_latent / derivative
    grid = pd.DataFrame(
        {
            "latent_rho": latent_rho,
            "spearman_rho": spearman_rho,
            "posterior_weight": weights,
            "posterior_density": density_spearman,
        }
    )
    return summary, grid


def load_subject_data(eligible_only: bool = False) -> pd.DataFrame:
    require(SUBJECT_DATA.is_file(), "Run 02_build_stage_contrasts.py first")
    data = pd.read_csv(SUBJECT_DATA)
    if eligible_only:
        data = data.loc[data["eligible_stage_coupling"].astype(bool)].copy()
    return data


def distribution_summary(data: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for variable in variables:
        x = pd.to_numeric(data[variable], errors="coerce").dropna()
        median = float(x.median())
        mad = float(np.median(np.abs(x - median)))
        robust_z = np.zeros(len(x)) if mad == 0 else 0.67448975 * (x.to_numpy() - median) / mad
        counts = x.value_counts()
        rows.append(
            {
                "variable": variable,
                "n": int(len(x)),
                "mean": float(x.mean()),
                "sd": float(x.std(ddof=1)),
                "median": median,
                "min": float(x.min()),
                "max": float(x.max()),
                "skew": float(x.skew()),
                "unique_values": int(x.nunique()),
                "n_values_in_ties": int(counts[counts > 1].sum()),
                "max_abs_robust_z": float(np.max(np.abs(robust_z))),
                "n_abs_robust_z_gt_3_5": int(np.sum(np.abs(robust_z) > 3.5)),
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def format_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"
