"""STEP 3D: equal-subfamily-weighting sensitivity analysis.

Reads frozen Definition A / Current64 inputs without modifying the source project.
All outputs are written to the independent teacher-report directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
import warnings

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

CANONICAL_ROOT = Path(
    os.path.join(os.environ["STRIATAL_PROJECT_ROOT"], 'CURRENT64_CANONICAL_RERUN')
)
VENDOR = CANONICAL_ROOT / ".python_packages"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import statsmodels
import statsmodels.formula.api as smf


OUTPUT = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP3_Equal_Subfamily_Weighting'))
SOURCE_PROJECT = Path(os.environ["STRIATAL_RELEASE_ROOT"])
PREVIOUS_STEP3A = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP3_ROI_robustness'))

ROI_LONG = CANONICAL_ROOT / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_FC_long.tsv"
STAGE_SAVED = CANONICAL_ROOT / "06_REWARD_STAGE" / "subject_bin_stage_FC_long.tsv"
ROI_META = (
    CANONICAL_ROOT
    / "01_FROZEN_INPUTS"
    / "reward_stage_mapping"
    / "roi_metadata_stage_subfamily_v2.tsv"
)
COVARIATES = (
    CANONICAL_ROOT / "01_FROZEN_INPUTS" / "covariates" / "current64_covariates.tsv"
)
SUBJECT_TABLE = (
    CANONICAL_ROOT / "01_FROZEN_INPUTS" / "subject_table" / "current64_subject_table.tsv"
)
DEFINITION_A = (
    CANONICAL_ROOT
    / "01_FROZEN_INPUTS"
    / "canonical_bins"
    / "fixed_375_G1_voxel_bins_2608.tsv"
)
HISTORICAL_BUILD_SCRIPT = (
    SOURCE_PROJECT / "src" / "clinical64" / "reward_stage" / "freeze_and_build_profiles.py"
)
HISTORICAL_MODEL_SCRIPT = (
    SOURCE_PROJECT / "src" / "clinical64" / "reward_stage" / "run_stage_analyses.py"
)
HISTORICAL_ADJUSTED_COEFF = (
    CANONICAL_ROOT / "07_STAGE_GRADIENT_MODELS" / "model_B_adjusted_coefficients.tsv"
)
PREVIOUS_EFFECT_TABLE = PREVIOUS_STEP3A / "effect_size_stability.tsv"

FORMULA = "FC ~ group * stage * G1_centered + age_c + sex + meanFD_c"
TERM = "group[T.SZ]:stage[T.Anticipation]:G1_centered"
MODEL_ORDER = ["A", "B", "C", "D"]
MODEL_META = {
    "A": {
        "label": "Full ROI, original ROI-equal aggregation",
        "short_label": "A Full / ROI-equal",
        "weighting": "ROI_equal",
        "minus_insula": False,
    },
    "B": {
        "label": "Minus Insula, original ROI-equal aggregation",
        "short_label": "B Minus Insula / ROI-equal",
        "weighting": "ROI_equal",
        "minus_insula": True,
    },
    "C": {
        "label": "Full ROI, equal-subfamily aggregation",
        "short_label": "C Full / subfamily-equal",
        "weighting": "Subfamily_equal",
        "minus_insula": False,
    },
    "D": {
        "label": "Minus Insula, equal-subfamily aggregation",
        "short_label": "D Minus Insula / subfamily-equal",
        "weighting": "Subfamily_equal",
        "minus_insula": True,
    },
}

TARGET_FULL = float(os.environ["STRIATAL_TARGET_FULL"])
TARGET_MINUS_INSULA = float(os.environ["STRIATAL_TARGET_MINUS_INSULA"])
GATE_TOLERANCE = 5e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_tsv(frame: pd.DataFrame, name: str, na_rep: str = "N/A") -> None:
    frame.to_csv(
        OUTPUT / name,
        sep="\t",
        index=False,
        float_format="%.17g",
        na_rep=na_rep,
        lineterminator="\n",
    )


def write_json(value: object, name: str) -> None:
    (OUTPUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_logging() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT / "analysis_execution.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def load_and_validate() -> dict[str, pd.DataFrame]:
    required = [
        ROI_LONG,
        STAGE_SAVED,
        ROI_META,
        COVARIATES,
        SUBJECT_TABLE,
        DEFINITION_A,
        HISTORICAL_BUILD_SCRIPT,
        HISTORICAL_MODEL_SCRIPT,
        HISTORICAL_ADJUSTED_COEFF,
        PREVIOUS_EFFECT_TABLE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "Missing required inputs:\n" + "\n".join(missing))

    data = {
        "roi": pd.read_csv(ROI_LONG, sep="\t"),
        "stage_saved": pd.read_csv(STAGE_SAVED, sep="\t"),
        "meta": pd.read_csv(ROI_META, sep="\t"),
        "cov": pd.read_csv(COVARIATES, sep="\t"),
        "subjects": pd.read_csv(SUBJECT_TABLE, sep="\t"),
        "bins": pd.read_csv(DEFINITION_A, sep="\t"),
    }

    roi = data["roi"]
    stage_saved = data["stage_saved"]
    meta = data["meta"]
    cov = data["cov"]
    subjects = data["subjects"]
    bins = data["bins"]

    require(roi.shape == (41600, 7), f"Unexpected ROI table shape: {roi.shape}")
    require(stage_saved.shape == (1280, 9), f"Unexpected saved stage table: {stage_saved.shape}")
    require(meta.shape == (65, 6), f"Unexpected ROI metadata: {meta.shape}")
    require(cov.shape == (64, 5), f"Unexpected covariates: {cov.shape}")
    require(subjects.shape == (64, 2), f"Unexpected subject table: {subjects.shape}")
    require(bins.shape[0] == 2608, "Definition A does not have 2,608 voxels")
    require(set(bins["bin"].astype(int)) == set(range(1, 11)), "Definition A bins are not 1-10")
    expected_bin_counts = [260, 260, 261, 261, 261, 261, 261, 261, 261, 261]
    actual_bin_counts = bins.groupby("bin").size().reindex(range(1, 11)).tolist()
    require(actual_bin_counts == expected_bin_counts, f"Definition A bin counts changed: {actual_bin_counts}")
    require(not roi.isna().any().any(), "ROI-level table contains missing values")
    require(not roi.duplicated(["Subject", "Bin", "ROI"]).any(), "Duplicate Subject-Bin-ROI rows")
    require(subjects["Group"].value_counts().to_dict() == {"HC": 32, "SZ": 32}, "Expected 32 HC/32 SZ")
    require(set(roi["Subject"]) == set(subjects["Subject"]), "ROI and subject IDs differ")
    require(set(cov["Subject"]) == set(subjects["Subject"]), "Covariate and subject IDs differ")

    assigned = meta.loc[
        meta["include_two_family"].astype(int).eq(1)
        & meta["stage"].isin(["Anticipation", "Outcome"])
    ].copy()
    require(len(assigned) == 62, "Assigned ROI count is not 62")
    stage_counts = assigned.groupby("stage").size().to_dict()
    family_counts = assigned.groupby("stage")["subfamily"].nunique().to_dict()
    require(stage_counts == {"Anticipation": 43, "Outcome": 19}, f"Stage ROI counts differ: {stage_counts}")
    require(family_counts == {"Anticipation": 6, "Outcome": 7}, f"Stage family counts differ: {family_counts}")
    require(int((meta["include_two_family"].astype(int) == 0).sum()) == 3, "Unassigned ROI count is not 3")

    build_text = HISTORICAL_BUILD_SCRIPT.read_text(encoding="utf-8")
    code_mean_evidence = (
        '"FC": float(np.mean(bin_roi[bin_index, stage_mask]))' in build_text
        and 'stage_mask = roi_metadata["stage"].to_numpy() == stage' in build_text
    )
    require(code_mean_evidence, "Historical aggregation code no longer shows direct ROI arithmetic mean")

    logging.info(
        "Frozen audit passed: 64 subjects, 62 assigned ROI, Anticipation 43/6 families, Outcome 19/7 families"
    )
    return data


def selected_roi_rows(roi: pd.DataFrame, minus_insula: bool) -> pd.DataFrame:
    selected = roi.loc[roi["RewardStage"].isin(["Outcome", "Anticipation"])].copy()
    if minus_insula:
        selected = selected.loc[selected["Subfamily"].ne("Insula")].copy()
    return selected


def build_stage_table(
    roi: pd.DataFrame,
    cov: pd.DataFrame,
    weighting: str,
    minus_insula: bool,
) -> pd.DataFrame:
    selected = selected_roi_rows(roi, minus_insula)
    keys = ["Subject", "Group", "Bin", "RewardStage"]
    if weighting == "ROI_equal":
        stage = selected.groupby(keys, observed=True, as_index=False)["FC"].mean()
    elif weighting == "Subfamily_equal":
        subfamily = (
            selected.groupby(keys + ["Subfamily"], observed=True, as_index=False)["FC"]
            .mean()
        )
        stage = subfamily.groupby(keys, observed=True, as_index=False)["FC"].mean()
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    stage = stage.rename(columns={"RewardStage": "Stage"})
    stage = stage.merge(cov, on=["Subject", "Group"], how="left", validate="many_to_one")
    stage["G1_bin_center"] = stage["Bin"].astype(float) - 5.5
    stage = stage[
        ["Subject", "Group", "Bin", "G1_bin_center", "Stage", "FC", "Age", "Sex", "MeanFD"]
    ].sort_values(["Subject", "Stage", "Bin"]).reset_index(drop=True)
    require(len(stage) == 1280, f"Stage table has {len(stage)} rather than 1,280 rows")
    require(not stage.isna().any().any(), "Stage table contains missing values")
    require(not stage.duplicated(["Subject", "Stage", "Bin"]).any(), "Duplicate stage rows")
    return stage


def compare_saved_stage(rebuilt: pd.DataFrame, saved: pd.DataFrame) -> float:
    keys = ["Subject", "Group", "Bin", "G1_bin_center", "Stage", "Age", "Sex", "MeanFD"]
    joined = saved.merge(rebuilt, on=keys, suffixes=("_saved", "_rebuilt"), validate="one_to_one")
    require(len(joined) == 1280, "Saved/rebuilt stage join lost rows")
    return float(np.max(np.abs(joined["FC_saved"] - joined["FC_rebuilt"])))


def normalize_for_model(stage: pd.DataFrame) -> pd.DataFrame:
    frame = stage.rename(
        columns={
            "Subject": "subject",
            "Group": "group",
            "Stage": "stage",
            "G1_bin_center": "G1_centered",
            "Age": "age",
            "Sex": "sex",
            "MeanFD": "meanFD",
        }
    ).copy()
    frame["group"] = pd.Categorical(frame["group"], categories=["HC", "SZ"])
    frame["stage"] = pd.Categorical(frame["stage"], categories=["Outcome", "Anticipation"])
    frame["sex"] = pd.Categorical(frame["sex"], categories=["Female", "Male"])
    people = frame.drop_duplicates("subject")
    frame["age_c"] = frame["age"] - people["age"].mean()
    frame["meanFD_c"] = frame["meanFD"] - people["meanFD"].mean()
    return frame


def fit_model(model_id: str, stage: pd.DataFrame) -> tuple[object, dict[str, object], list[dict[str, object]]]:
    frame = normalize_for_model(stage)
    model = smf.mixedlm(
        FORMULA,
        data=frame,
        groups=frame["subject"],
        re_formula="1 + G1_centered",
    )
    attempt_rows: list[dict[str, object]] = []
    result = None
    optimizer = ""
    warnings_all = []
    for method in ["lbfgs", "powell", "cg"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                candidate = model.fit(reml=False, method=method, maxiter=5000, disp=False)
                warning_text = " | ".join(str(item.message) for item in caught)
                attempt_rows.append(
                    {
                        "model": model_id,
                        "optimizer": method,
                        "converged": bool(candidate.converged),
                        "warning": warning_text,
                        "error": "",
                    }
                )
                warnings_all.append(f"{method}: converged={candidate.converged}; {warning_text}")
                if bool(candidate.converged):
                    result = candidate
                    optimizer = method
                    break
            except Exception as exc:
                attempt_rows.append(
                    {
                        "model": model_id,
                        "optimizer": method,
                        "converged": False,
                        "warning": "",
                        "error": repr(exc),
                    }
                )
                warnings_all.append(f"{method}: ERROR {exc!r}")
    if result is None:
        raise RuntimeError(f"Model {model_id} failed: {' || '.join(warnings_all)}")

    estimate = float(result.fe_params[TERM])
    se = float(result.bse_fe[TERM])
    z = estimate / se
    cov_re = np.asarray(result.cov_re, dtype=float)
    row = {
        "model": model_id,
        "model_label": MODEL_META[model_id]["label"],
        "short_label": MODEL_META[model_id]["short_label"],
        "weighting": MODEL_META[model_id]["weighting"],
        "minus_insula": MODEL_META[model_id]["minus_insula"],
        "formula": FORMULA,
        "group_reference": "HC",
        "stage_reference": "Outcome",
        "beta_group_stage_g1": estimate,
        "SE": se,
        "CI95_low": estimate - 1.96 * se,
        "CI95_high": estimate + 1.96 * se,
        "z_wald": z,
        "p_wald": float(2 * norm.sf(abs(z))),
        "converged": bool(result.converged),
        "optimizer": optimizer,
        "reml": False,
        "n_subjects": int(frame["subject"].nunique()),
        "n_observations": int(result.nobs),
        "logLik": float(result.llf),
        "AIC": float(result.aic),
        "BIC": float(result.bic),
        "residual_scale": float(result.scale),
        "random_intercept_variance": float(cov_re[0, 0]),
        "random_G1_slope_variance": float(cov_re[1, 1]),
        "random_intercept_G1_covariance": float(cov_re[0, 1]),
        "warnings": " || ".join(warnings_all),
    }
    return result, row, attempt_rows


def coefficient_rows(model_id: str, result: object) -> list[dict[str, object]]:
    rows = []
    for term in result.fe_params.index:
        estimate = float(result.fe_params[term])
        se = float(result.bse_fe[term])
        z = estimate / se
        rows.append(
            {
                "model": model_id,
                "term": term,
                "estimate": estimate,
                "SE": se,
                "CI95_low": estimate - 1.96 * se,
                "CI95_high": estimate + 1.96 * se,
                "z_wald": z,
                "p_wald": float(2 * norm.sf(abs(z))),
            }
        )
    return rows


def aggregation_weights(meta: pd.DataFrame) -> pd.DataFrame:
    assigned = meta.loc[
        meta["include_two_family"].astype(int).eq(1)
        & meta["stage"].isin(["Anticipation", "Outcome"])
    ].copy()
    counts = (
        assigned.groupby(["stage", "subfamily"], as_index=False)
        .agg(roi_count=("roi_index0", "nunique"))
        .sort_values(["stage", "subfamily"])
    )
    rows = []
    for model_id in MODEL_ORDER:
        spec = MODEL_META[model_id]
        for stage in ["Anticipation", "Outcome"]:
            stage_counts = counts.loc[counts["stage"].eq(stage)].copy()
            included = stage_counts["subfamily"].ne("Insula") if spec["minus_insula"] else pd.Series(True, index=stage_counts.index)
            included_roi_total = int(stage_counts.loc[included, "roi_count"].sum())
            included_family_total = int(included.sum())
            for item in stage_counts.itertuples(index=False):
                is_included = not (spec["minus_insula"] and item.subfamily == "Insula")
                if not is_included:
                    family_weight = 0.0
                    per_roi_weight = 0.0
                elif spec["weighting"] == "ROI_equal":
                    family_weight = float(item.roi_count / included_roi_total)
                    per_roi_weight = float(1.0 / included_roi_total)
                else:
                    family_weight = float(1.0 / included_family_total)
                    per_roi_weight = float(1.0 / (included_family_total * item.roi_count))
                rows.append(
                    {
                        "model": model_id,
                        "model_label": spec["label"],
                        "weighting": spec["weighting"],
                        "minus_insula": spec["minus_insula"],
                        "stage": stage,
                        "subfamily": item.subfamily,
                        "roi_count": int(item.roi_count),
                        "included": is_included,
                        "included_stage_roi_count": included_roi_total,
                        "included_stage_subfamily_count": included_family_total,
                        "subfamily_total_weight": family_weight,
                        "per_roi_weight": per_roi_weight,
                    }
                )
    result = pd.DataFrame(rows)
    included = result.loc[result["included"]]
    sums = included.groupby(["model", "stage"])["subfamily_total_weight"].sum()
    require(float(np.max(np.abs(sums.to_numpy() - 1.0))) < 1e-14, "Subfamily weights do not sum to one")
    return result


def build_profiles(stage_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model_id, table in stage_tables.items():
        summary = (
            table.groupby(["Group", "Stage", "Bin", "G1_bin_center"], observed=True)["FC"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_FC", "std": "sd_subject", "count": "n_subjects"})
        )
        summary["SE_subject"] = summary["sd_subject"] / np.sqrt(summary["n_subjects"])
        summary.insert(0, "model", model_id)
        summary.insert(1, "model_label", MODEL_META[model_id]["label"])
        rows.append(summary)
    profiles = pd.concat(rows, ignore_index=True)
    profiles["model"] = pd.Categorical(profiles["model"], categories=MODEL_ORDER, ordered=True)
    profiles = profiles.sort_values(["model", "Stage", "Group", "Bin"]).reset_index(drop=True)
    profiles["model"] = profiles["model"].astype(str)

    hc = profiles.loc[profiles["Group"].eq("HC")].copy()
    sz = profiles.loc[profiles["Group"].eq("SZ")].copy()
    merge_keys = ["model", "model_label", "Stage", "Bin", "G1_bin_center"]
    diff = hc.merge(sz, on=merge_keys, suffixes=("_HC", "_SZ"), validate="one_to_one")
    diff = diff[
        merge_keys
        + ["mean_FC_HC", "SE_subject_HC", "mean_FC_SZ", "SE_subject_SZ"]
    ].copy()
    diff["SZ_minus_HC"] = diff["mean_FC_SZ"] - diff["mean_FC_HC"]
    diff = diff.sort_values(["model", "Stage", "Bin"]).reset_index(drop=True)
    return profiles, diff


def curve_metrics(base: np.ndarray, candidate: np.ndarray, x: np.ndarray) -> dict[str, float]:
    delta = candidate - base
    offset = float(delta.mean())
    centered_delta = delta - offset
    denom = float(np.dot(x, x))
    slope_change = float(np.dot(x, delta) / denom)
    if np.std(base, ddof=1) > 0 and np.std(candidate, ddof=1) > 0:
        correlation = float(np.corrcoef(base, candidate)[0, 1])
    else:
        correlation = np.nan
    return {
        "RMSE": float(np.sqrt(np.mean(delta**2))),
        "signed_offset_change": offset,
        "shape_RMSE_after_offset_removal": float(np.sqrt(np.mean(centered_delta**2))),
        "linear_G1_slope_change": slope_change,
        "maximum_absolute_bin_change": float(np.max(np.abs(delta))),
        "profile_correlation": correlation,
    }


def profile_stability(profiles: pd.DataFrame, differences: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("B_minus_A", "A", "B"),
        ("C_minus_A", "A", "C"),
        ("D_minus_C", "C", "D"),
        ("D_minus_B", "B", "D"),
    ]
    rows = []
    x = np.arange(1, 11, dtype=float) - 5.5
    for comparison, base_model, candidate_model in comparisons:
        for stage in ["Anticipation", "Outcome"]:
            for group in ["HC", "SZ"]:
                base = profiles.loc[
                    profiles["model"].eq(base_model)
                    & profiles["Stage"].eq(stage)
                    & profiles["Group"].eq(group)
                ].sort_values("Bin")["mean_FC"].to_numpy(float)
                candidate = profiles.loc[
                    profiles["model"].eq(candidate_model)
                    & profiles["Stage"].eq(stage)
                    & profiles["Group"].eq(group)
                ].sort_values("Bin")["mean_FC"].to_numpy(float)
                rows.append(
                    {
                        "comparison": comparison,
                        "base_model": base_model,
                        "candidate_model": candidate_model,
                        "stage": stage,
                        "curve_type": f"{group}_FC",
                        **curve_metrics(base, candidate, x),
                    }
                )
            base_diff = differences.loc[
                differences["model"].eq(base_model) & differences["Stage"].eq(stage)
            ].sort_values("Bin")["SZ_minus_HC"].to_numpy(float)
            candidate_diff = differences.loc[
                differences["model"].eq(candidate_model) & differences["Stage"].eq(stage)
            ].sort_values("Bin")["SZ_minus_HC"].to_numpy(float)
            rows.append(
                {
                    "comparison": comparison,
                    "base_model": base_model,
                    "candidate_model": candidate_model,
                    "stage": stage,
                    "curve_type": "SZ_minus_HC",
                    **curve_metrics(base_diff, candidate_diff, x),
                }
            )
    return pd.DataFrame(rows)


def model_deltas(model_results: pd.DataFrame) -> pd.DataFrame:
    beta = model_results.set_index("model")["beta_group_stage_g1"].to_dict()
    rows = []
    for label, base, candidate in [
        ("B_minus_A", "A", "B"),
        ("C_minus_A", "A", "C"),
        ("D_minus_C", "C", "D"),
        ("D_minus_B", "B", "D"),
    ]:
        delta = float(beta[candidate] - beta[base])
        rows.append(
            {
                "comparison": label,
                "base_model": base,
                "candidate_model": candidate,
                "base_beta": beta[base],
                "candidate_beta": beta[candidate],
                "signed_delta_beta": delta,
                "absolute_delta_beta": abs(delta),
            }
        )
    return pd.DataFrame(rows)


def provenance() -> pd.DataFrame:
    paths = [
        ("ROI-level FC profiles", ROI_LONG),
        ("Saved stage table", STAGE_SAVED),
        ("ROI metadata", ROI_META),
        ("Covariates", COVARIATES),
        ("Subject table", SUBJECT_TABLE),
        ("Definition A bins", DEFINITION_A),
        ("Historical stage builder", HISTORICAL_BUILD_SCRIPT),
        ("Historical MixedLM script", HISTORICAL_MODEL_SCRIPT),
        ("Historical adjusted coefficients", HISTORICAL_ADJUSTED_COEFF),
        ("Prior STEP3A effect table", PREVIOUS_EFFECT_TABLE),
    ]
    return pd.DataFrame(
        [
            {
                "input": label,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for label, path in paths
        ]
    )


def write_blocking_report(exc: Exception) -> None:
    text = (
        "# STEP 3D Blocking Report\n\n"
        f"Status: BLOCKED\n\nReason: `{type(exc).__name__}: {exc}`\n\n"
        "The Full and Minus-Insula reproduction gates or a required audit failed. "
        "Equal-subfamily models were not accepted.\n"
    )
    (OUTPUT / "BLOCKING_REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    configure_logging()
    logging.info("STEP 3D Equal-Subfamily-Weighting Sensitivity Analysis started")
    logging.info("Python: %s", sys.version.replace("\n", " "))
    logging.info(
        "Versions: numpy=%s pandas=%s scipy=%s statsmodels=%s",
        np.__version__,
        pd.__version__,
        scipy.__version__,
        statsmodels.__version__,
    )

    data = load_and_validate()
    weights = aggregation_weights(data["meta"])

    # Required gate: only original-weighting Full and Minus-Insula are fit first.
    stage_tables: dict[str, pd.DataFrame] = {
        "A": build_stage_table(data["roi"], data["cov"], "ROI_equal", False),
        "B": build_stage_table(data["roi"], data["cov"], "ROI_equal", True),
    }
    rebuild_error = compare_saved_stage(stage_tables["A"], data["stage_saved"])
    require(rebuild_error < 1e-12, f"Original ROI mean does not reproduce saved stage table: {rebuild_error}")

    results: dict[str, object] = {}
    model_rows = []
    coefficient_table = []
    attempts = []
    for model_id in ["A", "B"]:
        result, row, attempt_rows = fit_model(model_id, stage_tables[model_id])
        results[model_id] = result
        model_rows.append(row)
        coefficient_table.extend(coefficient_rows(model_id, result))
        attempts.extend(attempt_rows)
        logging.info(
            "Gate model %s: beta=%.17g SE=%.17g p=%.17g optimizer=%s converged=%s",
            model_id,
            row["beta_group_stage_g1"],
            row["SE"],
            row["p_wald"],
            row["optimizer"],
            row["converged"],
        )

    gate_rows = [
        {
            "check": "Historical code directly averages assigned ROI within Stage",
            "observed": True,
            "expected": True,
            "absolute_error": 0.0,
            "tolerance": 0.0,
            "status": "PASS",
        },
        {
            "check": "Rebuilt original stage FC matches saved stage table",
            "observed": rebuild_error,
            "expected": 0.0,
            "absolute_error": rebuild_error,
            "tolerance": 1e-12,
            "status": "PASS" if rebuild_error < 1e-12 else "FAIL",
        },
    ]
    gate_targets = [("Model A beta", model_rows[0]["beta_group_stage_g1"], TARGET_FULL),
                    ("Model B beta", model_rows[1]["beta_group_stage_g1"], TARGET_MINUS_INSULA)]
    for check, observed, expected in gate_targets:
        error = abs(float(observed) - float(expected))
        gate_rows.append(
            {
                "check": check,
                "observed": observed,
                "expected": expected,
                "absolute_error": error,
                "tolerance": GATE_TOLERANCE,
                "status": "PASS" if error <= GATE_TOLERANCE else "FAIL",
            }
        )
    gate = pd.DataFrame(gate_rows)
    write_tsv(gate, "reproduction_and_aggregation_audit.tsv")
    require((gate["status"] == "PASS").all(), "Model A/B reproduction gate failed")
    logging.info("Model A/B reproduction gates passed; proceeding to equal-subfamily models")

    # Only after the required gates pass are equal-subfamily models built and fit.
    stage_tables["C"] = build_stage_table(data["roi"], data["cov"], "Subfamily_equal", False)
    stage_tables["D"] = build_stage_table(data["roi"], data["cov"], "Subfamily_equal", True)
    for model_id in ["C", "D"]:
        result, row, attempt_rows = fit_model(model_id, stage_tables[model_id])
        results[model_id] = result
        model_rows.append(row)
        coefficient_table.extend(coefficient_rows(model_id, result))
        attempts.extend(attempt_rows)
        logging.info(
            "Sensitivity model %s: beta=%.17g SE=%.17g p=%.17g optimizer=%s converged=%s",
            model_id,
            row["beta_group_stage_g1"],
            row["SE"],
            row["p_wald"],
            row["optimizer"],
            row["converged"],
        )

    model_results = pd.DataFrame(model_rows)
    model_results["model"] = pd.Categorical(model_results["model"], categories=MODEL_ORDER, ordered=True)
    model_results = model_results.sort_values("model").reset_index(drop=True)
    model_results["model"] = model_results["model"].astype(str)
    effect_deltas = model_deltas(model_results)

    all_stage = []
    for model_id in MODEL_ORDER:
        table = stage_tables[model_id].copy()
        table.insert(0, "model", model_id)
        table.insert(1, "model_label", MODEL_META[model_id]["label"])
        table.insert(2, "weighting", MODEL_META[model_id]["weighting"])
        table.insert(3, "minus_insula", MODEL_META[model_id]["minus_insula"])
        all_stage.append(table)
    all_stage_table = pd.concat(all_stage, ignore_index=True)

    profiles, differences = build_profiles(stage_tables)
    stability = profile_stability(profiles, differences)

    require(len(model_results) == 4 and model_results["converged"].all(), "Not all four models converged")
    require(len(profiles) == 160, f"Expected 160 profile rows, got {len(profiles)}")
    require(len(differences) == 80, f"Expected 80 difference-profile rows, got {len(differences)}")
    require(len(stability) == 24, f"Expected 24 stability rows, got {len(stability)}")

    write_tsv(model_results, "four_model_mixedlm_results.tsv")
    write_tsv(pd.DataFrame(coefficient_table), "four_model_fixed_effect_coefficients.tsv")
    write_tsv(effect_deltas, "four_model_effect_deltas.tsv")
    write_tsv(profiles, "ten_bin_group_stage_fc_profiles.tsv")
    write_tsv(differences, "ten_bin_sz_minus_hc_difference_profiles.tsv")
    write_tsv(stability, "profile_stability_metrics.tsv")
    write_tsv(weights, "subfamily_aggregation_weights.tsv")
    write_tsv(all_stage_table, "subject_bin_stage_fc_four_models.tsv")
    write_tsv(pd.DataFrame(attempts), "optimizer_attempts.tsv")
    write_tsv(provenance(), "input_provenance_sha256.tsv")

    qc = {
        "status": "SUCCESS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_project_modified": False,
        "definition": "Definition A; 64 subjects; 62 assigned ROI; 10 bins",
        "sample": {"N": 64, "HC": 32, "SZ": 32},
        "roi_audit": {
            "assigned": 62,
            "Anticipation": {"roi": 43, "subfamilies": 6},
            "Outcome": {"roi": 19, "subfamilies": 7},
            "unassigned_excluded": 3,
        },
        "original_aggregation": "Arithmetic mean across assigned ROI in each Stage",
        "equal_weighting": "Mean within Subfamily, then equal mean across retained Subfamilies in each Stage",
        "model_formula": FORMULA,
        "random_effects": "Subject random intercept plus random G1_centered slope",
        "reml": False,
        "optimizer_order": ["lbfgs", "powell", "cg"],
        "g1_predictor": "G1_centered = Bin - 5.5; no 375-person G1 recomputation",
        "rebuild_saved_stage_max_abs_error": rebuild_error,
        "model_A_beta_abs_error": abs(model_results.set_index("model").loc["A", "beta_group_stage_g1"] - TARGET_FULL),
        "model_B_beta_abs_error": abs(model_results.set_index("model").loc["B", "beta_group_stage_g1"] - TARGET_MINUS_INSULA),
        "weight_sum_max_abs_error": float(
            np.max(
                np.abs(
                    weights.loc[weights["included"]]
                    .groupby(["model", "stage"])["subfamily_total_weight"]
                    .sum()
                    .to_numpy()
                    - 1.0
                )
            )
        ),
        "rows": {
            "stage": len(all_stage_table),
            "profiles": len(profiles),
            "difference_profiles": len(differences),
            "profile_metrics": len(stability),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "not_run": ["bootstrap", "new local significance tests", "375-person G1 recomputation"],
    }
    write_json(qc, "analysis_qc.json")
    (OUTPUT / "BLOCKING_REPORT.md").unlink(missing_ok=True)

    logging.info("All four models completed")
    logging.info("Model results:\n%s", model_results[["model", "beta_group_stage_g1", "SE", "CI95_low", "CI95_high", "p_wald", "optimizer", "converged"]].to_string(index=False))
    logging.info("Effect deltas:\n%s", effect_deltas.to_string(index=False))
    logging.info("STEP 3D completed successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_blocking_report(exc)
        logging.exception("STEP 3D failed")
        raise
