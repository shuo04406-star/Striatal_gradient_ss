"""STEP 3A leave-one-subfamily-out robustness analysis.

Run with the project-compatible CPython 3.12 runtime. The script reads the
frozen Definition A inputs without modifying them and writes only to the
independent teacher-report directory.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
import warnings

# The models are small and are parallelized across bootstrap replicates.
# Pinning each numerical-library process to one BLAS thread avoids nested
# oversubscription and makes Windows startup/convergence reproducible.
for _thread_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")


CANONICAL_ROOT = Path(
    os.path.join(os.environ["STRIATAL_PROJECT_ROOT"], 'CURRENT64_CANONICAL_RERUN')
)
VENDOR = CANONICAL_ROOT / ".python_packages"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
import patsy
import scipy
from scipy.stats import chi2, norm, spearmanr
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM


OUTPUT = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP3_ROI_robustness'))
PROJECT = Path(os.environ["STRIATAL_RELEASE_ROOT"])

ROI_LONG = CANONICAL_ROOT / "04_BINWISE_FC_PROFILES" / "subject_bin_roi_FC_long.tsv"
STAGE_SAVED = CANONICAL_ROOT / "06_REWARD_STAGE" / "subject_bin_stage_FC_long.tsv"
ROI_META = (
    CANONICAL_ROOT
    / "01_FROZEN_INPUTS"
    / "reward_stage_mapping"
    / "roi_metadata_stage_subfamily_v2.tsv"
)
ROI_ORDER = (
    CANONICAL_ROOT / "01_FROZEN_INPUTS" / "roi_order" / "canonical_roi_order_65.tsv"
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
ORIGINAL_SCRIPT = CANONICAL_ROOT / "scripts" / "python" / "03_run_stage_analyses.py"
MODEL_A_COEFF = (
    CANONICAL_ROOT / "07_STAGE_GRADIENT_MODELS" / "model_A_base_coefficients.tsv"
)
MODEL_A_FIT = CANONICAL_ROOT / "07_STAGE_GRADIENT_MODELS" / "model_A_fit.tsv"
MODEL_A_LRT = CANONICAL_ROOT / "07_STAGE_GRADIENT_MODELS" / "likelihood_ratio_tests.tsv"
MODEL_B_COEFF = (
    CANONICAL_ROOT / "07_STAGE_GRADIENT_MODELS" / "model_B_adjusted_coefficients.tsv"
)
MODEL_C_COEFF = (
    CANONICAL_ROOT
    / "07_STAGE_GRADIENT_MODELS"
    / "model_C_gradient_sensitive_coefficients.tsv"
)
CATEGORICAL_COMPARISON = (
    CANONICAL_ROOT
    / "08_CATEGORICAL_SHAPE"
    / "CATEGORICAL_SPECIFICATION_COMPARISON.tsv"
)
MIXED_GEE_AUDIT = (
    PROJECT / "validation" / "categorical_mixedlm_vs_gee_audit.py"
)

SEED = 20261011
DEFAULT_BOOTSTRAP_REPS = 1000
DEFAULT_WORKERS = 4
STAGES = ["Outcome", "Anticipation"]
GROUPS = ["HC", "SZ"]
FORMULA_A = "FC ~ group * stage * G1_centered"
REDUCED_A = "FC ~ group * stage + group * G1_centered + stage * G1_centered"
FORMULA_B = "FC ~ group * stage * G1_centered + age_c + sex + meanFD_c"
REDUCED_B = "FC ~ group * stage + group * G1_centered + stage * G1_centered + age_c + sex + meanFD_c"
FORMULA_C = (
    "FC ~ group * stage * G1_centered + age_c * stage * G1_centered + "
    "sex_c * stage * G1_centered + meanFD_c * stage * G1_centered"
)
REDUCED_C = (
    "FC ~ group * stage + group * G1_centered + stage * G1_centered + "
    "age_c * stage * G1_centered + sex_c * stage * G1_centered + "
    "meanFD_c * stage * G1_centered"
)
EFFECTS = {
    "Group_x_Stage_x_G1": np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=float),
    "Outcome_SZ_minus_HC_G1_slope": np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=float),
    "Anticipation_SZ_minus_HC_G1_slope": np.array([0, 0, 0, 0, 0, 1, 0, 1], dtype=float),
}


BOOT_Y: dict[str, np.ndarray] = {}
BOOT_STARTS: dict[str, np.ndarray] = {}
BOOT_X: np.ndarray | None = None
BOOT_Z: np.ndarray | None = None
BOOT_GROUPS: np.ndarray | None = None
BOOT_SEED = SEED


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.17g")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame.loc[:, columns].copy()

    def fmt(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    rows.extend(
        "| " + " | ".join(fmt(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def configure_logging() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "STEP3A_EXECUTION_LOG.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def load_inputs() -> dict[str, pd.DataFrame]:
    required = [
        ROI_LONG,
        STAGE_SAVED,
        ROI_META,
        ROI_ORDER,
        COVARIATES,
        SUBJECT_TABLE,
        DEFINITION_A,
        ORIGINAL_SCRIPT,
        MODEL_A_COEFF,
        MODEL_A_FIT,
        MODEL_A_LRT,
        MODEL_B_COEFF,
        MODEL_C_COEFF,
        CATEGORICAL_COMPARISON,
        MIXED_GEE_AUDIT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "Missing required frozen inputs:\n" + "\n".join(missing))
    return {
        "roi": pd.read_csv(ROI_LONG, sep="\t"),
        "stage_saved": pd.read_csv(STAGE_SAVED, sep="\t"),
        "meta": pd.read_csv(ROI_META, sep="\t"),
        "roi_order": pd.read_csv(ROI_ORDER, sep="\t"),
        "cov": pd.read_csv(COVARIATES, sep="\t"),
        "subjects": pd.read_csv(SUBJECT_TABLE, sep="\t"),
        "bins": pd.read_csv(DEFINITION_A, sep="\t"),
    }


def validate_frozen_inputs(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    roi = data["roi"]
    stage_saved = data["stage_saved"]
    meta = data["meta"]
    roi_order = data["roi_order"]
    cov = data["cov"]
    subjects = data["subjects"]
    bins = data["bins"]

    require(roi.shape == (41600, 7), f"Unexpected ROI table shape: {roi.shape}")
    require(stage_saved.shape == (1280, 9), f"Unexpected stage table shape: {stage_saved.shape}")
    require(meta.shape == (65, 6), f"Unexpected metadata shape: {meta.shape}")
    require(roi_order.shape == (65, 5), f"Unexpected canonical ROI order shape: {roi_order.shape}")
    require(cov.shape == (64, 5), f"Unexpected covariate shape: {cov.shape}")
    require(subjects.shape == (64, 2), f"Unexpected subject table shape: {subjects.shape}")
    require(bins.shape[0] == 2608, f"Definition A must contain 2,608 voxels: {bins.shape}")
    require(set(bins["bin"].unique()) == set(range(1, 11)), "Definition A bin labels are not 1-10")
    expected_bin_counts = [260, 260, 261, 261, 261, 261, 261, 261, 261, 261]
    actual_bin_counts = bins.groupby("bin").size().reindex(range(1, 11)).tolist()
    require(actual_bin_counts == expected_bin_counts, f"Wrong Definition A bin counts: {actual_bin_counts}")

    require(not roi.isna().any().any(), "Missing values in ROI table")
    require(not stage_saved.isna().any().any(), "Missing values in stage table")
    require(not cov.isna().any().any(), "Missing values in covariates")
    require(not roi.duplicated(["Subject", "Bin", "ROI"]).any(), "Duplicate Subject-Bin-ROI key")
    require(
        not stage_saved.duplicated(["Subject", "Bin", "Stage"]).any(),
        "Duplicate Subject-Bin-Stage key",
    )
    require(subjects["Subject"].nunique() == 64, "Subject table IDs are not unique")
    require(subjects["Group"].value_counts().to_dict() == {"HC": 32, "SZ": 32}, "Group count mismatch")
    require(set(roi["Subject"]) == set(subjects["Subject"]), "ROI/subject ID mismatch")
    require(set(cov["Subject"]) == set(subjects["Subject"]), "Covariate/subject ID mismatch")
    require(
        np.allclose(stage_saved["G1_bin_center"], stage_saved["Bin"] - 5.5, atol=0, rtol=0),
        "G1_bin_center is not Bin - 5.5",
    )

    roi_key = roi[["ROI", "Subfamily", "RewardStage"]].drop_duplicates()
    order_key = roi_order[["roi_index", "roi_name", "subfamily", "reward_stage", "atlas_source"]].rename(
        columns={
            "roi_name": "ROI",
            "subfamily": "Subfamily",
            "reward_stage": "RewardStage",
        }
    )
    merged = roi_key.merge(order_key, on=["ROI", "Subfamily", "RewardStage"], how="outer", indicator=True)
    require(len(roi_key) == 65 and (merged["_merge"] == "both").all(), "ROI table/canonical order mismatch")
    order_meta = order_key.merge(
        meta,
        left_on="roi_index",
        right_on="roi_index0",
        validate="one_to_one",
    )
    require(
        (order_meta["Subfamily"] == order_meta["subfamily"]).all()
        and (order_meta["RewardStage"] == order_meta["stage"]).all()
        and (order_meta["atlas_source"] == order_meta["brainnetome_label"]).all(),
        "Canonical ROI order/metadata index mapping mismatch",
    )

    assigned_meta = meta.loc[
        meta["include_two_family"].astype(int).eq(1)
        & meta["stage"].isin(["Anticipation", "Outcome"])
    ].copy()
    require(len(assigned_meta) == 62, "Assigned ROI count is not 62")
    stage_counts = assigned_meta.groupby("stage").size().to_dict()
    require(stage_counts == {"Anticipation": 43, "Outcome": 19}, f"Assigned stage counts mismatch: {stage_counts}")
    not_assigned = meta.loc[meta["include_two_family"].astype(int).eq(0)]
    require(len(not_assigned) == 3, "Not-assigned ROI count is not 3")

    return {
        "definition_a_voxels": 2608,
        "definition_a_bin_counts": actual_bin_counts,
        "subjects": 64,
        "HC": 32,
        "SZ": 32,
        "assigned_roi": 62,
        "anticipation_roi": 43,
        "outcome_roi": 19,
        "not_assigned_roi": 3,
        "subfamilies": int(assigned_meta["subfamily"].nunique()),
    }


def assigned_subfamilies(meta: pd.DataFrame) -> pd.DataFrame:
    assigned = meta.loc[
        meta["include_two_family"].astype(int).eq(1)
        & meta["stage"].isin(["Anticipation", "Outcome"])
    ].copy()
    assigned["roi_index0"] = pd.to_numeric(assigned["roi_index0"], errors="raise").astype(int)
    rows = []
    for (stage, subfamily), part in assigned.groupby(["stage", "subfamily"], sort=True):
        part = part.sort_values("roi_index0")
        rows.append(
            {
                "subfamily": subfamily,
                "stage": stage,
                "deleted_roi_count": len(part),
                "removed_roi_names": "; ".join(part["brainnetome_label"].astype(str)),
                "removed_roi_indices0": "; ".join(part["roi_index0"].astype(str)),
            }
        )
    result = pd.DataFrame(rows).sort_values(["stage", "subfamily"]).reset_index(drop=True)
    require(len(result) == 13, f"Expected 13 assigned subfamilies, found {len(result)}")
    return result


def build_stage_table(
    roi: pd.DataFrame,
    cov: pd.DataFrame,
    minus_subfamily: str | None,
) -> pd.DataFrame:
    selected = roi.loc[roi["RewardStage"].isin(["Outcome", "Anticipation"])].copy()
    if minus_subfamily is not None:
        selected = selected.loc[selected["Subfamily"].ne(minus_subfamily)].copy()
    stage = (
        selected.groupby(["Subject", "Group", "Bin", "RewardStage"], observed=True, as_index=False)["FC"]
        .mean()
        .rename(columns={"RewardStage": "Stage"})
    )
    stage = stage.merge(cov, on=["Subject", "Group"], how="left", validate="many_to_one")
    stage["G1_bin_center"] = stage["Bin"] - 5.5
    stage = stage[
        ["Subject", "Group", "Bin", "G1_bin_center", "Stage", "FC", "Age", "Sex", "MeanFD"]
    ].sort_values(["Subject", "Stage", "Bin"]).reset_index(drop=True)
    require(len(stage) == 1280, f"Stage table after deletion has {len(stage)} rows")
    require(not stage.isna().any().any(), "Stage table after deletion contains missing values")
    require(not stage.duplicated(["Subject", "Stage", "Bin"]).any(), "Duplicate deletion-stage keys")
    return stage


def compare_saved_stage(rebuilt: pd.DataFrame, saved: pd.DataFrame) -> float:
    joined = saved.merge(
        rebuilt,
        on=["Subject", "Group", "Bin", "G1_bin_center", "Stage", "Age", "Sex", "MeanFD"],
        suffixes=("_saved", "_rebuilt"),
        validate="one_to_one",
    )
    require(len(joined) == 1280, "Rebuilt/saved stage join did not retain 1,280 rows")
    return float(np.max(np.abs(joined["FC_saved"] - joined["FC_rebuilt"])))


def subject_order(subjects: pd.DataFrame) -> list[str]:
    ordered = subjects.copy()
    ordered["_group_order"] = ordered["Group"].map({"HC": 0, "SZ": 1})
    ordered = ordered.sort_values(["_group_order", "Subject"])
    return ordered["Subject"].astype(str).tolist()


def stage_to_y_matrix(stage: pd.DataFrame, ordered_subjects: list[str]) -> np.ndarray:
    stage_rank = {"Outcome": 0, "Anticipation": 1}
    matrix = []
    for subject in ordered_subjects:
        rows = stage.loc[stage["Subject"].eq(subject)].copy()
        rows["_stage_rank"] = rows["Stage"].map(stage_rank)
        rows = rows.sort_values(["_stage_rank", "Bin"])
        require(len(rows) == 20, f"{subject} does not have 20 stage-bin rows")
        expected_stage = ["Outcome"] * 10 + ["Anticipation"] * 10
        require(rows["Stage"].tolist() == expected_stage, f"Stage order failure for {subject}")
        require(rows["Bin"].tolist() == list(range(1, 11)) * 2, f"Bin order failure for {subject}")
        matrix.append(rows["FC"].to_numpy(float))
    return np.vstack(matrix)


def design_matrices(ordered_subjects: list[str], subjects: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_map = subjects.set_index("Subject")["Group"].to_dict()
    within_stage = np.repeat([0.0, 1.0], 10)
    within_g1 = np.tile(np.arange(1, 11, dtype=float) - 5.5, 2)
    rows = []
    zrows = []
    groups = []
    for cluster, subject in enumerate(ordered_subjects):
        group = 1.0 if group_map[subject] == "SZ" else 0.0
        for stage, g1 in zip(within_stage, within_g1):
            rows.append([1.0, group, stage, group * stage, g1, group * g1, stage * g1, group * stage * g1])
            zrows.append([1.0, g1])
            groups.append(cluster)
    return np.asarray(rows), np.asarray(zrows), np.asarray(groups)


def fit_matrix_model(
    y: np.ndarray,
    x: np.ndarray,
    groups: np.ndarray,
    z: np.ndarray,
    start_params: np.ndarray | None = None,
    maxiter: int = 5000,
) -> tuple[object, str, str]:
    model = MixedLM(y, x, groups, exog_re=z)
    messages = []
    first_start = start_params
    for method in ["lbfgs", "powell", "cg"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = model.fit(
                    reml=False,
                    method=method,
                    start_params=first_start,
                    maxiter=maxiter,
                    disp=False,
                )
                warning_text = " | ".join(str(item.message) for item in caught)
                messages.append(f"{method}: converged={result.converged}; {warning_text}")
                if bool(result.converged):
                    return result, method, " || ".join(messages)
                first_start = np.asarray(result.params)
            except Exception as exc:
                messages.append(f"{method}: ERROR {exc!r}")
                first_start = start_params
    raise RuntimeError("MixedLM failed: " + " || ".join(messages))


def effect_rows(result: object) -> list[dict[str, float | str]]:
    beta = np.asarray(result.fe_params, dtype=float)
    cov = np.asarray(result.cov_params(), dtype=float)[:8, :8]
    rows = []
    for name, weights in EFFECTS.items():
        estimate = float(weights @ beta)
        variance = float(weights @ cov @ weights)
        se = math.sqrt(max(variance, 0.0))
        statistic = estimate / se
        rows.append(
            {
                "effect": name,
                "estimate": estimate,
                "SE": se,
                "CI95_low": estimate - 1.96 * se,
                "CI95_high": estimate + 1.96 * se,
                "z": statistic,
                "p_wald": float(2 * norm.sf(abs(statistic))),
            }
        )
    return rows


def normalize_formula_input(stage: pd.DataFrame) -> pd.DataFrame:
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
    male = frame["sex"].map({"Female": 0.0, "Male": 1.0}).astype(float)
    male_prop = float(people["sex"].map({"Female": 0.0, "Male": 1.0}).astype(float).mean())
    frame["sex_c"] = male - male_prop
    return frame


def fit_formula_model(formula: str, frame: pd.DataFrame) -> tuple[object, str, str]:
    model = smf.mixedlm(
        formula,
        data=frame,
        groups=frame["subject"],
        re_formula="1 + G1_centered",
    )
    attempts = []
    for method in ["lbfgs", "powell", "cg"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = model.fit(reml=False, method=method, maxiter=5000, disp=False)
                warning_text = " | ".join(str(item.message) for item in caught)
                attempts.append(f"{method}: converged={result.converged}; {warning_text}")
                if bool(result.converged):
                    return result, method, " || ".join(attempts)
            except Exception as exc:
                attempts.append(f"{method}: ERROR {exc!r}")
    raise RuntimeError("Formula MixedLM failed: " + " || ".join(attempts))


def validate_original_models(
    full_stage: pd.DataFrame,
    y_full: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    groups: np.ndarray,
    rebuild_max_abs_diff: float,
) -> tuple[pd.DataFrame, dict[str, object], object]:
    checks: list[dict[str, object]] = []
    model_a, optimizer_a, warnings_a = fit_matrix_model(y_full.reshape(-1), x, groups, z)
    reduced_a, optimizer_red, warnings_red = fit_matrix_model(y_full.reshape(-1), x[:, :7], groups, z)
    lr_a = float(2 * (model_a.llf - reduced_a.llf))
    p_lrt_a = float(chi2.sf(lr_a, 1))
    observed_a = {row["effect"]: row for row in effect_rows(model_a)}

    saved_coeff = pd.read_csv(MODEL_A_COEFF, sep="\t")
    saved_fit = pd.read_csv(MODEL_A_FIT, sep="\t").iloc[0]
    saved_lrt = pd.read_csv(MODEL_A_LRT, sep="\t")
    saved_lrt = saved_lrt.loc[saved_lrt["comparison"].eq("Model A: Group x Stage x G1")].iloc[0]
    saved_three = saved_coeff.loc[
        saved_coeff["term"].eq("group[T.SZ]:stage[T.Anticipation]:G1_centered")
    ].iloc[0]
    checks.extend(
        [
            {
                "check": "ROI-level unweighted aggregation reproduces saved 1,280-row stage table",
                "status": "PASS" if rebuild_max_abs_diff < 1e-12 else "FAIL",
                "observed": rebuild_max_abs_diff,
                "expected": "<1e-12",
            },
            {
                "check": "Model A log-likelihood",
                "status": "PASS" if abs(model_a.llf - float(saved_fit["logLik"])) < 1e-10 else "FAIL",
                "observed": float(model_a.llf),
                "expected": float(saved_fit["logLik"]),
            },
            {
                "check": "Model A Group x Stage x G1 estimate",
                "status": "PASS"
                if abs(observed_a["Group_x_Stage_x_G1"]["estimate"] - float(saved_three["estimate"])) < 1e-12
                else "FAIL",
                "observed": observed_a["Group_x_Stage_x_G1"]["estimate"],
                "expected": float(saved_three["estimate"]),
            },
            {
                "check": "Model A Group x Stage x G1 SE",
                "status": "PASS"
                if abs(observed_a["Group_x_Stage_x_G1"]["SE"] - float(saved_three["SE"])) < 1e-12
                else "FAIL",
                "observed": observed_a["Group_x_Stage_x_G1"]["SE"],
                "expected": float(saved_three["SE"]),
            },
            {
                "check": "Model A 1-df nested likelihood-ratio statistic",
                "status": "PASS" if abs(lr_a - float(saved_lrt["LR_chi2"])) < 1e-10 else "FAIL",
                "observed": lr_a,
                "expected": float(saved_lrt["LR_chi2"]),
            },
        ]
    )

    formula_frame = normalize_formula_input(full_stage)
    secondary = {}
    for label, full_formula, reduced_formula, saved_path in [
        ("B", FORMULA_B, REDUCED_B, MODEL_B_COEFF),
        ("C", FORMULA_C, REDUCED_C, MODEL_C_COEFF),
    ]:
        full, optimizer, warning_text = fit_formula_model(full_formula, formula_frame)
        reduced, reduced_optimizer, reduced_warning = fit_formula_model(reduced_formula, formula_frame)
        term = "group[T.SZ]:stage[T.Anticipation]:G1_centered"
        estimate = float(full.fe_params[term])
        saved = pd.read_csv(saved_path, sep="\t")
        saved_estimate = float(saved.loc[saved["term"].eq(term), "estimate"].iloc[0])
        lr = float(2 * (full.llf - reduced.llf))
        checks.append(
            {
                "check": f"Model {label} Group x Stage x G1 estimate",
                "status": "PASS" if abs(estimate - saved_estimate) < 1e-10 else "FAIL",
                "observed": estimate,
                "expected": saved_estimate,
            }
        )
        secondary[label] = {
            "estimate": estimate,
            "lrt": lr,
            "lrt_p": float(chi2.sf(lr, 1)),
            "optimizer": optimizer,
            "warnings": warning_text,
            "reduced_optimizer": reduced_optimizer,
            "reduced_warnings": reduced_warning,
        }

    check_frame = pd.DataFrame(checks)
    require((check_frame["status"] == "PASS").all(), "Original model reproduction gate failed")
    details = {
        "model_A": {
            "formula": FORMULA_A,
            "reduced_formula": REDUCED_A,
            "optimizer": optimizer_a,
            "reduced_optimizer": optimizer_red,
            "warnings": warnings_a,
            "reduced_warnings": warnings_red,
            "logLik": float(model_a.llf),
            "lrt": lr_a,
            "lrt_p": p_lrt_a,
            "effects": observed_a,
        },
        "secondary_full_validation": secondary,
    }
    return check_frame, details, model_a


def profile_vectors(stage_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    vectors: dict[tuple[str, str], np.ndarray] = {}
    for variant, table in stage_tables.items():
        means = table.groupby(["Group", "Stage", "Bin"], observed=True)["FC"].mean()
        for stage in ["Anticipation", "Outcome"]:
            hc = means.loc[("HC", stage)].reindex(range(1, 11)).to_numpy(float)
            sz = means.loc[("SZ", stage)].reindex(range(1, 11)).to_numpy(float)
            vectors[(variant, stage)] = sz - hc

    full = vectors[("Full", "Anticipation")]
    rows = []
    for variant in stage_tables:
        if variant == "Full":
            continue
        subfamily = variant.removeprefix("Minus ")
        for stage in ["Anticipation", "Outcome"]:
            full_vec = vectors[("Full", stage)]
            deleted = vectors[(variant, stage)]
            delta = deleted - full_vec
            rmse = float(np.sqrt(np.mean(delta**2)))
            max_abs = float(np.max(np.abs(delta)))
            correlation = float(np.corrcoef(full_vec, deleted)[0, 1])
            row: dict[str, object] = {
                "subfamily": subfamily,
                "stage": stage,
                "profile_RMSE": rmse,
                "maximum_absolute_bin_difference": max_abs,
                "profile_correlation": correlation,
            }
            for index in range(10):
                bin_id = index + 1
                row[f"full_D_bin{bin_id}"] = full_vec[index]
                row[f"deleted_D_bin{bin_id}"] = deleted[index]
                row[f"delta_from_full_bin{bin_id}"] = delta[index]
            rows.append(row)
    return pd.DataFrame(rows), vectors


def _bootstrap_initializer(
    y_data: dict[str, np.ndarray],
    starts: dict[str, np.ndarray],
    x: np.ndarray,
    z: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> None:
    global BOOT_Y, BOOT_STARTS, BOOT_X, BOOT_Z, BOOT_GROUPS, BOOT_SEED
    BOOT_Y = y_data
    BOOT_STARTS = starts
    BOOT_X = x
    BOOT_Z = z
    BOOT_GROUPS = groups
    BOOT_SEED = seed


def _bootstrap_worker(replicate: int) -> dict[str, object]:
    assert BOOT_X is not None and BOOT_Z is not None and BOOT_GROUPS is not None
    rng = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, replicate]))
    hc = rng.integers(0, 32, size=32)
    sz = rng.integers(32, 64, size=32)
    draw = np.concatenate([hc, sz])
    estimates: dict[str, dict[str, float]] = {}
    optimizers: dict[str, str] = {}
    errors: dict[str, str] = {}
    for variant, y_matrix in BOOT_Y.items():
        y = y_matrix[draw, :].reshape(-1)
        try:
            result, optimizer, warning_text = fit_matrix_model(
                y,
                BOOT_X,
                BOOT_GROUPS,
                BOOT_Z,
                start_params=BOOT_STARTS[variant],
                maxiter=5000,
            )
            estimates[variant] = {row["effect"]: float(row["estimate"]) for row in effect_rows(result)}
            optimizers[variant] = optimizer
        except Exception as exc:
            errors[variant] = repr(exc)
    return {
        "replicate": replicate,
        "estimates": estimates,
        "optimizers": optimizers,
        "errors": errors,
    }


def paired_bootstrap(
    y_matrices: dict[str, np.ndarray],
    starts: dict[str, np.ndarray],
    x: np.ndarray,
    z: np.ndarray,
    groups: np.ndarray,
    reps: int,
    workers: int,
) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame]:
    deltas: dict[tuple[str, str], list[float]] = {
        (variant, effect): []
        for variant in y_matrices
        if variant != "Full"
        for effect in EFFECTS
    }
    optimizer_counts: dict[tuple[str, str], int] = {}
    error_counts: dict[str, int] = {variant: 0 for variant in y_matrices}
    checkpoint = OUTPUT / f".step3a_bootstrap_seed{SEED}_reps{reps}.jsonl"
    completed: dict[int, dict[str, object]] = {}
    if checkpoint.exists():
        for line_number, line in enumerate(checkpoint.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                replicate = int(item["replicate"])
            except Exception as exc:
                raise RuntimeError(f"Invalid bootstrap checkpoint line {line_number}: {exc}") from exc
            if 0 <= replicate < reps:
                completed[replicate] = item
        logging.info("Resuming bootstrap checkpoint: %d/%d replicates recovered", len(completed), reps)

    def consume(result: dict[str, object]) -> None:
        estimates = result["estimates"]
        for variant, optimizer in result["optimizers"].items():
            optimizer_counts[(variant, optimizer)] = optimizer_counts.get((variant, optimizer), 0) + 1
        for variant in result["errors"]:
            error_counts[variant] += 1
        if "Full" in estimates:
            for variant in y_matrices:
                if variant == "Full" or variant not in estimates:
                    continue
                for effect in EFFECTS:
                    deltas[(variant, effect)].append(
                        estimates[variant][effect] - estimates["Full"][effect]
                    )

    for replicate in sorted(completed):
        consume(completed[replicate])

    finished = len(completed)
    missing = [replicate for replicate in range(reps) if replicate not in completed]
    context = mp.get_context("spawn")
    logging.info(
        "Starting paired stratified subject bootstrap: reps=%d, seed=%d, workers=%d",
        reps,
        SEED,
        workers,
    )
    if missing:
        with checkpoint.open("a", encoding="utf-8", buffering=1) as checkpoint_handle:
            with context.Pool(
                processes=workers,
                initializer=_bootstrap_initializer,
                initargs=(y_matrices, starts, x, z, groups, SEED),
            ) as pool:
                for result in pool.imap_unordered(_bootstrap_worker, missing, chunksize=1):
                    checkpoint_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    checkpoint_handle.flush()
                    consume(result)
                    finished += 1
                    if finished % 25 == 0 or finished == reps:
                        logging.info("Bootstrap progress: %d/%d replicates completed", finished, reps)

    rows = []
    array_result: dict[tuple[str, str], np.ndarray] = {}
    for key, values in deltas.items():
        array = np.asarray(values, dtype=float)
        array_result[key] = array
        rows.append(
            {
                "variant": key[0],
                "effect": key[1],
                "valid_paired_replicates": len(array),
                "requested_replicates": reps,
                "delta_bootstrap_mean": float(np.mean(array)) if len(array) else np.nan,
                "delta_bootstrap_ci_low": float(np.quantile(array, 0.025)) if len(array) else np.nan,
                "delta_bootstrap_ci_high": float(np.quantile(array, 0.975)) if len(array) else np.nan,
            }
        )
    for variant, errors in error_counts.items():
        logging.info("Bootstrap failures for %s: %d/%d", variant, errors, reps)
    for (variant, optimizer), count in sorted(optimizer_counts.items()):
        logging.info("Bootstrap optimizer count | %s | %s = %d", variant, optimizer, count)
    audit = pd.DataFrame(rows)
    minimum_valid = max(min(100, reps), int(math.ceil(0.95 * reps)))
    require(
        (audit["valid_paired_replicates"] >= minimum_valid).all(),
        "Too few valid paired bootstrap replicates for at least one deletion/effect",
    )
    if checkpoint.exists():
        checkpoint.unlink()
        logging.info("Completed bootstrap checkpoint removed after validation")
    return array_result, audit


def input_hashes() -> pd.DataFrame:
    paths = [
        ROI_LONG,
        STAGE_SAVED,
        ROI_META,
        ROI_ORDER,
        COVARIATES,
        SUBJECT_TABLE,
        DEFINITION_A,
        ORIGINAL_SCRIPT,
        MODEL_A_COEFF,
        MODEL_A_FIT,
        MODEL_A_LRT,
        MODEL_B_COEFF,
        MODEL_C_COEFF,
        CATEGORICAL_COMPARISON,
        MIXED_GEE_AUDIT,
    ]
    return pd.DataFrame(
        [{"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths]
    )


def write_validation_report(
    checks: pd.DataFrame,
    validation: dict[str, object],
    qc: dict[str, object],
    hashes: pd.DataFrame,
) -> None:
    model_a = validation["model_A"]
    model_b = validation["secondary_full_validation"]["B"]
    model_c = validation["secondary_full_validation"]["C"]
    report = f"""# STEP 3A Model Validation / 模型复现与版本锁定

## Gate result / 门槛结论

**PASS.** The full assigned-ROI model was reproduced before any subfamily deletion was analyzed.
**通过。** 在运行任何 subfamily 删除模型前，已从 41,600 行 ROI-level 表重建 Full stage 数据并复现原模型。

{markdown_table(checks, ['check', 'status', 'observed', 'expected'])}

## Frozen inputs / 冻结输入

- Definition A contains **{qc['definition_a_voxels']}** clinical voxels; bin counts are `{qc['definition_a_bin_counts']}`. No 2,847-voxel historical bin definition was used.
- ROI-level table: 41,600 rows = 64 subjects × 10 bins × 65 ROIs; assigned analysis uses 62 ROIs (Anticipation 43, Outcome 19), while 3 `Not_assigned` ROIs remain excluded.
- FC is resting-state Fisher-z transformed Pearson correlation.
- `Stage` is the ROI reward-stage classification, not a task epoch.
- Each bin averages voxels equally; each stage averages included ROIs equally. Leave-one-subfamily-out recalculates the affected stage mean with equal weight per retained ROI.
- Continuous model predictor is `G1_centered = Bin - 5.5`, giving `-4.5, ..., 4.5`. It is the centered 10-bin coordinate used by the original model, not a recomputed 375-person gradient.

## Original primary model / 原主要模型

- Formula: `{FORMULA_A}`.
- References: Group=HC, Stage=Outcome.
- Estimator: statsmodels MixedLM, maximum likelihood (`reml=False`).
- Random effects: subject random intercept + random `G1_centered` slope.
- Optimizer order: L-BFGS, Powell, CG; Full reproduced with `{model_a['optimizer']}`.
- Full observations: 1,280; subjects: 64 (HC=32, SZ=32).
- Full Group × Stage × G1 estimate: `{model_a['effects']['Group_x_Stage_x_G1']['estimate']:.12g}`; Wald 95% CI `[{model_a['effects']['Group_x_Stage_x_G1']['CI95_low']:.12g}, {model_a['effects']['Group_x_Stage_x_G1']['CI95_high']:.12g}]`.
- Nested ML LRT: χ²(1)=`{model_a['lrt']:.12g}`, p=`{model_a['lrt_p']:.12g}`.
- Boundary warning retained: `{model_a['warnings']}`.

## Covariate models / 协变量模型

The full-data covariate models were reproduced as validation, but the deletion series uses original primary Model A rather than selecting a model by significance.

- Model B: additive Age, Sex and MeanFD adjustment; three-way estimate `{model_b['estimate']:.12g}`, LRT p=`{model_b['lrt_p']:.12g}`.
- Model C: gradient-sensitive Age/Sex/MeanFD terms; three-way estimate `{model_c['estimate']:.12g}`, LRT p=`{model_c['lrt_p']:.12g}`.

## MixedLM and GEE provenance / MixedLM 与 GEE 冲突记录

The known MixedLM/GEE conflict concerns the **categorical** `Group × Stage × C(Bin)` omnibus, not this continuous-G1 Model A. The canonical categorical audit used a nine-degree-of-freedom robust GEE; a historical checkpoint reported a different result and its exact generating script was not recovered. STEP 3A does not choose between those categorical methods and does not refit the categorical model. It applies the recovered continuous-G1 MixedLM specification consistently to Full and every deletion.

## Input SHA-256

{markdown_table(hashes, ['path', 'sha256', 'bytes'])}
"""
    (OUTPUT / "STEP3A_MODEL_VALIDATION.md").write_text(report, encoding="utf-8")


def build_reports(
    subfamilies: pd.DataFrame,
    effect_table: pd.DataFrame,
    profile_table: pd.DataFrame,
    deletion_table: pd.DataFrame,
    bootstrap_reps: int,
) -> None:
    three = deletion_table.copy().sort_values("absolute_threeway_change", ascending=False)
    rmse_rank = deletion_table.copy().sort_values("affected_stage_profile_RMSE", ascending=False)
    top_effect = three.iloc[0]
    top_rmse = rmse_rank.iloc[0]
    insula = deletion_table.loc[deletion_table["subfamily"].eq("Insula")].iloc[0]
    key = deletion_table.loc[deletion_table["subfamily"].isin(["Insula", "SMA", "IFG"])].copy()
    count_effect_rho = float(spearmanr(deletion_table["deleted_roi_count"], deletion_table["absolute_threeway_change"]).statistic)
    count_rmse_rho = float(spearmanr(deletion_table["deleted_roi_count"], deletion_table["affected_stage_profile_RMSE"]).statistic)

    influence_display = three[
        [
            "subfamily",
            "stage",
            "deleted_roi_count",
            "threeway_effect",
            "threeway_change_from_full",
            "threeway_delta_boot_ci_low",
            "threeway_delta_boot_ci_high",
            "affected_stage_profile_RMSE",
            "affected_stage_profile_max_abs_difference",
        ]
    ].copy()
    key_display = key[
        [
            "subfamily",
            "stage",
            "deleted_roi_count",
            "threeway_change_from_full",
            "threeway_delta_boot_ci_low",
            "threeway_delta_boot_ci_high",
            "affected_stage_profile_RMSE",
        ]
    ]
    stable_delta = bool(
        (deletion_table["threeway_delta_boot_ci_low"] <= 0).all()
        and (deletion_table["threeway_delta_boot_ci_high"] >= 0).all()
    )
    delta_signals = deletion_table.loc[
        (deletion_table["threeway_delta_boot_ci_low"] > 0)
        | (deletion_table["threeway_delta_boot_ci_high"] < 0)
    ].sort_values("absolute_threeway_change", ascending=False)
    delta_signal_names = ", ".join(delta_signals["subfamily"].astype(str))
    report = f"""# STEP 3A Leave-One-Subfamily-Out Analysis Report

## Core conclusion / 核心结论

**中文：** 对 13 个 assigned subfamilies 逐一从 Full 数据重新删除后，Group × Stage × G1 点估计变化最大的 subfamily 是 **{top_effect['subfamily']}**（删除 {int(top_effect['deleted_roi_count'])} 个 ROI；Δ=`{top_effect['threeway_change_from_full']:.6g}`）。Group-difference profile RMSE 最大的是 **{top_rmse['subfamily']}**（affected stage={top_rmse['stage']}；RMSE=`{top_rmse['affected_stage_profile_RMSE']:.6g}`）。Insula 删除后的三阶效应变化为 `{insula['threeway_change_from_full']:.6g}`。{'所有 paired-bootstrap 效应差值区间均包含 0；这支持相对稳定的描述，但不证明等效。' if stable_delta else f'paired-bootstrap 效应差值区间不含 0 的删除为：{delta_signal_names}；这些是未作多重校正的敏感性信号，不能按主模型 p 值或“显著/不显著”二分。'}
**English:** Across 13 independent leave-one-subfamily-out models, the largest absolute change in the Group × Stage × G1 estimate followed removal of **{top_effect['subfamily']}** ({int(top_effect['deleted_roi_count'])} ROIs; Δ=`{top_effect['threeway_change_from_full']:.6g}`). The largest group-difference profile RMSE followed removal of **{top_rmse['subfamily']}** in {top_rmse['stage']} (RMSE=`{top_rmse['affected_stage_profile_RMSE']:.6g}`). The three-way change after removing Insula was `{insula['threeway_change_from_full']:.6g}`. {'All paired-bootstrap effect-difference intervals included zero; this supports a relative-stability description but does not establish equivalence.' if stable_delta else f'Effect-difference intervals excluded zero after removing {delta_signal_names}. These are unadjusted sensitivity signals, not a significance-based model-selection rule.'}

## Methods / 方法

- Each deletion started from Full; deletions were never cumulative.
- Primary model: `{FORMULA_A}`, ML MixedLM, subject random intercept and random G1 slope, HC and Outcome references.
- Stage-specific group effects are defined as the **SZ-minus-HC difference in the continuous-G1 slope** within Anticipation or Outcome.
- Model CIs are Wald 95% CIs from the unchanged original MixedLM specification.
- Effect-change CIs use **{bootstrap_reps}** stratified subject bootstraps (32 HC and 32 SZ draws with replacement), seed **{SEED}**. Each replicate uses the same subject draw for Full and all deletion models; duplicate draws receive unique bootstrap cluster IDs.
- Profile metric: `D(bin)=mean_SZ(bin)-mean_HC(bin)`. RMSE and maximum absolute bin difference compare deleted and Full D profiles. Correlation is supplementary only.
- No categorical model, random ROI-count-matched deletion, single-ROI deletion, or G1 recomputation was run.

## Influence ranking / 影响排序

{markdown_table(influence_display, list(influence_display.columns))}

## Insula, SMA and IFG / 关键 subfamilies

{markdown_table(key_display, list(key_display.columns))}

## ROI count relationship / ROI 数量关系

Across 13 subfamilies, the descriptive Spearman correlation between deleted ROI count and absolute three-way effect change was `ρ={count_effect_rho:.4f}`; the correlation with affected-stage profile RMSE was `ρ={count_rmse_rho:.4f}`. These are descriptive quantity relationships only. They do not separate ROI count from anatomical specificity. STEP 3B count-matched random deletion was intentionally not run.

## Interpretation limits / 解释边界

- Stability is assessed by effect-change magnitude, paired-bootstrap difference intervals, and D-profile RMSE/max difference, not by whether a p value crosses 0.05.
- A difference interval containing zero is not an equivalence test.
- Larger effects after deleting a larger subfamily cannot be assigned to regional specificity without ROI-count-matched deletion.
- This analysis does not recompute the 375-person G1 reference and therefore cannot assess normative-gradient stability.
"""
    (OUTPUT / "STEP3A_ANALYSIS_REPORT.md").write_text(report, encoding="utf-8")

    teacher = f"""# STEP 3A Teacher Summary / 导师汇报摘要

## Why this analysis? / 为什么做这个分析？

**中文：** 该分析检验原 Group × Stage × G1 结果是否由 Insula 或其他单一 ROI subfamily 主导。每个模型只删除一个 subfamily，并始终从相同 Full 数据重新开始。
**English:** This analysis tests whether the original Group × Stage × G1 result is driven by Insula or any other single ROI subfamily. Each model removes one subfamily from the same Full dataset.

## Main result / 主要结果

**中文：** 三阶效应变化最大的删除是 **{top_effect['subfamily']}**（{int(top_effect['deleted_roi_count'])} ROI，Δ={top_effect['threeway_change_from_full']:.6g}）；profile RMSE 最大的是 **{top_rmse['subfamily']}**（{top_rmse['stage']}，RMSE={top_rmse['affected_stage_profile_RMSE']:.6g}）。
**English:** The largest three-way effect change followed removal of **{top_effect['subfamily']}** ({int(top_effect['deleted_roi_count'])} ROIs; Δ={top_effect['threeway_change_from_full']:.6g}); the largest profile RMSE followed removal of **{top_rmse['subfamily']}** in {top_rmse['stage']} (RMSE={top_rmse['affected_stage_profile_RMSE']:.6g}).

**中文：** {'所有效应差值区间均跨 0。' if stable_delta else f'效应差值区间不跨 0 的删除为 {delta_signal_names}；这些区间未作多重比较校正，只作为敏感性信号。'}
**English:** {'Every effect-difference interval crossed zero.' if stable_delta else f'Effect-difference intervals excluded zero for {delta_signal_names}; these intervals are unadjusted and are treated as sensitivity signals.'}

## Insula, SMA and IFG / Insula、SMA、IFG

{markdown_table(key_display, list(key_display.columns))}

## ROI count / ROI 数量

**中文：** 删除 ROI 数量与绝对效应变化、profile RMSE 的描述性 Spearman 相关分别为 {count_effect_rho:.4f} 和 {count_rmse_rho:.4f}。该结果不能区分数量效应与脑区特异性；需要 STEP 3B 的 count-matched 随机删除才能进一步回答。
**English:** Deleted ROI count had descriptive Spearman correlations of {count_effect_rho:.4f} with absolute effect change and {count_rmse_rho:.4f} with profile RMSE. This does not separate count effects from anatomical specificity; that requires the prespecified STEP 3B count-matched analysis.

## What can be concluded? / 能得出什么结论？

**中文：** {'现有结果支持“对单个 subfamily 删除相对稳定”的描述，但不能证明效应等效或不存在 subfamily 贡献。' if stable_delta else '现有结果显示至少一个 subfamily 删除会产生值得关注的效应变化，因此不能将 Full 结果描述为对所有 subfamily 删除均稳定。'} 不能用显著性是否改变作为唯一判断。
**English:** {'The results support relative robustness to individual subfamily deletion, but do not prove equivalence or absence of subfamily contribution.' if stable_delta else 'At least one deletion produced a notable effect change, so the Full result should not be described as uniformly robust to every subfamily removal.'} Changes in statistical significance are not the stability criterion.

## What remains unanswered? / 仍不能回答什么？

**中文：** 本分析不能证明某一解剖 subfamily 具有特异性驱动作用，不能评估 375 人参考 G1 本身的稳定性，也没有执行 STEP 3B 或 STEP 3C。
**English:** This analysis cannot establish anatomically specific causation, cannot assess stability of the 375-person reference G1, and does not include STEP 3B or STEP 3C.
"""
    (OUTPUT / "STEP3A_TEACHER_SUMMARY.md").write_text(teacher, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    configure_logging()
    started = time.perf_counter()
    logging.info("STEP 3A started")
    logging.info("Python: %s", sys.version.replace("\n", " "))
    logging.info(
        "Versions: numpy=%s pandas=%s scipy=%s statsmodels=%s patsy=%s",
        np.__version__,
        pd.__version__,
        scipy.__version__,
        statsmodels.__version__,
        patsy.__version__,
    )
    logging.info("Platform: %s", platform.platform())
    logging.info("Output directory: %s", OUTPUT)

    try:
        data = load_inputs()
        qc = validate_frozen_inputs(data)
        subfamilies = assigned_subfamilies(data["meta"])
        logging.info("Frozen input validation passed: %s", json.dumps(qc, ensure_ascii=False))

        full_stage = build_stage_table(data["roi"], data["cov"], None)
        rebuild_diff = compare_saved_stage(full_stage, data["stage_saved"])
        logging.info("Full stage reconstruction max absolute difference: %.17g", rebuild_diff)

        order = subject_order(data["subjects"])
        stage_tables = {"Full": full_stage}
        for subfamily in subfamilies["subfamily"]:
            stage_tables[f"Minus {subfamily}"] = build_stage_table(data["roi"], data["cov"], subfamily)

        x, z, group_ids = design_matrices(order, data["subjects"])
        y_matrices = {variant: stage_to_y_matrix(table, order) for variant, table in stage_tables.items()}
        checks, validation, full_result = validate_original_models(
            full_stage,
            y_matrices["Full"],
            x,
            z,
            group_ids,
            rebuild_diff,
        )
        hashes = input_hashes()
        write_validation_report(checks, validation, qc, hashes)
        logging.info("Original model reproduction gate passed")
        if args.validate_only:
            logging.info("Validation-only mode completed")
            return

        observed_results: dict[str, object] = {"Full": full_result}
        observed_reduced: dict[str, object] = {}
        starts: dict[str, np.ndarray] = {"Full": np.asarray(full_result.params)}
        effect_records = []
        model_records = []

        full_reduced, full_red_optimizer, full_red_warn = fit_matrix_model(
            y_matrices["Full"].reshape(-1), x[:, :7], group_ids, z
        )
        observed_reduced["Full"] = full_reduced

        for variant, y_matrix in y_matrices.items():
            if variant == "Full":
                result = full_result
                optimizer = validation["model_A"]["optimizer"]
                warning_text = validation["model_A"]["warnings"]
                reduced = full_reduced
                reduced_optimizer = full_red_optimizer
                reduced_warning = full_red_warn
            else:
                result, optimizer, warning_text = fit_matrix_model(y_matrix.reshape(-1), x, group_ids, z)
                reduced, reduced_optimizer, reduced_warning = fit_matrix_model(
                    y_matrix.reshape(-1), x[:, :7], group_ids, z
                )
                observed_results[variant] = result
                observed_reduced[variant] = reduced
                starts[variant] = np.asarray(result.params)
            lr = float(2 * (result.llf - reduced.llf))
            model_records.append(
                {
                    "model": variant,
                    "converged": bool(result.converged),
                    "optimizer": optimizer,
                    "warnings": warning_text,
                    "N_subjects": 64,
                    "N_observations": 1280,
                    "logLik": float(result.llf),
                    "AIC": float(result.aic),
                    "BIC": float(result.bic),
                    "random_intercept_variance": float(np.asarray(result.cov_re)[0, 0]),
                    "random_G1_slope_variance": float(np.asarray(result.cov_re)[1, 1]),
                    "random_intercept_slope_covariance": float(np.asarray(result.cov_re)[0, 1]),
                    "reduced_optimizer": reduced_optimizer,
                    "reduced_warnings": reduced_warning,
                    "LRT_chi2_df1": lr,
                    "LRT_p": float(chi2.sf(lr, 1)),
                }
            )
            for row in effect_rows(result):
                effect_records.append({"model": variant, **row})

        model_table = pd.DataFrame(model_records)
        require(model_table["converged"].all(), "At least one observed deletion model did not converge")
        effect_table = pd.DataFrame(effect_records)
        full_effects = effect_table.loc[effect_table["model"].eq("Full")].set_index("effect")["estimate"]
        effect_table["change_from_full"] = effect_table.apply(
            lambda row: float(row["estimate"] - full_effects.loc[row["effect"]]), axis=1
        )
        effect_table["absolute_change_from_full"] = effect_table["change_from_full"].abs()
        effect_table["direction"] = np.where(
            effect_table["estimate"] > 0,
            "positive",
            np.where(effect_table["estimate"] < 0, "negative", "zero"),
        )

        profile_table, profile_vector_map = profile_vectors(stage_tables)
        profile_table = profile_table.merge(
            subfamilies[["subfamily", "stage", "deleted_roi_count"]].rename(columns={"stage": "deleted_subfamily_stage"}),
            on="subfamily",
            how="left",
            validate="many_to_one",
        )
        profile_table["stage_is_affected"] = profile_table["stage"].eq(profile_table["deleted_subfamily_stage"])
        unaffected = profile_table.loc[~profile_table["stage_is_affected"]]
        require(
            float(unaffected["maximum_absolute_bin_difference"].max()) < 1e-15,
            "Deletion unexpectedly altered the other Stage profile",
        )

        workers = max(1, min(int(args.workers), os.cpu_count() or 1, 4))
        bootstrap_arrays, bootstrap_audit = paired_bootstrap(
            y_matrices,
            starts,
            x,
            z,
            group_ids,
            int(args.bootstrap_reps),
            workers,
        )
        effect_table["paired_bootstrap_replicates"] = np.nan
        effect_table["change_bootstrap_ci_low"] = np.nan
        effect_table["change_bootstrap_ci_high"] = np.nan
        for variant in y_matrices:
            if variant == "Full":
                continue
            for effect in EFFECTS:
                values = bootstrap_arrays[(variant, effect)]
                mask = effect_table["model"].eq(variant) & effect_table["effect"].eq(effect)
                effect_table.loc[mask, "paired_bootstrap_replicates"] = len(values)
                effect_table.loc[mask, "change_bootstrap_ci_low"] = float(np.quantile(values, 0.025))
                effect_table.loc[mask, "change_bootstrap_ci_high"] = float(np.quantile(values, 0.975))

        summary_rows = []
        full_three = float(full_effects.loc["Group_x_Stage_x_G1"])
        for meta_row in subfamilies.itertuples(index=False):
            variant = f"Minus {meta_row.subfamily}"
            effect_lookup = effect_table.loc[effect_table["model"].eq(variant)].set_index("effect")
            profile_lookup = profile_table.loc[profile_table["subfamily"].eq(meta_row.subfamily)].set_index("stage")
            three = effect_lookup.loc["Group_x_Stage_x_G1"]
            affected_profile = profile_lookup.loc[meta_row.stage]
            other_stage = "Outcome" if meta_row.stage == "Anticipation" else "Anticipation"
            other_profile = profile_lookup.loc[other_stage]
            retained_stage = (43 if meta_row.stage == "Anticipation" else 19) - int(meta_row.deleted_roi_count)
            row = {
                "subfamily": meta_row.subfamily,
                "stage": meta_row.stage,
                "deleted_roi_count": int(meta_row.deleted_roi_count),
                "retained_assigned_roi_count_total": 62 - int(meta_row.deleted_roi_count),
                "retained_roi_count_in_affected_stage": retained_stage,
                "removed_roi_names": meta_row.removed_roi_names,
                "removed_roi_indices0": meta_row.removed_roi_indices0,
                "other_stage": other_stage,
                "other_stage_numerically_unchanged": bool(other_profile["maximum_absolute_bin_difference"] < 1e-15),
                "full_threeway_effect": full_three,
                "threeway_effect": float(three["estimate"]),
                "threeway_CI95_low": float(three["CI95_low"]),
                "threeway_CI95_high": float(three["CI95_high"]),
                "threeway_Wald_p": float(three["p_wald"]),
                "threeway_change_from_full": float(three["change_from_full"]),
                "absolute_threeway_change": float(three["absolute_change_from_full"]),
                "threeway_delta_boot_ci_low": float(three["change_bootstrap_ci_low"]),
                "threeway_delta_boot_ci_high": float(three["change_bootstrap_ci_high"]),
                "affected_stage_profile_RMSE": float(affected_profile["profile_RMSE"]),
                "affected_stage_profile_max_abs_difference": float(affected_profile["maximum_absolute_bin_difference"]),
                "affected_stage_profile_correlation": float(affected_profile["profile_correlation"]),
                "effect_change_per_removed_roi": float(three["absolute_change_from_full"] / int(meta_row.deleted_roi_count)),
                "profile_RMSE_per_removed_roi": float(affected_profile["profile_RMSE"] / int(meta_row.deleted_roi_count)),
            }
            for effect in ["Anticipation_SZ_minus_HC_G1_slope", "Outcome_SZ_minus_HC_G1_slope"]:
                prefix = "anticipation" if effect.startswith("Anticipation") else "outcome"
                effect_row = effect_lookup.loc[effect]
                row[f"{prefix}_stage_specific_group_effect"] = float(effect_row["estimate"])
                row[f"{prefix}_effect_change_from_full"] = float(effect_row["change_from_full"])
                row[f"{prefix}_delta_boot_ci_low"] = float(effect_row["change_bootstrap_ci_low"])
                row[f"{prefix}_delta_boot_ci_high"] = float(effect_row["change_bootstrap_ci_high"])
            summary_rows.append(row)

        deletion_table = pd.DataFrame(summary_rows)
        deletion_table["absolute_effect_change_rank"] = deletion_table["absolute_threeway_change"].rank(
            method="min", ascending=False
        ).astype(int)
        deletion_table["profile_RMSE_rank"] = deletion_table["affected_stage_profile_RMSE"].rank(
            method="min", ascending=False
        ).astype(int)

        write_tsv(deletion_table, OUTPUT / "subfamily_deletion_results.tsv")
        write_tsv(effect_table, OUTPUT / "effect_size_stability.tsv")
        write_tsv(profile_table, OUTPUT / "profile_stability_metrics.tsv")
        write_tsv(model_table, OUTPUT / "model_fit_diagnostics.tsv")
        write_tsv(bootstrap_audit, OUTPUT / "paired_bootstrap_audit.tsv")
        build_reports(
            subfamilies,
            effect_table,
            profile_table,
            deletion_table,
            int(args.bootstrap_reps),
        )
        logging.info("All statistical outputs and reports written")
        logging.info("Total analysis elapsed seconds: %.3f", time.perf_counter() - started)
        logging.info("STEP 3A analysis complete; plotting is a separate Python step")
    except Exception:
        logging.exception("STEP 3A failed")
        error_report = (
            "# STEP 3A ERROR REPORT\n\n"
            "The original-model reproduction or a required downstream validation failed. "
            "Per the prespecified gate, no inferential conclusion should be drawn. "
            "See `STEP3A_EXECUTION_LOG.txt` for the exact traceback.\n"
        )
        (OUTPUT / "STEP3A_ANALYSIS_REPORT.md").write_text(error_report, encoding="utf-8")
        raise


if __name__ == "__main__":
    mp.freeze_support()
    main()
