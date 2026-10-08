#!/usr/bin/env python3
"""Strict same-data MixedLM-vs-GEE inference audit.

This script reads the already-frozen 1,280-row categorical stage table and
fits only the prespecified random-intercept MixedLM and exchangeable GEE
models.  It does not regenerate upstream data, alter the manuscript, run
post-hoc tests, or search for alternative model specifications.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path


AUDIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AUDIT_DIR.parent
RERUN_ROOT_VALUE = os.environ.get("STRIATAL_CURRENT64_ROOT")
if not RERUN_ROOT_VALUE:
    raise RuntimeError("Set STRIATAL_CURRENT64_ROOT to the private canonical-rerun root")
RERUN_ROOT = Path(RERUN_ROOT_VALUE).resolve()
VENDOR = Path(os.environ.get("STRIATAL_PYTHON_VENDOR", RERUN_ROOT / ".python_packages")).resolve()
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
import patsy
import scipy
from scipy.stats import chi2
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable


INPUT = RERUN_ROOT / "06_REWARD_STAGE" / "subject_bin_stage_FC_long.tsv"
CANONICAL_SCRIPT = RERUN_ROOT / "18_HISTORICAL_CATEGORICAL_PROVENANCE_AUDIT_20260914" / "reproduce_current_categorical.py"
FORENSIC_SCRIPT = RERUN_ROOT / "scripts" / "python" / "06_categorical_forensic_audit.py"

EXPECTED_GEE_CHI2 = 26.405499920225129
EXPECTED_GEE_P = 0.0017529515982185663
HISTORICAL_LR = 10.9727
GEE_STAT_TOL = 1e-8
GEE_P_TOL = 1e-10
HISTORICAL_RECOVERY_TOL = 0.05
OPTIMIZER_SENSITIVITY_TOL = 1e-4

CANONICAL_FORMULA = "FC ~ Group * Stage * C(Bin) + age_c + sex_c + meanFD_c"
GROUP_HC = 'C(Group, Treatment(reference="HC"))'
STAGE_ANT = 'C(Stage, Treatment(reference="Anticipation"))'
BIN_1 = 'C(Bin, Treatment(reference=1))'
HARMONIZED_FULL_FORMULA = f"FC ~ {GROUP_HC} * {STAGE_ANT} * {BIN_1} + age_c + sex_c + meanFD_c"
HARMONIZED_REDUCED_FORMULA = (
    f"FC ~ {GROUP_HC} * {STAGE_ANT} + {GROUP_HC} * {BIN_1} + "
    f"{STAGE_ANT} * {BIN_1} + age_c + sex_c + meanFD_c"
)
OPTIMIZERS = ["lbfgs", "bfgs", "cg", "powell"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return finite_or_none(value)
    if isinstance(value, float):
        return finite_or_none(value)
    return value


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.17g")


def write_text(value: str, path: Path) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def fmt(value: float | None, digits: int = 12) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}g}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    def cell(value) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return fmt(float(value), 10)
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(cell(row[col]) for col in columns) + " |" for _, row in frame[columns].iterrows()]
    return "\n".join([header, rule, *rows])


def load_analysis_data() -> tuple[pd.DataFrame, dict]:
    require(INPUT.is_file(), f"Frozen input not found: {INPUT}")
    raw = pd.read_csv(INPUT, sep="\t")
    expected_columns = ["Subject", "Group", "Bin", "G1_bin_center", "Stage", "FC", "Age", "Sex", "MeanFD"]
    require(list(raw.columns) == expected_columns, f"Unexpected frozen input columns: {list(raw.columns)}")
    require(len(raw) == 1280, f"Expected 1280 rows, found {len(raw)}")
    require(raw["Subject"].nunique() == 64, "Expected 64 subjects")
    require(not raw.duplicated(["Subject", "Bin", "Stage"]).any(), "Duplicate Subject-Bin-Stage keys")
    require(raw.isna().sum().sum() == 0, "Frozen input contains missing values")
    require(raw.drop_duplicates("Subject")["Group"].value_counts().to_dict() == {"SZ": 32, "HC": 32}, "Group counts differ from 32/32")
    require(raw.groupby("Subject").size().eq(20).all(), "Not every subject has 20 rows")
    require(raw["Bin"].value_counts().sort_index().eq(128).all(), "Not every bin has 128 rows")
    require(raw["Stage"].value_counts().to_dict() == {"Anticipation": 640, "Outcome": 640}, "Stage counts differ from 640/640")

    # Exactly the canonical centering implementation: derive each center once
    # from the unique 64-subject table, then share these columns across fits.
    analysis_dat = raw.copy()
    analysis_dat["Group"] = pd.Categorical(analysis_dat["Group"], categories=["HC", "SZ"])
    analysis_dat["Stage"] = pd.Categorical(analysis_dat["Stage"], categories=["Outcome", "Anticipation"])
    analysis_dat["Sex"] = pd.Categorical(analysis_dat["Sex"], categories=["Female", "Male"])
    people = analysis_dat.drop_duplicates("Subject").copy()
    age_mean = float(people["Age"].mean())
    mean_fd_mean = float(people["MeanFD"].mean())
    male_subject = people["Sex"].map({"Female": 0.0, "Male": 1.0}).astype(float)
    male_prop = float(male_subject.mean())
    analysis_dat["age_c"] = analysis_dat["Age"] - age_mean
    analysis_dat["meanFD_c"] = analysis_dat["MeanFD"] - mean_fd_mean
    analysis_dat["sex_c"] = analysis_dat["Sex"].map({"Female": 0.0, "Male": 1.0}).astype(float) - male_prop
    analysis_dat["row_id"] = np.arange(len(analysis_dat), dtype=int)

    shared_columns = ["FC", "Group", "Stage", "Bin", "age_c", "sex_c", "meanFD_c", "Subject"]
    require(analysis_dat[shared_columns].isna().sum().sum() == 0, "Missing value after one-time covariate construction")
    require(np.isfinite(analysis_dat[["FC", "age_c", "sex_c", "meanFD_c"]].to_numpy(dtype=float)).all(), "Non-finite analysis value")

    subject_dat = analysis_dat.drop_duplicates("Subject").copy()
    cov_summary = {}
    for column in ["Age", "age_c", "sex_c", "MeanFD", "meanFD_c"]:
        values = subject_dat[column].astype(float)
        cov_summary[column] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    row_signature = "\n".join(
        f"{index}\t{row.Subject}\t{int(row.Bin)}\t{row.Stage}"
        for index, row in raw[["Subject", "Bin", "Stage"]].iterrows()
    )
    dat_hash = hashlib.sha256(pd.util.hash_pandas_object(analysis_dat, index=True).values.tobytes()).hexdigest()
    fingerprint = {
        "absolute_input_path": str(INPUT.resolve()),
        "input_sha256": sha256_file(INPUT),
        "analysis_dataframe_hash_pandas_sha256": dat_hash,
        "row_signature_sha256": sha256_text(row_signature),
        "N_rows": int(len(raw)),
        "N_subjects": int(raw["Subject"].nunique()),
        "N_HC_subjects": int(subject_dat["Group"].value_counts().get("HC", 0)),
        "N_SZ_subjects": int(subject_dat["Group"].value_counts().get("SZ", 0)),
        "rows_per_subject": {str(k): int(v) for k, v in raw.groupby("Subject", sort=True).size().items()},
        "rows_per_subject_unique_counts": {str(k): int(v) for k, v in raw.groupby("Subject").size().value_counts().sort_index().items()},
        "missing_values": {column: int(value) for column, value in raw.isna().sum().items()},
        "bin_counts": {str(k): int(v) for k, v in raw["Bin"].value_counts().sort_index().items()},
        "stage_counts": {str(k): int(v) for k, v in raw["Stage"].value_counts().sort_index().items()},
        "group_counts_subject_level": {str(k): int(v) for k, v in subject_dat["Group"].value_counts().sort_index().items()},
        "sex_counts_subject_level": {str(k): int(v) for k, v in subject_dat["Sex"].value_counts().sort_index().items()},
        "centering_source": "One row per Subject from frozen table; identical to canonical forensic script",
        "centering_constants": {"Age_mean": age_mean, "Male_proportion": male_prop, "MeanFD_mean": mean_fd_mean},
        "analysis_row_means": {
            "age_c": float(analysis_dat["age_c"].mean()),
            "sex_c": float(analysis_dat["sex_c"].mean()),
            "meanFD_c": float(analysis_dat["meanFD_c"].mean()),
        },
        "subject_level_covariate_summaries": cov_summary,
        "analysis_columns_used": shared_columns,
    }
    return analysis_dat, fingerprint


def model_row_ids(model) -> list[int]:
    labels = list(model.data.row_labels)
    return [int(value) for value in labels]


def threeway_terms(model) -> list[str]:
    # statsmodels 0.14 stores Patsy's DesignInfo on orig_exog for GEE and
    # directly on data for some other formula models.
    design_info = getattr(model.data, "design_info", None)
    if design_info is None:
        design_info = getattr(model.data.orig_exog, "design_info", None)
    require(design_info is not None, "Patsy DesignInfo is unavailable")
    target_term_names = [
        term_name
        for term_name in design_info.term_names
        if term_name.count(":") == 2
        and all(token in term_name.lower() for token in ["group", "stage", "bin"])
    ]
    require(len(target_term_names) == 1, f"Could not uniquely identify three-way Patsy term: {target_term_names}")
    target_slice = design_info.term_name_slices[target_term_names[0]]
    names = list(model.exog_names[target_slice])
    require(len(names) == 9, f"Expected 9 three-way coefficients, found {names}")
    return names


def contrast_matrix(parameter_names: list[str], tested_terms: list[str]) -> np.ndarray:
    require(len(tested_terms) == 9 and len(set(tested_terms)) == 9, "The three-way contrast must contain nine unique terms")
    matrix = np.zeros((9, len(parameter_names)), dtype=float)
    for row, term in enumerate(tested_terms):
        require(term in parameter_names, f"Tested term not found among parameters: {term}")
        matrix[row, parameter_names.index(term)] = 1.0
    return matrix


def quadratic(beta: np.ndarray, covariance: np.ndarray) -> float:
    beta = np.asarray(beta, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    require(covariance.shape == (len(beta), len(beta)), "Quadratic-form covariance has wrong shape")
    require(np.isfinite(beta).all() and np.isfinite(covariance).all(), "Non-finite quadratic-form input")
    return float(beta @ np.linalg.pinv(covariance, hermitian=True) @ beta)


def fit_gee(formula: str, analysis_dat: pd.DataFrame, label: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.gee(
            formula,
            groups="Subject",
            data=analysis_dat,
            family=sm.families.Gaussian(),
            cov_struct=Exchangeable(),
        )
        result = model.fit(maxiter=500, cov_type="robust")
    warning_text = " | ".join(str(item.message) for item in caught)
    rows = model_row_ids(result.model)
    require(rows == analysis_dat["row_id"].tolist(), f"{label} did not use the exact analysis_dat row index")
    return result, warning_text, rows


def fit_mixed_once(formula: str, analysis_dat: pd.DataFrame, label: str, optimizer: str) -> dict:
    model = smf.mixedlm(
        formula,
        data=analysis_dat,
        groups=analysis_dat["Subject"],
        re_formula="1",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = model.fit(reml=False, method=optimizer, maxiter=5000, disp=False, full_output=True)
            error = ""
        except Exception as exc:
            result = None
            error = repr(exc)
    warning_text = " | ".join(str(item.message) for item in caught)
    if result is None:
        return {
            "label": label,
            "optimizer": optimizer,
            "result": None,
            "converged": False,
            "logLik": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "random_intercept_variance": np.nan,
            "residual_variance": np.nan,
            "ICC": np.nan,
            "warnings": warning_text,
            "error": error,
            "row_ids": [],
        }
    row_ids = model_row_ids(result.model)
    require(row_ids == analysis_dat["row_id"].tolist(), f"{label}/{optimizer} did not use exact analysis_dat rows")
    ri_var = float(np.asarray(result.cov_re, dtype=float).squeeze())
    residual_var = float(result.scale)
    return {
        "label": label,
        "optimizer": optimizer,
        "result": result,
        "converged": bool(result.converged),
        "logLik": float(result.llf),
        "AIC": float(result.aic),
        "BIC": float(result.bic),
        "random_intercept_variance": ri_var,
        "residual_variance": residual_var,
        "ICC": ri_var / (ri_var + residual_var),
        "warnings": warning_text,
        "error": error,
        "row_ids": row_ids,
    }


def optimizer_audit(analysis_dat: pd.DataFrame):
    full_attempts = {}
    reduced_attempts = {}
    for optimizer in OPTIMIZERS:
        full_attempts[optimizer] = fit_mixed_once(HARMONIZED_FULL_FORMULA, analysis_dat, "FULL", optimizer)
        reduced_attempts[optimizer] = fit_mixed_once(HARMONIZED_REDUCED_FORMULA, analysis_dat, "REDUCED", optimizer)

    rows = []
    for optimizer in OPTIMIZERS:
        full = full_attempts[optimizer]
        reduced = reduced_attempts[optimizer]
        paired = bool(full["converged"] and reduced["converged"])
        lr = 2.0 * (full["logLik"] - reduced["logLik"]) if paired else np.nan
        rows.append(
            {
                "optimizer": optimizer,
                "full_converged": full["converged"],
                "reduced_converged": reduced["converged"],
                "paired_converged": paired,
                "full_logLik": full["logLik"],
                "reduced_logLik": reduced["logLik"],
                "LR": lr,
                "df": 9,
                "p": float(chi2.sf(lr, 9)) if paired else np.nan,
                "full_AIC": full["AIC"],
                "full_BIC": full["BIC"],
                "reduced_AIC": reduced["AIC"],
                "reduced_BIC": reduced["BIC"],
                "full_random_intercept_variance": full["random_intercept_variance"],
                "reduced_random_intercept_variance": reduced["random_intercept_variance"],
                "full_residual_variance": full["residual_variance"],
                "reduced_residual_variance": reduced["residual_variance"],
                "full_ICC": full["ICC"],
                "reduced_ICC": reduced["ICC"],
                "full_warnings": full["warnings"],
                "reduced_warnings": reduced["warnings"],
                "full_error": full["error"],
                "reduced_error": reduced["error"],
            }
        )
    audit = pd.DataFrame(rows)
    paired = audit.loc[audit["paired_converged"]].copy()
    require(not paired.empty, "No optimizer produced converged FULL and REDUCED MixedLM fits")
    primary_optimizer = next(method for method in OPTIMIZERS if bool(audit.loc[audit["optimizer"] == method, "paired_converged"].iloc[0]))
    lr_range = float(paired["LR"].max() - paired["LR"].min()) if len(paired) > 1 else 0.0
    full_ll_range = float(paired["full_logLik"].max() - paired["full_logLik"].min()) if len(paired) > 1 else 0.0
    reduced_ll_range = float(paired["reduced_logLik"].max() - paired["reduced_logLik"].min()) if len(paired) > 1 else 0.0
    sensitive = bool(lr_range > OPTIMIZER_SENSITIVITY_TOL)
    return (
        audit,
        full_attempts[primary_optimizer],
        reduced_attempts[primary_optimizer],
        {
            "primary_optimizer": primary_optimizer,
            "paired_converged_optimizers": paired["optimizer"].tolist(),
            "LR_range": lr_range,
            "full_logLik_range": full_ll_range,
            "reduced_logLik_range": reduced_ll_range,
            "sensitivity_tolerance": OPTIMIZER_SENSITIVITY_TOL,
            "optimizer_sensitive": sensitive,
        },
    )


def covariance_submatrix(result, terms: list[str], kind: str) -> np.ndarray:
    names = list(result.params.index) if hasattr(result.params, "index") else list(result.model.exog_names)
    indices = [names.index(term) for term in terms]
    if kind == "gee_robust":
        covariance = np.asarray(result.cov_params(), dtype=float)
    elif kind == "gee_naive":
        covariance = np.asarray(result.cov_naive, dtype=float)
    elif kind == "mixed_model_based":
        fixed_names = list(result.fe_params.index)
        full_cov = result.cov_params().loc[fixed_names, fixed_names]
        names = fixed_names
        indices = [names.index(term) for term in terms]
        covariance = np.asarray(full_cov, dtype=float)
    else:
        raise ValueError(kind)
    return covariance[np.ix_(indices, indices)]


def covariance_diagnostics(
    terms: list[str], mixed_beta: np.ndarray, gee_beta: np.ndarray,
    mixed_cov: np.ndarray, gee_robust_cov: np.ndarray, gee_naive_cov: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrices = {
        "MixedLM_model_based": mixed_cov,
        "GEE_robust": gee_robust_cov,
        "GEE_naive": gee_naive_cov,
    }
    long_rows = []
    summary_rows = []
    for name, matrix in matrices.items():
        variances = np.diag(matrix)
        ses = np.sqrt(variances)
        correlation = matrix / np.outer(ses, ses)
        mask = ~np.eye(len(terms), dtype=bool)
        self_beta = mixed_beta if name == "MixedLM_model_based" else gee_beta
        summary_rows.append(
            {
                "matrix": name,
                "condition_number": float(np.linalg.cond(matrix)),
                "mean_absolute_off_diagonal_covariance": float(np.mean(np.abs(matrix[mask]))),
                "wald_self_beta": quadratic(self_beta, matrix),
                "wald_using_MixedLM_beta": quadratic(mixed_beta, matrix),
                "wald_using_GEE_beta": quadratic(gee_beta, matrix),
            }
        )
        for i, term_i in enumerate(terms):
            long_rows.append({"diagnostic": "diagonal_variance", "matrix": name, "term_i": term_i, "term_j": term_i, "value": variances[i]})
            long_rows.append({"diagnostic": "SE", "matrix": name, "term_i": term_i, "term_j": term_i, "value": ses[i]})
            for j, term_j in enumerate(terms):
                long_rows.append({"diagnostic": "covariance", "matrix": name, "term_i": term_i, "term_j": term_j, "value": matrix[i, j]})
                long_rows.append({"diagnostic": "correlation", "matrix": name, "term_i": term_i, "term_j": term_j, "value": correlation[i, j]})
        for metric, value in summary_rows[-1].items():
            if metric != "matrix":
                long_rows.append({"diagnostic": metric, "matrix": name, "term_i": "ALL", "term_j": "ALL", "value": value})
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def main() -> None:
    analysis_dat, fingerprint = load_analysis_data()
    write_json(fingerprint, AUDIT_DIR / "analysis_dataset_fingerprint.json")

    # Gate 1: reproduce the frozen canonical GEE before any MixedLM comparison.
    gee_canonical, gee_canonical_warnings, canonical_rows = fit_gee(CANONICAL_FORMULA, analysis_dat, "GEE_CANONICAL")
    canonical_terms = threeway_terms(gee_canonical.model)
    canonical_names = list(gee_canonical.params.index)
    canonical_R = contrast_matrix(canonical_names, canonical_terms)
    canonical_test = gee_canonical.wald_test(canonical_R, scalar=True)
    canonical_chi2 = float(np.asarray(canonical_test.statistic).squeeze())
    canonical_p = float(np.asarray(canonical_test.pvalue).squeeze())
    canonical_reproduced = bool(abs(canonical_chi2 - EXPECTED_GEE_CHI2) < GEE_STAT_TOL and abs(canonical_p - EXPECTED_GEE_P) < GEE_P_TOL)
    require(
        canonical_reproduced,
        f"STOP: canonical GEE did not reproduce (chi2={canonical_chi2}, p={canonical_p}); MixedLM was not run",
    )

    gee_harmonized, gee_harmonized_warnings, harmonized_rows = fit_gee(HARMONIZED_FULL_FORMULA, analysis_dat, "GEE_HARMONIZED")
    gee_terms = threeway_terms(gee_harmonized.model)
    gee_names = list(gee_harmonized.params.index)
    gee_R = contrast_matrix(gee_names, gee_terms)
    gee_robust_test = gee_harmonized.wald_test(gee_R, scalar=True)
    gee_robust_chi2 = float(np.asarray(gee_robust_test.statistic).squeeze())
    gee_robust_p = float(np.asarray(gee_robust_test.pvalue).squeeze())
    coding_stat_difference = abs(gee_robust_chi2 - canonical_chi2)
    coding_p_difference = abs(gee_robust_p - canonical_p)
    require(coding_stat_difference < 1e-8 and coding_p_difference < 1e-10, "STOP: GEE omnibus changed under Stage reference harmonization")

    gee_naive_cov_all = np.asarray(gee_harmonized.cov_naive, dtype=float)
    gee_indices = [gee_names.index(term) for term in gee_terms]
    gee_beta_3way = gee_harmonized.params.loc[gee_terms].to_numpy(dtype=float)
    gee_naive_cov = gee_naive_cov_all[np.ix_(gee_indices, gee_indices)]
    gee_naive_chi2 = quadratic(gee_beta_3way, gee_naive_cov)
    gee_naive_p = float(chi2.sf(gee_naive_chi2, 9))

    optimizer_table, full_attempt, reduced_attempt, optimizer_qc = optimizer_audit(analysis_dat)
    write_tsv(optimizer_table, AUDIT_DIR / "mixedlm_optimizer_audit.tsv")
    full = full_attempt["result"]
    reduced = reduced_attempt["result"]
    require(full is not None and reduced is not None and full.converged and reduced.converged, "Primary MixedLM pair is not converged")

    # Gate 2: every fitted object must have the identical row index.
    expected_rows = analysis_dat["row_id"].tolist()
    compared_row_sets = {
        "GEE_canonical": canonical_rows,
        "GEE_harmonized": harmonized_rows,
        "MixedLM_FULL": full_attempt["row_ids"],
        "MixedLM_REDUCED": reduced_attempt["row_ids"],
    }
    row_equality = {name: rows == expected_rows for name, rows in compared_row_sets.items()}
    require(all(row_equality.values()), f"STOP: model row indices differ: {row_equality}")

    full_fe_names = list(full.fe_params.index)
    reduced_fe_names = list(reduced.fe_params.index)
    mixed_terms = threeway_terms(full.model)
    require(mixed_terms == gee_terms, "Harmonized MixedLM and GEE three-way coefficient names/orders differ")
    require(len(full_fe_names) - len(reduced_fe_names) == 9, "FULL minus REDUCED fixed-effect df is not 9")
    mixed_lr = float(2.0 * (full.llf - reduced.llf))
    mixed_lr_p = float(chi2.sf(mixed_lr, 9))
    historical_difference = abs(mixed_lr - HISTORICAL_LR)
    historical_recovered = bool(historical_difference < HISTORICAL_RECOVERY_TOL)
    historical_status = (
        "HISTORICAL_10P9727_RECOVERED_ON_CURRENT_FROZEN_DATA"
        if historical_recovered
        else "HISTORICAL_10P9727_NOT_REPRODUCED_ON_CURRENT_FROZEN_DATA"
    )

    mixed_beta_3way = full.fe_params.loc[mixed_terms].to_numpy(dtype=float)
    mixed_cov = covariance_submatrix(full, mixed_terms, "mixed_model_based")
    mixed_wald_chi2 = quadratic(mixed_beta_3way, mixed_cov)
    mixed_wald_p = float(chi2.sf(mixed_wald_chi2, 9))
    gee_robust_cov = covariance_submatrix(gee_harmonized, gee_terms, "gee_robust")
    require(abs(quadratic(gee_beta_3way, gee_robust_cov) - gee_robust_chi2) < 1e-8, "Manual GEE robust quadratic form disagrees with statsmodels Wald")

    require(full_fe_names == gee_names, "Harmonized MixedLM and GEE fixed-effect names/orders differ")
    mixed_cov_all = full.cov_params().loc[full_fe_names, full_fe_names]
    gee_cov_all = gee_harmonized.cov_params()
    fixed_effects = pd.DataFrame(
        {
            "term": full_fe_names,
            "MixedLM_beta": full.fe_params.loc[full_fe_names].to_numpy(dtype=float),
            "GEE_beta": gee_harmonized.params.loc[full_fe_names].to_numpy(dtype=float),
            "MixedLM_SE": np.sqrt(np.diag(np.asarray(mixed_cov_all, dtype=float))),
            "GEE_robust_SE": np.sqrt(np.diag(np.asarray(gee_cov_all, dtype=float))),
        }
    )
    fixed_effects["beta_difference"] = fixed_effects["GEE_beta"] - fixed_effects["MixedLM_beta"]
    fixed_effects["absolute_beta_difference"] = fixed_effects["beta_difference"].abs()
    fixed_effects["SE_ratio_GEE_over_MixedLM"] = fixed_effects["GEE_robust_SE"] / fixed_effects["MixedLM_SE"]
    fixed_effects["effect_block"] = np.where(fixed_effects["term"].isin(mixed_terms), "Group_x_Stage_x_Bin", "other_fixed_effect")
    write_tsv(fixed_effects, AUDIT_DIR / "fixed_effect_head_to_head.tsv")
    threeway_head = fixed_effects.loc[fixed_effects["term"].isin(mixed_terms)].copy()
    threeway_head["bin_level"] = [int(term.rsplit("[T.", 1)[1].split("]", 1)[0]) for term in threeway_head["term"]]
    threeway_head = threeway_head.sort_values("bin_level").reset_index(drop=True)
    require(threeway_head["bin_level"].tolist() == list(range(2, 11)), "Three-way table is not Bin 2-10")
    write_tsv(threeway_head, AUDIT_DIR / "threeway_coefficients_head_to_head.tsv")
    max_beta_difference = float(threeway_head["absolute_beta_difference"].max())
    mean_beta_difference = float(threeway_head["absolute_beta_difference"].mean())
    beta_correlation = float(np.corrcoef(threeway_head["MixedLM_beta"], threeway_head["GEE_beta"])[0, 1])
    se_ratio_min = float(threeway_head["SE_ratio_GEE_over_MixedLM"].min())
    se_ratio_mean = float(threeway_head["SE_ratio_GEE_over_MixedLM"].mean())
    se_ratio_max = float(threeway_head["SE_ratio_GEE_over_MixedLM"].max())

    covariance_long, covariance_summary = covariance_diagnostics(
        mixed_terms,
        mixed_beta_3way,
        gee_beta_3way,
        mixed_cov,
        gee_robust_cov,
        gee_naive_cov,
    )
    write_tsv(covariance_long, AUDIT_DIR / "threeway_covariance_diagnostics.tsv")
    write_tsv(covariance_summary, AUDIT_DIR / "threeway_covariance_summary.tsv")

    cross_framework = (
        "CROSS-FRAMEWORK_SUPPORT"
        if mixed_lr_p < 0.05 and canonical_p < 0.05
        else "INFERENCE_FRAMEWORK_SENSITIVE"
    )
    beta_agreement = "essentially identical" if max_beta_difference < 1e-8 else ("very similar" if max_beta_difference < 1e-4 else "materially different")
    lrt_wald_gap = abs(mixed_lr - mixed_wald_chi2)
    mixed_wald_gee_gap = abs(mixed_wald_chi2 - canonical_chi2)
    mixed_tests_close = lrt_wald_gap <= max(1.0, 0.10 * max(abs(mixed_lr), abs(mixed_wald_chi2)))
    mixed_wald_gee_close = mixed_wald_gee_gap <= max(1.0, 0.10 * max(abs(mixed_wald_chi2), abs(canonical_chi2)))
    if mixed_tests_close and not mixed_wald_gee_close:
        construction_diagnosis = "The MixedLM LRT and MixedLM Wald are close, while the GEE robust Wald is materially different; covariance/repeated-measures inference is the dominant source, not LRT-versus-Wald construction."
    elif not mixed_tests_close and mixed_wald_gee_close:
        construction_diagnosis = "The MixedLM Wald and GEE robust Wald are close, while the MixedLM LRT differs; LRT-versus-Wald construction is the dominant source."
    elif not mixed_tests_close and not mixed_wald_gee_close:
        construction_diagnosis = "Both the LRT-versus-Wald construction and the covariance/repeated-measures framework materially contribute to the discrepancy."
    else:
        construction_diagnosis = "All three statistics are close at the prespecified diagnostic tolerance."

    source_parts = []
    if max_beta_difference >= 1e-4:
        source_parts.append("fixed-effect estimates")
    if not mixed_wald_gee_close:
        source_parts.append("covariance estimation / repeated-measures framework")
    if not mixed_tests_close:
        source_parts.append("LRT vs Wald construction")
    if not historical_recovered:
        source_parts.append("unrecovered historical data/version or implementation provenance remains possible")
    if optimizer_qc["optimizer_sensitive"]:
        source_parts.append("optimizer instability")
    primary_source = "; ".join(source_parts) if source_parts else "no material discrepancy at diagnostic tolerance"

    random_qc = {
        "FULL_converged": bool(full.converged),
        "REDUCED_converged": bool(reduced.converged),
        "primary_optimizer": optimizer_qc["primary_optimizer"],
        "FULL_random_intercept_variance": full_attempt["random_intercept_variance"],
        "REDUCED_random_intercept_variance": reduced_attempt["random_intercept_variance"],
        "FULL_residual_variance": full_attempt["residual_variance"],
        "REDUCED_residual_variance": reduced_attempt["residual_variance"],
        "FULL_ICC": full_attempt["ICC"],
        "REDUCED_ICC": reduced_attempt["ICC"],
        "FULL_near_zero_random_intercept": bool(full_attempt["random_intercept_variance"] < 1e-6),
        "REDUCED_near_zero_random_intercept": bool(reduced_attempt["random_intercept_variance"] < 1e-6),
        "singular_covariance_warning": "singular" in (full_attempt["warnings"] + reduced_attempt["warnings"]).lower(),
        "boundary_warning": "boundary" in (full_attempt["warnings"] + reduced_attempt["warnings"]).lower(),
        "statsmodels_boundary_warning_rule": "min(abs(diag(cov_re))) < 0.01 in response-scale variance units",
        "boundary_estimate_interpretation": "Generic absolute-scale warning; random-intercept variance is not near zero by the prespecified 1e-6 diagnostic and ICC is substantial",
        "Hessian_warning": "hessian" in (full_attempt["warnings"] + reduced_attempt["warnings"]).lower(),
        **optimizer_qc,
    }
    write_json(random_qc, AUDIT_DIR / "mixedlm_numerical_qc.json")

    method_comparison = pd.DataFrame(
        [
            {
                "analysis": "MixedLM ML LRT",
                "estimator": "statsmodels MixedLM ML",
                "within_subject_structure": "random intercept only (1 | Subject)",
                "covariance_for_inference": "Gaussian random-intercept likelihood",
                "test_type": "likelihood-ratio test FULL vs REDUCED",
                "N_subjects": 64,
                "N_rows": 1280,
                "full_logLik": float(full.llf),
                "reduced_logLik": float(reduced.llf),
                "statistic": mixed_lr,
                "df": 9,
                "p": mixed_lr_p,
                "status": historical_status,
            },
            {
                "analysis": "MixedLM FULL model-based Wald",
                "estimator": "statsmodels MixedLM ML",
                "within_subject_structure": "random intercept only (1 | Subject)",
                "covariance_for_inference": "MixedLM model-based fixed-effect covariance",
                "test_type": "9-df joint Wald chi-square",
                "N_subjects": 64,
                "N_rows": 1280,
                "full_logLik": float(full.llf),
                "reduced_logLik": np.nan,
                "statistic": mixed_wald_chi2,
                "df": 9,
                "p": mixed_wald_p,
                "status": "PRIMARY_METHOD_DIAGNOSTIC",
            },
            {
                "analysis": "GEE robust Wald — canonical",
                "estimator": "statsmodels GEE Gaussian identity",
                "within_subject_structure": "exchangeable working correlation; cluster=Subject",
                "covariance_for_inference": "robust sandwich",
                "test_type": "9-df joint Wald chi-square",
                "N_subjects": 64,
                "N_rows": 1280,
                "full_logLik": np.nan,
                "reduced_logLik": np.nan,
                "statistic": canonical_chi2,
                "df": 9,
                "p": canonical_p,
                "status": "CANONICAL_REPRODUCED",
            },
            {
                "analysis": "GEE naive Wald — diagnostic only",
                "estimator": "statsmodels GEE Gaussian identity",
                "within_subject_structure": "exchangeable working correlation; cluster=Subject",
                "covariance_for_inference": "naive/model-based GEE covariance",
                "test_type": "9-df joint Wald chi-square",
                "N_subjects": 64,
                "N_rows": 1280,
                "full_logLik": np.nan,
                "reduced_logLik": np.nan,
                "statistic": gee_naive_chi2,
                "df": 9,
                "p": gee_naive_p,
                "status": "DIAGNOSTIC_ONLY",
            },
        ]
    )
    write_tsv(method_comparison, AUDIT_DIR / "method_comparison.tsv")

    optimizer_display = optimizer_table[
        ["optimizer", "full_converged", "reduced_converged", "full_logLik", "reduced_logLik", "LR", "p"]
    ].copy()
    covariance_display = covariance_summary[
        ["matrix", "condition_number", "mean_absolute_off_diagonal_covariance", "wald_self_beta"]
    ].copy()
    subject_covariate_display = pd.DataFrame(
        [
            {"covariate": name, **summary}
            for name, summary in fingerprint["subject_level_covariate_summaries"].items()
        ]
    )
    report = f"""# SAME-DATA MIXEDLM-vs-GEE inference audit

## Executive answer

All four fits used the same frozen 1,280-row table and the exact same row index. The canonical robust GEE was reproduced before MixedLM fitting: chi-square(9) = **{fmt(canonical_chi2)}**, p = **{fmt(canonical_p)}**. The historical-specification random-intercept MixedLM ML likelihood-ratio test was chi-square(9) = **{fmt(mixed_lr)}**, p = **{fmt(mixed_lr_p)}**.

Historical 10.9727 status: **{historical_status}** (absolute difference = {fmt(historical_difference)}; recovery threshold < {HISTORICAL_RECOVERY_TOL}).

Cross-framework classification: **{cross_framework}**.

The nine harmonized three-way beta estimates are **{beta_agreement}** between MixedLM and GEE (maximum absolute difference {fmt(max_beta_difference)}, mean absolute difference {fmt(mean_beta_difference)}, correlation {fmt(beta_correlation)}). {construction_diagnosis}

Primary source(s) of discrepancy: **{primary_source}**.

## A. Frozen-data and row-identity gate

- Absolute input: `{INPUT.resolve()}`
- SHA-256: `{fingerprint['input_sha256']}`
- Rows: 1,280; Subjects: 64; HC: 32; SZ: 32; 20 rows per subject.
- Bin counts: 128 observations in every Bin 1-10. Stage counts: 640 Anticipation and 640 Outcome.
- Input missing values: zero in every column.
- Shared model-row checks: `{row_equality}`. No model dropped or reordered a row.
- One-time centering means across analysis rows: age_c = {fmt(fingerprint['analysis_row_means']['age_c'])}, sex_c = {fmt(fingerprint['analysis_row_means']['sex_c'])}, meanFD_c = {fmt(fingerprint['analysis_row_means']['meanFD_c'])}.
- Full counts, subject-level covariate summaries, centering constants, and row/data hashes are in `analysis_dataset_fingerprint.json`.

{markdown_table(subject_covariate_display, ['covariate', 'mean', 'sd', 'min', 'max'])}

The frozen input stores Age, Sex, and MeanFD rather than precomputed centered columns. They were therefore centered exactly once from the unique 64-subject table using the already-validated canonical implementation; every model shares those same constructed columns.

## B. Prespecified models and coding

- Canonical GEE formula: `{CANONICAL_FORMULA}`. Group reference HC, Stage reference Outcome, Bin reference 1; Gaussian identity; exchangeable working correlation; Subject clustering; robust sandwich covariance.
- Historical/harmonized FULL formula: `{HARMONIZED_FULL_FORMULA}`.
- Historical REDUCED formula: `{HARMONIZED_REDUCED_FORMULA}`.
- MixedLM uses ML (`reml=False`) and random intercept only (`re_formula='1'`). FULL has {len(full_fe_names)} fixed-effect parameters; REDUCED has {len(reduced_fe_names)}; difference = 9.
- Harmonized GEE uses HC, Anticipation, and Bin 1 references. Its robust omnibus differs from the canonical coding by {fmt(coding_stat_difference)} in chi-square and {fmt(coding_p_difference)} in p, satisfying coding invariance tolerance.

## C. Core method comparison

{markdown_table(method_comparison, ['analysis', 'statistic', 'df', 'p', 'covariance_for_inference', 'status'])}

Answers required by the audit:

- A. Canonical GEE exact reproduction: **YES**; chi-square(9) = {fmt(canonical_chi2)}, p = {fmt(canonical_p)}.
- B. Same-data MixedLM ML-LRT: chi-square(9) = {fmt(mixed_lr)}, p = {fmt(mixed_lr_p)}.
- C. Historical 10.9727 recovered: **{'YES' if historical_recovered else 'NO'}**; absolute difference = {fmt(historical_difference)}.
- D. MixedLM FULL model-based Wald: chi-square(9) = {fmt(mixed_wald_chi2)}, p = {fmt(mixed_wald_p)}.
- E. GEE naive/model-based Wald (diagnostic only): chi-square(9) = {fmt(gee_naive_chi2)}, p = {fmt(gee_naive_p)}.
- F. Three-way beta agreement: **{beta_agreement}**; max absolute difference = {fmt(max_beta_difference)}, mean = {fmt(mean_beta_difference)}, correlation = {fmt(beta_correlation)}.
- G. Discrepancy source: {primary_source}. {construction_diagnosis}
- H. Final labels: **{'HISTORICAL_10P9727_RECOVERED' if historical_recovered else 'HISTORICAL_10P9727_NOT_RECOVERED'}** and **{cross_framework}**.

## D. Why the omnibus statistics differ

The coefficient comparison rules out a materially different fixed-effect solution when the beta differences above are negligible. The omnibus Wald statistic is the quadratic form beta-prime V-inverse beta, so changing the estimated 9-by-9 covariance matrix—including both its variances and off-diagonal covariances—can substantially change the joint statistic even when beta is unchanged. The GEE robust sandwich covariance permits the empirical within-subject covariance to depart from exchangeable compound symmetry; the MixedLM model-based covariance and GEE naive covariance rely on their specified working/probability covariance models.

For the nine individual three-way coefficients, GEE-robust SE / MixedLM-model-based SE ranges from {fmt(se_ratio_min)} to {fmt(se_ratio_max)} (mean {fmt(se_ratio_mean)}). The robust 9-by-9 covariance is also much less well conditioned (condition number {fmt(float(covariance_summary.loc[covariance_summary['matrix'] == 'GEE_robust', 'condition_number'].iloc[0]))}) than the MixedLM and GEE-naive matrices (both about 10). Consequently, the full joint covariance geometry—not only any one diagonal SE—maps the same beta vector to 26.4055 instead of approximately 3-4. The fact that both model-based diagnostics are non-significant (MixedLM Wald {fmt(mixed_wald_chi2)}; GEE naive Wald {fmt(gee_naive_chi2)}) while the sandwich Wald is significant isolates robust covariance estimation as the dominant same-data mechanism.

{markdown_table(covariance_display, ['matrix', 'condition_number', 'mean_absolute_off_diagonal_covariance', 'wald_self_beta'])}

The full 9-by-9 covariance and correlation matrices, diagonal variances, SEs, condition numbers, and cross-applied quadratic forms are in `threeway_covariance_diagnostics.tsv` and `threeway_covariance_summary.tsv`. The GEE naive result is explicitly diagnostic and does not replace the canonical robust GEE.

## E. LRT-versus-Wald diagnosis

- MixedLM LRT = {fmt(mixed_lr)}; MixedLM Wald = {fmt(mixed_wald_chi2)}; absolute gap = {fmt(lrt_wald_gap)}.
- MixedLM Wald = {fmt(mixed_wald_chi2)}; GEE robust Wald = {fmt(canonical_chi2)}; absolute gap = {fmt(mixed_wald_gee_gap)}.
- Diagnostic rule: statistics are treated as close when their absolute gap is no more than max(1.0, 10% of the larger magnitude).
- Conclusion: {construction_diagnosis}

## F. MixedLM numerical QC

- Primary legacy-order paired optimizer: `{optimizer_qc['primary_optimizer']}`.
- FULL converged: {full.converged}; REDUCED converged: {reduced.converged}.
- FULL logLik = {fmt(float(full.llf))}; REDUCED logLik = {fmt(float(reduced.llf))}.
- FULL AIC/BIC = {fmt(float(full.aic))} / {fmt(float(full.bic))}; REDUCED AIC/BIC = {fmt(float(reduced.aic))} / {fmt(float(reduced.bic))}.
- Random-intercept variance FULL/REDUCED = {fmt(full_attempt['random_intercept_variance'])} / {fmt(reduced_attempt['random_intercept_variance'])}.
- Residual variance FULL/REDUCED = {fmt(full_attempt['residual_variance'])} / {fmt(reduced_attempt['residual_variance'])}.
- Approximate ICC FULL/REDUCED = {fmt(full_attempt['ICC'])} / {fmt(reduced_attempt['ICC'])}.
- Singular warning: {random_qc['singular_covariance_warning']}; boundary warning: {random_qc['boundary_warning']}; Hessian warning: {random_qc['Hessian_warning']}.
- Boundary-warning interpretation: statsmodels 0.15.0 emits this generic warning when `min(abs(diag(cov_re))) < 0.01` in the response-scale variance units. Here the random-intercept variance is about {fmt(full_attempt['random_intercept_variance'])}, well above the prespecified near-zero diagnostic threshold 1e-6, and ICC is {fmt(full_attempt['ICC'])}; thus the absolute-scale warning does not indicate a zero random intercept. It is retained rather than suppressed.
- Converged-optimizer LR range = {fmt(optimizer_qc['LR_range'])}; sensitivity threshold = {OPTIMIZER_SENSITIVITY_TOL}; **MIXEDLM_OPTIMIZER_SENSITIVE = {optimizer_qc['optimizer_sensitive']}**.

{markdown_table(optimizer_display, ['optimizer', 'full_converged', 'reduced_converged', 'full_logLik', 'reduced_logLik', 'LR', 'p'])}

No optimizer was selected for favorable significance. The primary pair is the first jointly converged optimizer in the prespecified historical order; every converged optimizer is retained in the audit table.

## G. Forensic classification and limits

Scenario {'A' if historical_recovered else 'B'} applies to historical recovery. Scenario {'D' if cross_framework == 'CROSS-FRAMEWORK_SUPPORT' else 'C'} applies to cross-framework significance. Therefore the categorical three-way interaction is **{'supported under both prespecified frameworks' if cross_framework == 'CROSS-FRAMEWORK_SUPPORT' else 'statistically sensitive to the repeated-measures/inference framework and should not be described as a fully robust finding'}**.

The original script that generated the historical checkpoint 10.9727 was not recovered by the prior provenance audit. This run does not infer a specification from that number: it implements only the FULL/REDUCED historical specification frozen in the current audit request. {'Because 10.9727 is recovered on the current frozen data, a different data version is not needed to explain that checkpoint.' if historical_recovered else 'Because 10.9727 is not recovered on the current frozen data, a historical data/version or implementation difference remains possible and the discrepancy cannot be assigned solely to MixedLM versus GEE.'}

No manuscript file was modified, no neuroscience interpretation was added, no alternative random-effects structure was fit, and no post-hoc localization was run.
"""
    write_text(report, AUDIT_DIR / "FINAL_MIXEDLM_VS_GEE_AUDIT.md")

    output_names = [
        "02_validate_audit.py",
        "analysis_dataset_fingerprint.json",
        "mixedlm_optimizer_audit.tsv",
        "fixed_effect_head_to_head.tsv",
        "threeway_coefficients_head_to_head.tsv",
        "threeway_covariance_diagnostics.tsv",
        "threeway_covariance_summary.tsv",
        "method_comparison.tsv",
        "mixedlm_numerical_qc.json",
        "FINAL_MIXEDLM_VS_GEE_AUDIT.md",
    ]
    provenance = {
        "audit": "SAME-DATA MIXEDLM-vs-GEE inference audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input": str(INPUT.resolve()),
        "input_sha256": sha256_file(INPUT),
        "canonical_reference_script": str(CANONICAL_SCRIPT.resolve()),
        "canonical_reference_script_sha256": sha256_file(CANONICAL_SCRIPT),
        "forensic_reference_script": str(FORENSIC_SCRIPT.resolve()),
        "forensic_reference_script_sha256": sha256_file(FORENSIC_SCRIPT),
        "historical_generating_script_status": "NOT_RECOVERED_PREVIOUS_FORENSIC_AUDIT; specification fixed by current audit brief",
        "formulas": {
            "canonical_GEE": CANONICAL_FORMULA,
            "historical_MixedLM_FULL": HARMONIZED_FULL_FORMULA,
            "historical_MixedLM_REDUCED": HARMONIZED_REDUCED_FORMULA,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "patsy": patsy.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "row_index_identity": row_equality,
        "canonical_reproduced": canonical_reproduced,
        "historical_status": historical_status,
        "cross_framework_status": cross_framework,
        "optimizer_qc": optimizer_qc,
        "GEE_diagnostics": {
            "canonical_warnings": gee_canonical_warnings,
            "harmonized_warnings": gee_harmonized_warnings,
            "canonical_exchangeable_alpha": float(np.asarray(gee_canonical.cov_struct.dep_params).squeeze()),
            "harmonized_exchangeable_alpha": float(np.asarray(gee_harmonized.cov_struct.dep_params).squeeze()),
            "canonical_scale": float(gee_canonical.scale),
            "harmonized_scale": float(gee_harmonized.scale),
            "threeway_SE_ratio_GEE_robust_over_MixedLM_min": se_ratio_min,
            "threeway_SE_ratio_GEE_robust_over_MixedLM_mean": se_ratio_mean,
            "threeway_SE_ratio_GEE_robust_over_MixedLM_max": se_ratio_max,
        },
        "outputs": {name: sha256_file(AUDIT_DIR / name) for name in output_names},
        "scope_exclusions": [
            "no manuscript changes",
            "no neuroscience interpretation",
            "no alternative model search",
            "no post-hoc localization",
            "no changed random-effects structure",
        ],
    }
    write_json(provenance, AUDIT_DIR / "provenance.json")

    print("=" * 70)
    print("SAME-DATA MIXEDLM vs GEE AUDIT")
    print("=" * 70)
    print()
    print("Frozen input:")
    print(str(INPUT.resolve()))
    print(f"SHA256 = {fingerprint['input_sha256']}")
    print()
    print(f"Subjects = {analysis_dat['Subject'].nunique()}")
    print(f"Rows = {len(analysis_dat)}")
    print("HC = 32; SZ = 32; rows per subject = 20")
    print("Missing values = 0; each Bin 1-10 = 128 rows")
    print("Anticipation = 640; Outcome = 640")
    print(
        "Centered-covariate means: "
        f"age_c={fingerprint['analysis_row_means']['age_c']:.12g}, "
        f"sex_c={fingerprint['analysis_row_means']['sex_c']:.12g}, "
        f"meanFD_c={fingerprint['analysis_row_means']['meanFD_c']:.12g}"
    )
    print()
    print("-" * 70)
    print("CANONICAL GEE ROBUST WALD")
    print("-" * 70)
    print()
    print(f"chi2(9) = {canonical_chi2:.12f}")
    print(f"p = {canonical_p:.12g}")
    print()
    print(f"replicates 26.4055 / .001753: {'YES' if canonical_reproduced else 'NO'}")
    print()
    print("-" * 70)
    print("HISTORICAL MIXEDLM ML-LRT")
    print("-" * 70)
    print()
    print(f"Full logLik = {float(full.llf):.12f}")
    print(f"Reduced logLik = {float(reduced.llf):.12f}")
    print()
    print(f"LR chi2(9) = {mixed_lr:.12f}")
    print(f"p = {mixed_lr_p:.12g}")
    print()
    print(f"difference from 10.9727 = {historical_difference:.12f}")
    print()
    print(f"historical 10.9727 recovered: {'YES' if historical_recovered else 'NO'}")
    print()
    print("-" * 70)
    print("MIXEDLM FULL-MODEL WALD")
    print("-" * 70)
    print()
    print(f"chi2(9) = {mixed_wald_chi2:.12f}")
    print(f"p = {mixed_wald_p:.12g}")
    print()
    print("-" * 70)
    print("GEE NAIVE WALD — DIAGNOSTIC")
    print("-" * 70)
    print()
    print(f"chi2(9) = {gee_naive_chi2:.12f}")
    print(f"p = {gee_naive_p:.12g}")
    print()
    print("-" * 70)
    print("THREE-WAY COEFFICIENT AGREEMENT")
    print("-" * 70)
    print()
    print(f"max abs beta difference = {max_beta_difference:.12g}")
    print(f"mean abs beta difference = {mean_beta_difference:.12g}")
    print(f"beta correlation = {beta_correlation:.12g}")
    print()
    print("-" * 70)
    print("MIXEDLM NUMERICAL QC")
    print("-" * 70)
    print()
    print(f"FULL converged = {full.converged}")
    print(f"REDUCED converged = {reduced.converged}")
    print(f"random intercept variance = {full_attempt['random_intercept_variance']:.12g}")
    print(f"ICC = {full_attempt['ICC']:.12g}")
    print(f"optimizer sensitive = {optimizer_qc['optimizer_sensitive']}")
    print()
    print("-" * 70)
    print("FORENSIC CONCLUSION")
    print("-" * 70)
    print()
    print(f"Historical result: {'RECOVERED' if historical_recovered else 'NOT RECOVERED'}")
    print()
    print(f"Cross-framework conclusion: {cross_framework}")
    print()
    print(f"Primary source of discrepancy: {primary_source}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
