#!/usr/bin/env python3
"""Run CURRENT64 canonical Steps 6--10 (normative modules excluded)."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT_VALUE = os.environ.get("STRIATAL_CURRENT64_ROOT")
if not ROOT_VALUE:
    raise RuntimeError("Set STRIATAL_CURRENT64_ROOT to the private canonical-rerun root")
ROOT = Path(ROOT_VALUE).resolve()
VENDOR = Path(os.environ.get("STRIATAL_PYTHON_VENDOR", ROOT / ".python_packages")).resolve()
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
import patsy
import scipy
from scipy.stats import chi2, norm, pearsonr, spearmanr, t as student_t
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests


STAGE_INPUT = ROOT / "06_REWARD_STAGE" / "subject_bin_stage_FC_long.tsv"
BEHAVIOR_INPUT = ROOT / "01_FROZEN_INPUTS" / "behavior" / "current64_behavior_final_scores.tsv"
ROI_MAPPING = ROOT / "01_FROZEN_INPUTS" / "reward_stage_mapping" / "reward_stage_mapping_65.tsv"
NOT_ASSIGNED = ROOT / "06_REWARD_STAGE" / "not_assigned_ROIs.tsv"
ARCHIVE = Path(os.environ.get("STRIATAL_HISTORICAL_ARCHIVE", ROOT / "historical_archive")).resolve()

BEHAVIOR_PAIRS = [
    ("BB01", "TEPS_anticipatory", "Anticipation_slope"),
    ("BB02", "TEPS_consummatory", "Outcome_slope"),
    ("BB03", "ACIPS_anticipatory", "Anticipation_slope"),
    ("BB04", "ACIPS_consummatory", "Outcome_slope"),
]

RESUME = "--resume" in sys.argv
if RESUME:
    sys.argv.remove("--resume")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    df.to_csv(buffer, sep="\t", index=False, na_rep="NA", float_format="%.17g")
    content = buffer.getvalue()
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if RESUME and existing.replace("\r\n", "\n") == content.replace("\r\n", "\n"):
            logging.info("Resume verified existing output content (newline-normalized): %s", path)
            return
        raise RuntimeError(f"Default no-overwrite policy: {path} exists or differs")
    path.write_text(content, encoding="utf-8", newline="")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if RESUME and path.read_text(encoding="utf-8") == text:
            logging.info("Resume verified existing text byte-for-byte: %s", path)
            return
        raise RuntimeError(f"Default no-overwrite policy: {path} exists or differs")
    path.write_text(text, encoding="utf-8", newline="\n")


def write_dynamic_text(text: str, path: Path, ignored_prefixes: tuple[str, ...]) -> None:
    """Verify stable resume content while allowing run timestamps to differ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if RESUME:
            stable_existing = [line for line in existing.splitlines() if not line.startswith(ignored_prefixes)]
            stable_new = [line for line in text.splitlines() if not line.startswith(ignored_prefixes)]
            if stable_existing == stable_new:
                logging.info("Resume verified existing dynamic text after excluding %s: %s", ignored_prefixes, path)
                return
        raise RuntimeError(f"Default no-overwrite policy: {path} exists or differs")
    path.write_text(text, encoding="utf-8", newline="\n")


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=1)
    require(bool(np.isfinite(sd)) and sd > 0, f"Cannot standardize {series.name}")
    return (x - x.mean()) / sd


def descriptive_ci(data: pd.DataFrame, group_cols: list[str], value: str) -> pd.DataFrame:
    out = (
        data.groupby(group_cols, observed=True, as_index=False)
        .agg(N=(value, "count"), Mean_FC=(value, "mean"), SD=(value, "std"))
    )
    out["SE"] = out["SD"] / np.sqrt(out["N"])
    crit = student_t.ppf(0.975, out["N"] - 1)
    out["CI95_low"] = out["Mean_FC"] - crit * out["SE"]
    out["CI95_high"] = out["Mean_FC"] + crit * out["SE"]
    return out


def normalize_stage_input() -> pd.DataFrame:
    require(STAGE_INPUT.is_file(), f"Missing canonical stage input: {STAGE_INPUT}")
    raw = pd.read_csv(STAGE_INPUT, sep="\t")
    required = {"Subject", "Group", "Bin", "G1_bin_center", "Stage", "FC", "Age", "Sex", "MeanFD"}
    require(required.issubset(raw.columns), f"Missing stage columns: {sorted(required - set(raw.columns))}")
    require(len(raw) == 1280, "Canonical stage table must contain 1,280 rows")
    require(raw["Subject"].nunique() == 64, "Canonical stage table must contain 64 subjects")
    require(not raw.duplicated(["Subject", "Bin", "Stage"]).any(), "Duplicate stage-table keys")
    require(np.isfinite(raw["FC"]).all(), "Non-finite stage FC")
    require(sorted(raw["G1_bin_center"].unique().tolist()) == [x - 5.5 for x in range(1, 11)], "Wrong G1 predictor")
    df = raw.rename(
        columns={
            "Subject": "subject",
            "Group": "group",
            "Bin": "bin",
            "G1_bin_center": "G1_centered",
            "Stage": "stage",
            "Age": "age",
            "Sex": "sex",
            "MeanFD": "meanFD",
        }
    ).copy()
    df["group"] = pd.Categorical(df["group"], categories=["HC", "SZ"])
    df["stage"] = pd.Categorical(df["stage"], categories=["Outcome", "Anticipation"])
    df["sex"] = pd.Categorical(df["sex"], categories=["Female", "Male"])
    people = df.drop_duplicates("subject")
    require(people["group"].value_counts().to_dict() == {"HC": 32, "SZ": 32}, "Group count mismatch")
    df["age_c"] = df["age"] - people["age"].mean()
    df["meanFD_c"] = df["meanFD"] - people["meanFD"].mean()
    sex_num = df["sex"].map({"Female": 0.0, "Male": 1.0}).astype(float)
    male_prop = float(people["sex"].map({"Female": 0.0, "Male": 1.0}).astype(float).mean())
    df["sex_c"] = sex_num - male_prop
    return df


def fit_mixed(formula: str, data: pd.DataFrame, label: str):
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["subject"],
        re_formula="1 + G1_centered",
    )
    attempts: list[dict[str, object]] = []
    for method in ["lbfgs", "powell", "cg"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = model.fit(reml=False, method=method, maxiter=5000, disp=False)
                warning_text = " | ".join(str(w.message) for w in caught)
                attempts.append(
                    {
                        "model": label,
                        "optimizer": method,
                        "converged": bool(result.converged),
                        "logLik": float(result.llf),
                        "warnings": warning_text,
                        "error": "",
                    }
                )
                if result.converged:
                    return result, method, warning_text, pd.DataFrame(attempts)
            except Exception as exc:  # recorded, then historical optimizer fallback
                attempts.append(
                    {
                        "model": label,
                        "optimizer": method,
                        "converged": False,
                        "logLik": np.nan,
                        "warnings": " | ".join(str(w.message) for w in caught),
                        "error": repr(exc),
                    }
                )
    raise RuntimeError(f"Mixed model {label} did not converge: {attempts}")


def coefficient_table(result) -> pd.DataFrame:
    fe = result.fe_params.copy()
    names = list(fe.index)
    cov = result.cov_params().loc[names, names]
    rows = []
    for term in names:
        estimate = float(fe[term])
        se = float(np.sqrt(cov.loc[term, term]))
        statistic = estimate / se
        rows.append(
            {
                "term": term,
                "estimate": estimate,
                "SE": se,
                "df_if_available": np.nan,
                "statistic": statistic,
                "p": float(2 * norm.sf(abs(statistic))),
                "CI95_low": estimate - 1.96 * se,
                "CI95_high": estimate + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


def find_term(names: list[str], required_tokens: list[str], forbidden_tokens: list[str] | None = None) -> str:
    forbidden_tokens = forbidden_tokens or []
    matches = [
        term
        for term in names
        if all(token in term for token in required_tokens)
        and not any(token in term for token in forbidden_tokens)
    ]
    require(len(matches) == 1, f"Term identification failed: required={required_tokens}, matches={matches}")
    return matches[0]


def simple_effects(result) -> pd.DataFrame:
    fe = result.fe_params.copy()
    names = list(fe.index)
    cov = result.cov_params().loc[names, names]
    term_g1 = "G1_centered"
    term_stage = find_term(names, ["stage[T.Anticipation]", "G1_centered"], ["group[T.SZ]", "age_c", "sex_c", "meanFD_c"])
    term_group = find_term(names, ["group[T.SZ]", "G1_centered"], ["stage[T.Anticipation]"])
    term_three = find_term(names, ["group[T.SZ]", "stage[T.Anticipation]", "G1_centered"])

    def one(name: str, weights: dict[str, float]) -> dict[str, object]:
        w = np.zeros(len(names))
        for term, weight in weights.items():
            w[names.index(term)] = weight
        estimate = float(w @ fe.to_numpy())
        se = float(np.sqrt(max(float(w @ cov.to_numpy() @ w), 0.0)))
        statistic = estimate / se
        return {
            "contrast": name,
            "estimate": estimate,
            "SE": se,
            "CI95_low": estimate - 1.96 * se,
            "CI95_high": estimate + 1.96 * se,
            "statistic": statistic,
            "p": float(2 * norm.sf(abs(statistic))),
        }

    return pd.DataFrame(
        [
            one("HC Outcome slope", {term_g1: 1}),
            one("HC Anticipation slope", {term_g1: 1, term_stage: 1}),
            one("SZ Outcome slope", {term_g1: 1, term_group: 1}),
            one("SZ Anticipation slope", {term_g1: 1, term_group: 1, term_stage: 1, term_three: 1}),
            one("HC Anticipation minus Outcome slope difference", {term_stage: 1}),
            one("SZ Anticipation minus Outcome slope difference", {term_stage: 1, term_three: 1}),
            one("SZ minus HC difference in stage-gradient differentiation", {term_three: 1}),
        ]
    )


def fit_info(result, optimizer: str, warning_text: str, formula: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "N_subjects": int(pd.Series(result.model.groups).nunique()),
                "N_observations": int(result.nobs),
                "logLik": float(result.llf),
                "AIC": float(result.aic),
                "BIC": float(result.bic),
                "converged": bool(result.converged),
                "optimizer": optimizer,
                "random_effect_structure": "subject random intercept + random G1_centered slope",
                "reml": False,
                "formula": " ".join(formula.split()),
                "warnings": warning_text,
            }
        ]
    )


def mixed_diagnostic_row(label: str, result, optimizer: str, warning_text: str) -> dict[str, object]:
    resid = np.asarray(result.resid, dtype=float)
    cov_re = np.asarray(result.cov_re, dtype=float)
    return {
        "model": label,
        "converged": bool(result.converged),
        "optimizer": optimizer,
        "N_observations": int(result.nobs),
        "N_subjects": int(pd.Series(result.model.groups).nunique()),
        "design_rank": int(np.linalg.matrix_rank(result.model.exog)),
        "design_columns": int(result.model.exog.shape[1]),
        "residual_mean": float(np.mean(resid)),
        "residual_SD": float(np.std(resid, ddof=1)),
        "residual_min": float(np.min(resid)),
        "residual_max": float(np.max(resid)),
        "random_intercept_variance": float(cov_re[0, 0]),
        "random_slope_variance": float(cov_re[1, 1]),
        "random_intercept_slope_covariance": float(cov_re[0, 1]),
        "warnings": warning_text,
        "influence_summary": "MixedLM observation influence not used; residual and random-effect diagnostics reported.",
    }


def ols_diagnostics(label: str, result, n_subjects: int) -> dict[str, object]:
    resid = np.asarray(result.resid, dtype=float)
    influence = result.get_influence()
    cooks = np.asarray(influence.cooks_distance[0], dtype=float)
    leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
    return {
        "model": label,
        "N_observations": int(result.nobs),
        "N_subjects_or_clusters": n_subjects,
        "design_rank": int(np.linalg.matrix_rank(result.model.exog)),
        "design_columns": int(result.model.exog.shape[1]),
        "residual_mean": float(np.mean(resid)),
        "residual_SD": float(np.std(resid, ddof=1)),
        "residual_min": float(np.min(resid)),
        "residual_max": float(np.max(resid)),
        "max_Cooks_D": float(np.max(cooks)),
        "n_Cooks_D_gt_4_over_N": int(np.sum(cooks > 4 / len(cooks))),
        "max_leverage": float(np.max(leverage)),
        "warnings": "",
    }


def run_mixed_models(df: pd.DataFrame):
    formulas = {
        "A": (
            "FC ~ group * stage * G1_centered",
            "FC ~ group * stage + group * G1_centered + stage * G1_centered",
        ),
        "B": (
            "FC ~ group * stage * G1_centered + age_c + sex + meanFD_c",
            "FC ~ group * stage + group * G1_centered + stage * G1_centered + age_c + sex + meanFD_c",
        ),
        "C": (
            "FC ~ group * stage * G1_centered + age_c * stage * G1_centered + sex_c * stage * G1_centered + meanFD_c * stage * G1_centered",
            "FC ~ group * stage + group * G1_centered + stage * G1_centered + age_c * stage * G1_centered + sex_c * stage * G1_centered + meanFD_c * stage * G1_centered",
        ),
    }
    results: dict[str, object] = {}
    lrt_rows = []
    diagnostic_rows = []
    attempt_tables = []
    for label, (full_formula, reduced_formula) in formulas.items():
        logging.info("Fitting Model %s full", label)
        full, optimizer, warning_text, attempts = fit_mixed(full_formula, df, f"Model_{label}_full")
        logging.info("Fitting Model %s reduced", label)
        reduced, reduced_optimizer, reduced_warning, reduced_attempts = fit_mixed(reduced_formula, df, f"Model_{label}_reduced")
        attempt_tables.extend([attempts, reduced_attempts])
        lr = float(2 * (full.llf - reduced.llf))
        lrt_rows.append(
            {
                "comparison": f"Model {label}: Group x Stage x G1",
                "full_model": f"Model_{label}_full",
                "reduced_model": f"Model_{label}_reduced",
                "df_difference": 1,
                "LR_chi2": lr,
                "p": float(chi2.sf(lr, 1)),
                "nested": True,
            }
        )
        coeff = coefficient_table(full)
        simple = simple_effects(full)
        fit = fit_info(full, optimizer, warning_text, full_formula)
        model_dir = ROOT / "07_STAGE_GRADIENT_MODELS"
        if label == "A":
            coefficient_path = model_dir / "model_A_base_coefficients.tsv"
            simple_path = model_dir / "model_A_simple_effects.tsv"
            fit_path = model_dir / "model_A_fit.tsv"
        elif label == "B":
            coefficient_path = model_dir / "model_B_adjusted_coefficients.tsv"
            simple_path = model_dir / "model_B_adjusted_simple_effects.tsv"
            fit_path = model_dir / "model_B_fit.tsv"
        else:
            coefficient_path = model_dir / "model_C_gradient_sensitive_coefficients.tsv"
            simple_path = model_dir / "model_C_gradient_sensitive_simple_effects.tsv"
            fit_path = model_dir / "model_C_fit.tsv"
        write_tsv(coeff, coefficient_path)
        write_tsv(simple, simple_path)
        write_tsv(fit, fit_path)
        diagnostic_rows.append(mixed_diagnostic_row(f"Model_{label}_full", full, optimizer, warning_text))
        diagnostic_rows.append(mixed_diagnostic_row(f"Model_{label}_reduced", reduced, reduced_optimizer, reduced_warning))
        results[label] = {"full": full, "simple": simple, "coeff": coeff, "fit": fit}

    moderation = results["C"]["coeff"].copy()
    moderation = moderation[
        moderation["term"].str.contains("age_c|sex_c|meanFD_c", regex=True)
        & moderation["term"].str.contains("G1_centered", regex=False)
    ].copy()
    moderation.insert(0, "moderator", moderation["term"].str.extract(r"^(age_c|sex_c|meanFD_c)", expand=False))
    moderation.insert(1, "moderation_type", np.where(moderation["term"].str.contains("stage"), "Stage_x_G1", "G1"))
    write_tsv(moderation, ROOT / "07_STAGE_GRADIENT_MODELS" / "model_C_covariate_moderation.tsv")
    write_tsv(pd.DataFrame(lrt_rows), ROOT / "07_STAGE_GRADIENT_MODELS" / "likelihood_ratio_tests.tsv")
    write_tsv(pd.DataFrame(diagnostic_rows), ROOT / "07_STAGE_GRADIENT_MODELS" / "diagnostics" / "mixed_model_diagnostics.tsv")
    write_tsv(pd.concat(attempt_tables, ignore_index=True), ROOT / "07_STAGE_GRADIENT_MODELS" / "diagnostics" / "optimizer_attempts.tsv")
    return results, pd.DataFrame(lrt_rows), moderation


def build_individual_slopes(df: pd.DataFrame):
    long_rows = []
    for (subject, group, stage_name), d in df.groupby(["subject", "group", "stage"], observed=True, sort=False):
        d = d.sort_values("G1_centered")
        x = d["G1_centered"].to_numpy(dtype=float)
        y = d["FC"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        first = d.iloc[0]
        long_rows.append(
            {
                "Subject": subject,
                "Group": str(group),
                "Stage": str(stage_name),
                "Slope": float(slope),
                "Intercept": float(intercept),
                "N_bins": len(d),
                "R2": r2,
                "Age": first["age"],
                "Sex": str(first["sex"]),
                "MeanFD": first["meanFD"],
            }
        )
    long = pd.DataFrame(long_rows)
    require(len(long) == 128, "Expected 128 subject-stage slopes")
    wide_values = long.pivot(index=["Subject", "Group", "Age", "Sex", "MeanFD"], columns="Stage", values="Slope").reset_index()
    wide_values.columns.name = None
    wide = wide_values.rename(columns={"Anticipation": "Anticipation_slope", "Outcome": "Outcome_slope"})
    wide["Delta_slope"] = wide["Anticipation_slope"] - wide["Outcome_slope"]
    wide = wide[["Subject", "Group", "Anticipation_slope", "Outcome_slope", "Delta_slope", "Age", "Sex", "MeanFD"]]
    require(len(wide) == 64 and wide[["Anticipation_slope", "Outcome_slope", "Delta_slope"]].notna().all().all(), "Slope wide QC failed")

    long["z_within_stage"] = long.groupby("Stage", observed=True)["Slope"].transform(lambda x: (x - x.mean()) / x.std(ddof=1))
    qc = long[["Subject", "Group", "Stage", "Slope", "N_bins", "R2", "z_within_stage"]].copy()
    qc["is_na"] = qc["Slope"].isna()
    qc["is_inf"] = np.isinf(qc["Slope"])
    qc["is_extreme_abs_z_gt_3"] = qc["z_within_stage"].abs() > 3
    qc["pass_structural_qc"] = (~qc["is_na"]) & (~qc["is_inf"]) & qc["N_bins"].eq(10)
    write_tsv(long.drop(columns="z_within_stage"), ROOT / "09_INDIVIDUAL_SLOPES" / "subject_stage_slopes_long.tsv")
    write_tsv(wide, ROOT / "09_INDIVIDUAL_SLOPES" / "subject_stage_slopes_wide.tsv")
    write_tsv(qc, ROOT / "09_INDIVIDUAL_SLOPES" / "slope_QC.tsv")
    return long, wide, qc


def delta_slope_models(wide: pd.DataFrame):
    d = wide.rename(columns={"Subject": "subject", "Group": "group", "Age": "age", "Sex": "sex", "MeanFD": "meanFD"}).copy()
    d["group"] = pd.Categorical(d["group"], categories=["HC", "SZ"])
    d["sex"] = pd.Categorical(d["sex"], categories=["Female", "Male"])
    d["age_c"] = d["age"] - d["age"].mean()
    d["meanFD_c"] = d["meanFD"] - d["meanFD"].mean()
    rows = []
    diagnostics = []
    for label, formula in [
        ("Unadjusted_HC3", "Delta_slope ~ group"),
        ("Adjusted_Age_Sex_MeanFD_HC3", "Delta_slope ~ group + age_c + sex + meanFD_c"),
    ]:
        model = smf.ols(formula, d).fit(cov_type="HC3")
        term = "group[T.SZ]"
        ci = model.conf_int().loc[term]
        hc = d.loc[d["group"] == "HC", "Delta_slope"]
        sz = d.loc[d["group"] == "SZ", "Delta_slope"]
        rows.append(
            {
                "Model": label,
                "N_HC": len(hc),
                "N_SZ": len(sz),
                "HC_mean": hc.mean(),
                "HC_SD": hc.std(ddof=1),
                "SZ_mean": sz.mean(),
                "SZ_SD": sz.std(ddof=1),
                "SZ_minus_HC_beta": model.params[term],
                "SE": model.bse[term],
                "CI95_low": ci.iloc[0],
                "CI95_high": ci.iloc[1],
                "statistic": model.tvalues[term],
                "p": model.pvalues[term],
                "covariance": "HC3",
            }
        )
        diagnostics.append(ols_diagnostics(f"Delta_slope_{label}", model, 64))
    result = pd.DataFrame(rows)
    write_tsv(result, ROOT / "09_INDIVIDUAL_SLOPES" / "delta_slope_group_models.tsv")
    write_tsv(pd.DataFrame(diagnostics), ROOT / "09_INDIVIDUAL_SLOPES" / "delta_slope_diagnostics.tsv")
    return result


def run_categorical(df: pd.DataFrame):
    formula = "FC ~ group * stage * C(bin)"
    gee_warning = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gee_model = smf.gee(formula, groups="subject", data=df, cov_struct=Exchangeable(), family=sm.families.Gaussian())
        gee = gee_model.fit(maxiter=200)
        gee_warning = " | ".join(str(w.message) for w in caught)
    gee_terms = list(gee.params.index)
    target_terms = [t for t in gee_terms if "group[T.SZ]" in t and "stage[T.Anticipation]" in t and "C(bin)" in t]
    require(len(target_terms) == 9, f"Expected 9 GEE three-way terms, got {target_terms}")
    R = np.zeros((9, len(gee_terms)))
    for i, term in enumerate(target_terms):
        R[i, gee_terms.index(term)] = 1
    gee_wald = gee.wald_test(R, scalar=True)
    gee_ci = gee.conf_int()
    gee_coeff = pd.DataFrame(
        {
            "term": gee_terms,
            "estimate": gee.params,
            "SE": gee.bse,
            "statistic": gee.params / gee.bse,
            "p": gee.pvalues,
            "CI95_low": gee_ci.iloc[:, 0],
            "CI95_high": gee_ci.iloc[:, 1],
        }
    ).reset_index(drop=True)
    gee_omnibus = pd.DataFrame(
        [
            {
                "test": "Group x Stage x categorical Bin",
                "chi2": float(np.asarray(gee_wald.statistic).squeeze()),
                "df": 9,
                "p": float(np.asarray(gee_wald.pvalue).squeeze()),
                "tested_terms": " | ".join(target_terms),
            }
        ]
    )
    dep = np.asarray(gee.cov_struct.dep_params).squeeze()
    gee_fit = pd.DataFrame(
        [
            {
                "N_subjects": df["subject"].nunique(),
                "N_observations": len(df),
                "working_correlation": "Exchangeable",
                "cluster": "Subject",
                "family": "Gaussian",
                "link": "identity",
                "estimated_exchangeable_alpha": float(dep),
                "converged": bool(gee.converged),
                "iterations": len(gee.fit_history.get("params", [])),
                "design_rank": int(np.linalg.matrix_rank(gee.model.exog)),
                "design_columns": int(gee.model.exog.shape[1]),
                "covariance": "GEE robust sandwich",
                "adjustment": "Unadjusted primary; exact historical generating script not recovered",
                "warnings": gee_warning,
            }
        ]
    )

    ols_warning = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ols = smf.ols(formula, data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["subject"], "use_correction": True, "df_correction": True},
            use_t=True,
        )
        ols_warning = " | ".join(str(w.message) for w in caught)
    ols_terms = list(ols.params.index)
    ols_target = [t for t in ols_terms if "group[T.SZ]" in t and "stage[T.Anticipation]" in t and "C(bin)" in t]
    require(len(ols_target) == 9, "Expected 9 clustered-OLS three-way terms")
    R_ols = np.zeros((9, len(ols_terms)))
    for i, term in enumerate(ols_target):
        R_ols[i, ols_terms.index(term)] = 1
    ols_wald = ols.wald_test(R_ols, use_f=True, scalar=True)
    ols_ci = ols.conf_int()
    ols_coeff = pd.DataFrame(
        {
            "term": ols_terms,
            "estimate": ols.params,
            "SE": ols.bse,
            "statistic": ols.tvalues,
            "p": ols.pvalues,
            "CI95_low": ols_ci.iloc[:, 0],
            "CI95_high": ols_ci.iloc[:, 1],
        }
    ).reset_index(drop=True)
    ols_omnibus = pd.DataFrame(
        [
            {
                "test": "Group x Stage x categorical Bin",
                "statistic_type": "cluster-robust F",
                "statistic": float(np.asarray(ols_wald.statistic).squeeze()),
                "df_num": float(ols_wald.df_num),
                "df_denom": float(ols_wald.df_denom),
                "p": float(np.asarray(ols_wald.pvalue).squeeze()),
                "tested_terms": " | ".join(ols_target),
                "cluster_correction": True,
            }
        ]
    )

    grid = pd.DataFrame(
        [(group, stage, bin_value) for group in ["HC", "SZ"] for stage in ["Outcome", "Anticipation"] for bin_value in range(1, 11)],
        columns=["group", "stage", "bin"],
    )
    grid["group"] = pd.Categorical(grid["group"], categories=["HC", "SZ"])
    grid["stage"] = pd.Categorical(grid["stage"], categories=["Outcome", "Anticipation"])
    # statsmodels 0.15 no longer exposes Patsy's design_info on PandasData.
    # Rebuild the prespecified RHS and align columns explicitly to fitted terms.
    design_frame = patsy.dmatrix("group * stage * C(bin)", grid, return_type="dataframe")
    design_frame = design_frame.reindex(columns=gee.params.index, fill_value=0.0)
    design = design_frame.to_numpy(dtype=float)
    estimate = design @ gee.params.to_numpy()
    cov = gee.cov_params().to_numpy()
    se = np.sqrt(np.einsum("ij,jk,ik->i", design, cov, design))
    estimated = pd.DataFrame(
        {
            "Group": grid["group"].astype(str),
            "Stage": grid["stage"].astype(str),
            "Bin": grid["bin"],
            "Estimate": estimate,
            "SE": se,
            "CI95_low": estimate - 1.96 * se,
            "CI95_high": estimate + 1.96 * se,
            "Model": "GEE_exchangeable",
            "Adjustment": "Unadjusted",
        }
    )

    delta = (
        df.pivot_table(index=["subject", "group", "bin", "age", "sex", "meanFD"], columns="stage", values="FC", observed=True)
        .reset_index()
    )
    delta.columns.name = None
    delta["Delta_FC"] = delta["Anticipation"] - delta["Outcome"]
    delta = delta.rename(columns={"subject": "Subject", "group": "Group", "bin": "Bin", "age": "Age", "sex": "Sex", "meanFD": "MeanFD"})
    delta["Group"] = delta["Group"].astype(str)
    delta["Sex"] = delta["Sex"].astype(str)
    delta_desc = descriptive_ci(delta, ["Group", "Bin"], "Delta_FC").rename(columns={"Mean_FC": "Estimate"})

    write_tsv(gee_coeff, ROOT / "08_CATEGORICAL_SHAPE" / "GEE_coefficients.tsv")
    write_tsv(gee_omnibus, ROOT / "08_CATEGORICAL_SHAPE" / "GEE_omnibus_tests.tsv")
    write_tsv(gee_fit, ROOT / "08_CATEGORICAL_SHAPE" / "GEE_fit_information.tsv")
    write_tsv(ols_coeff, ROOT / "08_CATEGORICAL_SHAPE" / "clustered_OLS_coefficients.tsv")
    write_tsv(ols_omnibus, ROOT / "08_CATEGORICAL_SHAPE" / "clustered_OLS_omnibus_tests.tsv")
    write_tsv(estimated, ROOT / "08_CATEGORICAL_SHAPE" / "estimated_profiles.tsv")
    write_tsv(delta, ROOT / "08_CATEGORICAL_SHAPE" / "subject_bin_deltaFC.tsv")
    write_tsv(delta_desc, ROOT / "08_CATEGORICAL_SHAPE" / "group_deltaFC_profiles.tsv")
    cat_diag = pd.DataFrame(
        [
            {
                "model": "GEE_exchangeable",
                "N_observations": len(df),
                "N_subjects_or_clusters": df["subject"].nunique(),
                "design_rank": int(np.linalg.matrix_rank(gee.model.exog)),
                "design_columns": int(gee.model.exog.shape[1]),
                "converged": bool(gee.converged),
                "working_alpha": float(dep),
                "warnings": gee_warning,
            },
            {
                "model": "OLS_subject_clustered",
                "N_observations": len(df),
                "N_subjects_or_clusters": df["subject"].nunique(),
                "design_rank": int(np.linalg.matrix_rank(ols.model.exog)),
                "design_columns": int(ols.model.exog.shape[1]),
                "converged": "not_applicable_closed_form",
                "working_alpha": np.nan,
                "warnings": ols_warning,
            },
        ]
    )
    write_tsv(cat_diag, ROOT / "08_CATEGORICAL_SHAPE" / "diagnostics" / "categorical_model_diagnostics.tsv")
    return gee_omnibus, ols_omnibus, estimated, delta, delta_desc


def run_brain_behavior(wide: pd.DataFrame):
    behavior = pd.read_csv(BEHAVIOR_INPUT, sep="\t")
    data = wide.merge(behavior, on=["Subject", "Group"], how="left", validate="one_to_one")
    require(len(data) == 64, "Brain-behavior merge is not 64 rows")
    result_rows = []
    diag_rows = []
    fig_rows = []
    available_rows = []
    fitted_models = {}
    for analysis_id, behavior_name, neural_name in BEHAVIOR_PAIRS:
        cols = ["Subject", "Group", behavior_name, neural_name, "Age", "Sex", "MeanFD"]
        d = data[cols].dropna().copy().rename(columns={behavior_name: "Behavior", neural_name: "Neural", "Group": "group", "Age": "age", "Sex": "sex", "MeanFD": "meanFD"})
        d["group"] = pd.Categorical(d["group"], categories=["HC", "SZ"])
        d["sex"] = pd.Categorical(d["sex"], categories=["Female", "Male"])
        d["age_c"] = d["age"] - d["age"].mean()
        d["meanFD_c"] = d["meanFD"] - d["meanFD"].mean()
        d["Behavior_z"] = zscore(d["Behavior"])
        d["Neural_z"] = zscore(d["Neural"])
        d["age_z"] = zscore(d["age"])
        d["meanFD_z"] = zscore(d["meanFD"])
        r, r_p = pearsonr(d["Behavior"], d["Neural"])
        rho, rho_p = spearmanr(d["Behavior"], d["Neural"])
        unstd = smf.ols("Neural ~ Behavior + group + age_c + sex + meanFD_c", data=d).fit(cov_type="HC3")
        std = smf.ols("Neural_z ~ Behavior_z + group + age_z + sex + meanFD_z", data=d).fit(cov_type="HC3")
        interaction = smf.ols("Neural_z ~ Behavior_z * group + age_z + sex + meanFD_z", data=d).fit(cov_type="HC3")
        interaction_term = find_term(list(interaction.params.index), ["Behavior_z", "group[T.SZ]"])
        ci_unstd = unstd.conf_int().loc["Behavior"]
        ci_std = std.conf_int().loc["Behavior_z"]
        ci_int = interaction.conf_int().loc[interaction_term]
        result_rows.append(
            {
                "Analysis_ID": analysis_id,
                "Behavior": behavior_name,
                "Neural_metric": neural_name,
                "N_total": len(d),
                "N_HC": int((d["group"] == "HC").sum()),
                "N_SZ": int((d["group"] == "SZ").sum()),
                "Model": "HC3 adjusted: Group + Age + Sex + MeanFD",
                "Beta": unstd.params["Behavior"],
                "Unstandardized_CI95_low": ci_unstd.iloc[0],
                "Unstandardized_CI95_high": ci_unstd.iloc[1],
                "Standardized_beta": std.params["Behavior_z"],
                "SE": std.bse["Behavior_z"],
                "CI95_low": ci_std.iloc[0],
                "CI95_high": ci_std.iloc[1],
                "p_raw": std.pvalues["Behavior_z"],
                "q_FDR": np.nan,
                "Pearson_r": r,
                "Pearson_p": r_p,
                "Spearman_rho": rho,
                "Spearman_p": rho_p,
                "Group_behavior_interaction_beta": interaction.params[interaction_term],
                "Group_behavior_interaction_SE": interaction.bse[interaction_term],
                "Group_behavior_interaction_CI95_low": ci_int.iloc[0],
                "Group_behavior_interaction_CI95_high": ci_int.iloc[1],
                "Group_behavior_interaction_p": interaction.pvalues[interaction_term],
                "FDR_family": "BH across four prespecified adjusted Behavior terms",
            }
        )
        diag_rows.append(ols_diagnostics(f"{analysis_id}_adjusted_main", std, len(d)))
        diag_rows.append(ols_diagnostics(f"{analysis_id}_group_interaction", interaction, len(d)))
        for _, row in d.iterrows():
            fig_rows.append(
                {
                    "Row_type": "Observed",
                    "Analysis_ID": analysis_id,
                    "Behavior": behavior_name,
                    "Neural_metric": neural_name,
                    "Subject": row["Subject"],
                    "Group": str(row["group"]),
                    "Behavior_z": row["Behavior_z"],
                    "Neural_z": row["Neural_z"],
                    "Estimate": np.nan,
                    "CI95_low": np.nan,
                    "CI95_high": np.nan,
                    "N": len(d),
                }
            )

        sex_props = d["sex"].value_counts(normalize=True).to_dict()
        x_grid = np.linspace(float(d["Behavior_z"].min()), float(d["Behavior_z"].max()), 80)
        for group in ["HC", "SZ"]:
            for x in x_grid:
                weighted_design = None
                for sex in ["Female", "Male"]:
                    pred = pd.DataFrame({"Behavior_z": [x], "group": [group], "age_z": [0.0], "sex": [sex], "meanFD_z": [0.0]})
                    pred["group"] = pd.Categorical(pred["group"], categories=["HC", "SZ"])
                    pred["sex"] = pd.Categorical(pred["sex"], categories=["Female", "Male"])
                    design_frame = patsy.dmatrix(
                        "Behavior_z * group + age_z + sex + meanFD_z",
                        pred,
                        return_type="dataframe",
                    ).reindex(columns=interaction.params.index, fill_value=0.0)
                    design = design_frame.to_numpy(dtype=float)[0]
                    weight = float(sex_props.get(sex, 0.0))
                    weighted_design = design * weight if weighted_design is None else weighted_design + design * weight
                estimate = float(weighted_design @ interaction.params.to_numpy())
                se = float(np.sqrt(weighted_design @ interaction.cov_params().to_numpy() @ weighted_design))
                fig_rows.append(
                    {
                        "Row_type": "Prediction",
                        "Analysis_ID": analysis_id,
                        "Behavior": behavior_name,
                        "Neural_metric": neural_name,
                        "Subject": "",
                        "Group": group,
                        "Behavior_z": x,
                        "Neural_z": np.nan,
                        "Estimate": estimate,
                        "CI95_low": estimate - 1.96 * se,
                        "CI95_high": estimate + 1.96 * se,
                        "N": len(d),
                    }
                )
        available_rows.append(
            {
                "Analysis_ID": analysis_id,
                "Behavior": behavior_name,
                "Neural_metric": neural_name,
                "N_available": len(d),
                "N_missing_from_64": 64 - len(d),
                "N_HC": int((d["group"] == "HC").sum()),
                "N_SZ": int((d["group"] == "SZ").sum()),
            }
        )
        fitted_models[analysis_id] = interaction

    results = pd.DataFrame(result_rows)
    results["q_FDR"] = multipletests(results["p_raw"].to_numpy(), method="fdr_bh")[1]
    write_tsv(results, ROOT / "10_BRAIN_BEHAVIOR" / "brain_behavior_results.tsv")
    write_tsv(data, ROOT / "10_BRAIN_BEHAVIOR" / "brain_behavior_subject_level.tsv")
    write_tsv(pd.DataFrame(available_rows), ROOT / "10_BRAIN_BEHAVIOR" / "diagnostics" / "available_case_missingness.tsv")
    write_tsv(pd.DataFrame(diag_rows), ROOT / "10_BRAIN_BEHAVIOR" / "diagnostics" / "brain_behavior_model_diagnostics.tsv")
    return results, pd.DataFrame(fig_rows)


def historical_validation(models, delta_models, gee_omnibus, ols_omnibus, brain_results):
    def contrast(model_label: str, name: str) -> float:
        table = models[model_label]["simple"]
        return float(table.loc[table["contrast"] == name, "estimate"].iloc[0])

    rows = []
    def add(analysis, statistic, historical, canonical, interpretation, same_direction=None):
        difference = canonical - historical if np.isfinite(historical) else np.nan
        if same_direction is None:
            same_direction = bool(np.sign(historical) == np.sign(canonical)) if np.isfinite(historical) and historical != 0 and canonical != 0 else np.nan
        rows.append(
            {
                "Analysis": analysis,
                "Statistic": statistic,
                "Historical_value": historical,
                "Canonical_value": canonical,
                "Difference": difference,
                "Same_direction": same_direction,
                "Interpretation": interpretation,
            }
        )

    add("HC Stage x G1 Model A", "estimate", -0.0069598721377975015, contrast("A", "HC Anticipation minus Outcome slope difference"), "EXPECTED_SIGN_REVERSAL", False)
    add("SZ Stage x G1 Model A", "estimate", -0.004886598112923626, contrast("A", "SZ Anticipation minus Outcome slope difference"), "EXPECTED_SIGN_REVERSAL", False)
    add("Group x Stage x G1 Model A", "estimate", 0.002073274024873876, contrast("A", "SZ minus HC difference in stage-gradient differentiation"), "EXPECTED_SIGN_REVERSAL", False)
    add("Group x Stage x G1 Model B", "estimate", -0.002073274024873857, contrast("B", "SZ minus HC difference in stage-gradient differentiation"), "NUMERIC_TOLERANCE")
    add("Group x Stage x G1 Model C", "estimate", -0.001077694369200558, contrast("C", "SZ minus HC difference in stage-gradient differentiation"), "NUMERIC_TOLERANCE")
    adjusted_delta = delta_models.loc[delta_models["Model"] == "Adjusted_Age_Sex_MeanFD_HC3"].iloc[0]
    add("Individual Delta slope adjusted group", "p", 0.646, float(adjusted_delta["p"]), "NUMERIC_TOLERANCE", np.nan)
    add("Categorical GEE Group x Stage x Bin", "p", 0.2414, float(gee_omnibus["p"].iloc[0]), "MODEL_SPECIFICATION_DIFFERENCE", np.nan)
    add("Categorical clustered OLS Group x Stage x Bin", "p", 0.2776, float(ols_omnibus["p"].iloc[0]), "MODEL_SPECIFICATION_DIFFERENCE", np.nan)
    for _, row in brain_results.iterrows():
        add(f"{row['Analysis_ID']} brain-behavior", "standardized_beta", np.nan, float(row["Standardized_beta"]), "OTHER", np.nan)
        add(f"{row['Analysis_ID']} Group x Behavior", "p", np.nan, float(row["Group_behavior_interaction_p"]), "OTHER", np.nan)
    validation = pd.DataFrame(rows)
    write_tsv(validation, ROOT / "16_VALIDATION" / "stage_analysis_historical_vs_canonical.tsv")
    return validation


def write_figure_sources(desc, slope_long, slope_wide, delta_desc, brain_fig, models, delta_models, gee_omnibus, ols_omnibus, brain_results):
    source_dir = ROOT / "14_R_FIGURES" / "source_data"
    fig1 = desc.rename(columns={"Mean_FC": "Estimate"}).copy()
    fig1["Model"] = "Observed_descriptive"
    fig1["Adjustment"] = "Unadjusted"
    write_tsv(fig1, source_dir / "Fig_stage_gradient_profiles.tsv")

    fig2_long = slope_long[["Subject", "Group", "Stage", "Slope", "Age", "Sex", "MeanFD"]].copy()
    fig2_long["Row_type"] = "Stage_slope"
    fig2_long["Delta_slope"] = np.nan
    delta_part = slope_wide[["Subject", "Group", "Delta_slope", "Age", "Sex", "MeanFD"]].copy()
    delta_part["Stage"] = "Delta"
    delta_part["Slope"] = np.nan
    delta_part["Row_type"] = "Delta_slope"
    fig2 = pd.concat([fig2_long, delta_part], ignore_index=True, sort=False)
    write_tsv(fig2, source_dir / "Fig_individual_stage_slopes.tsv")

    fig3 = delta_desc.copy()
    fig3["Model"] = "Observed_descriptive_delta_FC"
    fig3["Adjustment"] = "Unadjusted"
    write_tsv(fig3, source_dir / "Fig_stage_difference_profile.tsv")
    write_tsv(brain_fig, source_dir / "Fig_brain_behavior.tsv")

    stat_rows = []
    def add(figure, panel, test, row, source_model, q=np.nan):
        stat_rows.append(
            {
                "Figure": figure,
                "Panel": panel,
                "Test": test,
                "Effect": row.get("estimate", row.get("SZ_minus_HC_beta", row.get("Standardized_beta", np.nan))),
                "CI95_low": row.get("CI95_low", np.nan),
                "CI95_high": row.get("CI95_high", np.nan),
                "p": row.get("p", row.get("p_raw", np.nan)),
                "q": q,
                "Source_model": source_model,
            }
        )
    for group, panel in [("HC", "A"), ("SZ", "B")]:
        r = models["A"]["simple"].loc[models["A"]["simple"]["contrast"] == f"{group} Anticipation minus Outcome slope difference"].iloc[0]
        add("Fig_stage_gradient_profiles", panel, f"{group} Anticipation minus Outcome G1 slope", r, "Model_A_base")
    r = models["A"]["simple"].loc[models["A"]["simple"]["contrast"] == "SZ minus HC difference in stage-gradient differentiation"].iloc[0]
    add("Fig_stage_gradient_profiles", "A-B", "Group x Stage x G1", r, "Model_A_base")
    add("Fig_individual_stage_slopes", "B", "Adjusted SZ minus HC Delta slope", delta_models.loc[delta_models["Model"] == "Adjusted_Age_Sex_MeanFD_HC3"].iloc[0], "Delta_slope_adjusted_HC3")
    add("Fig_stage_difference_profile", "A", "GEE Group x Stage x categorical Bin", {"p": gee_omnibus["p"].iloc[0]}, "GEE_exchangeable")
    add("Fig_stage_difference_profile", "A", "Clustered OLS Group x Stage x categorical Bin", {"p": ols_omnibus["p"].iloc[0]}, "OLS_subject_clustered")
    for _, row in brain_results.iterrows():
        add("Fig_brain_behavior", row["Analysis_ID"], f"{row['Behavior']} adjusted association", row, "HC3_adjusted", row["q_FDR"])
        stat_rows.append(
            {
                "Figure": "Fig_brain_behavior",
                "Panel": row["Analysis_ID"],
                "Test": f"{row['Behavior']} Group x Behavior",
                "Effect": row["Group_behavior_interaction_beta"],
                "CI95_low": row["Group_behavior_interaction_CI95_low"],
                "CI95_high": row["Group_behavior_interaction_CI95_high"],
                "p": row["Group_behavior_interaction_p"],
                "q": np.nan,
                "Source_model": "HC3_group_interaction",
            }
        )
    stats = pd.DataFrame(stat_rows)
    write_tsv(stats, source_dir / "figure_statistics.tsv")
    return stats


def final_tables(desc, models, delta_models, gee_omnibus, ols_omnibus, slope_wide, brain_results):
    out = ROOT / "15_FINAL_TABLES"
    write_tsv(desc, out / "Table_reward_stage_descriptives.tsv")
    for label, name in [("A", "A"), ("B", "B"), ("C", "C")]:
        write_tsv(models[label]["coeff"], out / f"Table_stage_gradient_model_{name}.tsv")
    simple_all = pd.concat([models[k]["simple"].assign(Model=k) for k in ["A", "B", "C"]], ignore_index=True)
    write_tsv(simple_all, out / "Table_stage_gradient_simple_effects.tsv")
    categorical = pd.concat(
        [
            gee_omnibus.assign(Model="GEE_exchangeable"),
            ols_omnibus.rename(columns={"statistic": "chi2", "df_num": "df"}).assign(Model="OLS_subject_clustered"),
        ],
        ignore_index=True,
        sort=False,
    )
    write_tsv(categorical, out / "Table_categorical_shape_tests.tsv")
    write_tsv(slope_wide, out / "Table_individual_slopes.tsv")
    write_tsv(delta_models, out / "Table_delta_slope_group_test.tsv")
    write_tsv(brain_results, out / "Table_brain_behavior.tsv")


def provenance_table() -> pd.DataFrame:
    paths = [
        ("Model A historical script", ARCHIVE / "20_current64_reward_stage_fc" / "02_primary_group_stage_g1" / "02_primary_group_stage_g1.py", "RECOVERED"),
        ("Model B historical script", ARCHIVE / "20_current64_reward_stage_fc" / "05_covariate_adjusted" / "01_covariate_adjusted_primary_and_bins.py", "RECOVERED"),
        ("Model C historical script", ARCHIVE / "20_current64_reward_stage_fc" / "06_gradient_sensitive_covariates" / "01_final_gradient_sensitive_covariate_model.py", "RECOVERED"),
        ("Historical manuscript slope table", ARCHIVE / "20_current64_reward_stage_fc" / "01_fixed375_stage_profiles_manuscript" / "subject_stage_slopes.tsv", "RECOVERED_VALIDATION_ONLY"),
        ("Historical categorical generating script", ROOT.parent / "AUDIT_REPORT.md", "NOT_RECOVERED_CHECKPOINT_ONLY"),
        ("Historical brain-behavior specification", ROOT.parent / "02_main_375" / "28_reward_behavior_stage_analysis" / "01_primary_delta_slope_behavior.py", "PARTIAL_375_METHOD_PROVENANCE"),
    ]
    rows = []
    for role, path, status in paths:
        rows.append({"role": role, "path": str(path.resolve()), "sha256": sha256_file(path) if path.is_file() else "NA", "status": status})
    table = pd.DataFrame(rows)
    write_tsv(table, ROOT / "16_VALIDATION" / "stage_analysis_provenance.tsv")
    return table


def main() -> int:
    log_dir = ROOT / "17_LOGS"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    log_path = log_dir / f"stage_analysis_steps6_to10_{timestamp}.log"
    require(not log_path.exists(), f"Log exists: {log_path}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    script_path = Path(__file__).resolve()
    logging.info("CURRENT64 Steps 6-10 started")
    logging.info("Input=%s SHA256=%s", STAGE_INPUT, sha256_file(STAGE_INPUT))
    logging.info("Behavior=%s SHA256=%s", BEHAVIOR_INPUT, sha256_file(BEHAVIOR_INPUT))
    logging.info("Script=%s SHA256=%s", script_path, sha256_file(script_path))

    mapping = pd.read_csv(ROI_MAPPING, sep="\t")
    not_assigned = pd.read_csv(NOT_ASSIGNED, sep="\t")
    require(mapping["reward_stage"].value_counts().to_dict() == {"Anticipation": 43, "Outcome": 19, "Not_assigned": 3}, "Reward-stage mapping changed")
    require(not_assigned["roi_name"].tolist() == ["VTA_bi", "SNr_bi", "VP_VeP_bi"], "Not-assigned ROI provenance mismatch")

    df = normalize_stage_input()
    subject_level = df.rename(
        columns={"subject": "Subject", "group": "Group", "bin": "Bin", "stage": "Stage", "age": "Age", "sex": "Sex", "meanFD": "MeanFD", "G1_centered": "G1_bin_center"}
    )[["Subject", "Group", "Bin", "Stage", "FC", "Age", "Sex", "MeanFD", "G1_bin_center"]].copy()
    subject_level["Group"] = subject_level["Group"].astype(str)
    subject_level["Stage"] = subject_level["Stage"].astype(str)
    subject_level["Sex"] = subject_level["Sex"].astype(str)
    write_tsv(subject_level, ROOT / "06_REWARD_STAGE" / "reward_stage_subject_level.tsv")
    desc_input = subject_level.copy()
    desc = descriptive_ci(desc_input, ["Group", "Stage", "Bin"], "FC")
    write_tsv(desc, ROOT / "06_REWARD_STAGE" / "reward_stage_descriptives.tsv")

    slope_long, slope_wide, slope_qc = build_individual_slopes(df)
    delta_models = delta_slope_models(slope_wide)
    models, lrt, moderation = run_mixed_models(df)
    gee_omnibus, ols_omnibus, estimated, delta, delta_desc = run_categorical(df)
    brain_results, brain_fig = run_brain_behavior(slope_wide)
    validation = historical_validation(models, delta_models, gee_omnibus, ols_omnibus, brain_results)
    figure_stats = write_figure_sources(desc, slope_long, slope_wide, delta_desc, brain_fig, models, delta_models, gee_omnibus, ols_omnibus, brain_results)
    final_tables(desc, models, delta_models, gee_omnibus, ols_omnibus, slope_wide, brain_results)
    provenance = provenance_table()

    environment = "\n".join(
        [
            f"Generated: {datetime.now().astimezone().isoformat()}",
            f"Platform: {platform.platform()}",
            f"Python: {sys.version.replace(chr(10), ' ')}",
            f"NumPy: {np.__version__}",
            f"pandas: {pd.__version__}",
            f"SciPy: {scipy.__version__}",
            f"statsmodels: {statsmodels.__version__}",
            f"patsy: {patsy.__version__}",
            f"Vendored packages: {VENDOR}",
        ]
    ) + "\n"
    write_dynamic_text(environment, ROOT / "17_LOGS" / "python_environment.txt", ("Generated:",))

    mixed_warning_count = sum(bool(str(x)) for x in pd.read_csv(ROOT / "07_STAGE_GRADIENT_MODELS" / "diagnostics" / "mixed_model_diagnostics.tsv", sep="\t")["warnings"].fillna(""))
    summary = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "scope": "Steps 6-10 plus source-data preparation; normative modules excluded",
        "N_subjects": 64,
        "N_stage_rows": 1280,
        "N_slopes": {"Anticipation": 64, "Outcome": 64, "Delta": 64},
        "mixed_model_warning_rows": int(mixed_warning_count),
        "categorical_GEE_p": float(gee_omnibus["p"].iloc[0]),
        "categorical_clustered_OLS_p": float(ols_omnibus["p"].iloc[0]),
        "brain_behavior_tests": 4,
        "normative_deviation_run": False,
        "bin10_posthoc_run": False,
        "figures_rendered": False,
    }
    write_dynamic_text(
        json.dumps(summary, indent=2) + "\n",
        ROOT / "16_VALIDATION" / "stage_analysis_execution_summary.json",
        ('  "completed_at":',),
    )
    logging.info("Outputs generated before R rendering: %d validation rows; %d figure-stat rows", len(validation), len(figure_stats))
    logging.info("CURRENT64 Steps 6-10 statistical stage completed")
    print(json.dumps(summary, indent=2))
    print(f"LOG_PATH={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
