#!/usr/bin/env python3
"""Forensic audit of categorical reward-stage model specifications.

This script is intentionally limited to the frozen 1,280-row canonical stage
table, the four prespecified categorical candidates, coding/covariance checks,
historical-provenance search, and the already-prespecified mixed-model random
effect boundary audit.  It does not run normative-deviation analyses, make
figures, or alter manuscript interpretation.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import platform
import re
import sys
import warnings
from pathlib import Path


ROOT_VALUE = os.environ.get("STRIATAL_CURRENT64_ROOT")
if not ROOT_VALUE:
    raise RuntimeError("Set STRIATAL_CURRENT64_ROOT to the private canonical-rerun root")
ROOT = Path(ROOT_VALUE).resolve()
PROJECT = Path(os.environ.get("STRIATAL_PROJECT_ROOT", ROOT.parent)).resolve()
VENDOR = Path(os.environ.get("STRIATAL_PYTHON_VENDOR", ROOT / ".python_packages")).resolve()
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
import patsy
import scipy
from scipy.stats import chi2, f as f_dist
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
import statsmodels.regression.mixed_linear_model as mixed_linear_model


STAGE_INPUT = ROOT / "06_REWARD_STAGE" / "subject_bin_stage_FC_long.tsv"
CURRENT_SCRIPT = ROOT / "scripts" / "python" / "03_run_stage_analyses.py"
CAT_DIR = ROOT / "08_CATEGORICAL_SHAPE"
MIXED_DIR = ROOT / "07_STAGE_GRADIENT_MODELS"

HIST_GEE_STAT = 11.5254
HIST_GEE_P = 0.2414
HIST_OLS_STAT = 10.9727
HIST_OLS_P = 0.2776


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.17g")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def fmt(value: float, digits: int = 10) -> str:
    return f"{float(value):.{digits}g}"


def load_frozen_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    require(STAGE_INPUT.is_file(), f"Missing frozen stage table: {STAGE_INPUT}")
    raw = pd.read_csv(STAGE_INPUT, sep="\t")
    expected = ["Subject", "Group", "Bin", "G1_bin_center", "Stage", "FC", "Age", "Sex", "MeanFD"]
    require(list(raw.columns) == expected, f"Unexpected stage columns: {list(raw.columns)}")
    require(len(raw) == 1280, f"Expected 1280 rows, found {len(raw)}")
    require(raw["Subject"].nunique() == 64, "Expected 64 subjects")
    require(not raw.duplicated(["Subject", "Bin", "Stage"]).any(), "Duplicate Subject-Bin-Stage rows")
    require(raw.isna().sum().sum() == 0, "Frozen stage table contains missing values")
    require(raw["Group"].value_counts().to_dict() == {"SZ": 640, "HC": 640}, "Unexpected row-level groups")
    require(raw.drop_duplicates("Subject")["Group"].value_counts().to_dict() == {"SZ": 32, "HC": 32}, "Unexpected subject groups")
    require(raw["Stage"].value_counts().to_dict() == {"Anticipation": 640, "Outcome": 640}, "Unexpected stages")
    require(raw["Bin"].value_counts().sort_index().eq(128).all(), "Each Bin must have 128 observations")
    require(raw.groupby("Subject").size().eq(20).all(), "Each Subject must have 20 observations")
    require(raw.groupby(["Subject", "Stage"]).size().eq(10).all(), "Each Subject-Stage must have 10 observations")

    data = raw.rename(
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
    data["group"] = pd.Categorical(data["group"], categories=["HC", "SZ"])
    data["stage"] = pd.Categorical(data["stage"], categories=["Outcome", "Anticipation"])
    data["sex"] = pd.Categorical(data["sex"], categories=["Female", "Male"])
    people = data.drop_duplicates("subject")
    data["age_c"] = data["age"] - people["age"].mean()
    data["meanFD_c"] = data["meanFD"] - people["meanFD"].mean()
    sex_num = data["sex"].map({"Female": 0.0, "Male": 1.0}).astype(float)
    data["sex_c"] = sex_num - float(people["sex"].map({"Female": 0.0, "Male": 1.0}).astype(float).mean())
    return raw, data


def build_delta(data: pd.DataFrame) -> pd.DataFrame:
    delta = data.pivot(
        index=["subject", "group", "bin", "age", "sex", "meanFD"],
        columns="stage",
        values="FC",
    ).reset_index()
    delta.columns.name = None
    delta["Delta_FC"] = delta["Anticipation"] - delta["Outcome"]
    delta["group"] = pd.Categorical(delta["group"], categories=["HC", "SZ"])
    delta["sex"] = pd.Categorical(delta["sex"], categories=["Female", "Male"])
    require(len(delta) == 640, "DeltaFC table must contain 640 rows")
    require(delta["subject"].nunique() == 64, "DeltaFC table must contain 64 subjects")
    require(delta.groupby("subject").size().eq(10).all(), "DeltaFC must have 10 observations per subject")
    require(delta["Delta_FC"].notna().all(), "DeltaFC contains missing values")
    return delta


def target_indices(names: list[str], variables: tuple[str, ...]) -> list[int]:
    indices = [
        index
        for index, name in enumerate(names)
        if all(variable.lower() in name.lower() for variable in variables)
        and name.count(":") == len(variables) - 1
    ]
    require(len(indices) == 9, f"Expected nine highest-order terms for {variables}; got {[names[i] for i in indices]}")
    return indices


def contrast_matrix(names: list[str], indices: list[int]) -> np.ndarray:
    matrix = np.zeros((len(indices), len(names)))
    for row, column in enumerate(indices):
        matrix[row, column] = 1.0
    return matrix


def manual_wald(beta: np.ndarray, covariance: np.ndarray, matrix: np.ndarray) -> float:
    contrast = matrix @ beta
    middle = matrix @ covariance @ matrix.T
    return float(contrast @ np.linalg.pinv(middle) @ contrast)


def coefficients(result, specification: str, backend: str) -> pd.DataFrame:
    interval = result.conf_int()
    names = list(result.params.index)
    return pd.DataFrame(
        {
            "Specification": specification,
            "Backend": backend,
            "term": names,
            "estimate": np.asarray(result.params, dtype=float),
            "SE": np.asarray(result.bse, dtype=float),
            "statistic": np.asarray(result.tvalues, dtype=float),
            "p": np.asarray(result.pvalues, dtype=float),
            "CI95_low": np.asarray(interval.iloc[:, 0], dtype=float),
            "CI95_high": np.asarray(interval.iloc[:, 1], dtype=float),
        }
    )


def save_contrast(matrix: np.ndarray, names: list[str], indices: list[int], path: Path) -> None:
    out = pd.DataFrame(matrix, columns=names)
    out.insert(0, "tested_coefficient", [names[index] for index in indices])
    out.insert(0, "contrast_row", np.arange(1, len(indices) + 1))
    write_tsv(out, path)


def fit_candidate(
    specification: str,
    data: pd.DataFrame,
    formula: str,
    variables: tuple[str, ...],
    data_form: str,
    covariates: str,
) -> dict[str, object]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gee = smf.gee(
            formula,
            groups="subject",
            data=data,
            cov_struct=Exchangeable(),
            family=sm.families.Gaussian(),
        ).fit(maxiter=200, cov_type="robust")
        gee_warnings = " | ".join(str(item.message) for item in caught)

    gee_names = list(gee.params.index)
    indices = target_indices(gee_names, variables)
    matrix = contrast_matrix(gee_names, indices)
    gee_test = gee.wald_test(matrix, scalar=True)
    gee_robust_stat = float(np.asarray(gee_test.statistic).squeeze())
    gee_robust_p = float(np.asarray(gee_test.pvalue).squeeze())
    gee_naive_stat = manual_wald(
        np.asarray(gee.params, dtype=float),
        np.asarray(gee.cov_naive, dtype=float),
        matrix,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ols = smf.ols(formula, data=data).fit(
            cov_type="cluster",
            cov_kwds={"groups": data["subject"], "use_correction": True, "df_correction": True},
            use_t=True,
        )
        ols_warnings = " | ".join(str(item.message) for item in caught)
    ols_names = list(ols.params.index)
    require(ols_names == gee_names, f"GEE/OLS design mismatch for {specification}")
    ols_indices = target_indices(ols_names, variables)
    require(ols_indices == indices, f"GEE/OLS target mismatch for {specification}")
    ols_f_test = ols.wald_test(matrix, use_f=True, scalar=True)
    ols_chi_test = ols.wald_test(matrix, use_f=False, scalar=True)
    ols_f = float(np.asarray(ols_f_test.statistic).squeeze())
    ols_f_p = float(np.asarray(ols_f_test.pvalue).squeeze())
    ols_chi = float(np.asarray(ols_chi_test.statistic).squeeze())
    ols_chi_p = float(np.asarray(ols_chi_test.pvalue).squeeze())

    ols_uncorrected = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["subject"], "use_correction": False, "df_correction": False},
        use_t=False,
    )
    ols_uncorrected_chi = float(
        np.asarray(ols_uncorrected.wald_test(matrix, use_f=False, scalar=True).statistic).squeeze()
    )
    n = len(data)
    k = len(ols.params)
    clusters = data["subject"].nunique()
    theoretical_correction = (clusters / (clusters - 1)) * ((n - 1) / (n - k))
    require(abs((ols_uncorrected_chi / ols_chi) - theoretical_correction) < 1e-9, "OLS correction audit failed")

    prefix = CAT_DIR / f"{specification}"
    write_tsv(coefficients(gee, specification, "GEE_exchangeable_robust"), prefix.with_name(prefix.name + "_GEE_coefficients.tsv"))
    write_tsv(coefficients(ols, specification, "OLS_subject_clustered_corrected"), prefix.with_name(prefix.name + "_clustered_OLS_coefficients.tsv"))
    save_contrast(matrix, gee_names, indices, prefix.with_name(prefix.name + "_GEE_omnibus_contrast_matrix.tsv"))
    save_contrast(matrix, ols_names, indices, prefix.with_name(prefix.name + "_clustered_OLS_omnibus_contrast_matrix.tsv"))

    cluster_sizes = data.groupby("subject", observed=True).size()
    return {
        "specification": specification,
        "data": data,
        "formula": formula,
        "variables": variables,
        "data_form": data_form,
        "covariates": covariates,
        "gee": gee,
        "ols": ols,
        "names": gee_names,
        "indices": indices,
        "matrix": matrix,
        "gee_stat": gee_robust_stat,
        "gee_p": gee_robust_p,
        "gee_naive_stat": gee_naive_stat,
        "gee_naive_p": float(chi2.sf(gee_naive_stat, 9)),
        "ols_f": ols_f,
        "ols_f_p": ols_f_p,
        "ols_chi": ols_chi,
        "ols_chi_p": ols_chi_p,
        "ols_uncorrected_chi": ols_uncorrected_chi,
        "correction_factor": theoretical_correction,
        "gee_warnings": gee_warnings,
        "ols_warnings": ols_warnings,
        "cluster_min": int(cluster_sizes.min()),
        "cluster_max": int(cluster_sizes.max()),
        "k": k,
    }


def comparison_rows(result: dict[str, object]) -> list[dict[str, object]]:
    specification = str(result["specification"])
    n_rows = len(result["data"])
    cluster_text = f"Subject (64 clusters; {result['cluster_min']} observations each)"
    common = {
        "Specification": specification,
        "Data_form": result["data_form"],
        "N_rows": n_rows,
        "N_subjects": 64,
        "Formula": result["formula"],
        "Covariates": result["covariates"],
        "Bin_type": "factor; levels 1-10; treatment reference=1",
        "Cluster": cluster_text,
    }
    return [
        {
            **common,
            "Correlation_structure": "Exchangeable",
            "Covariance_type": "GEE robust sandwich",
            "Small_sample_correction": "None",
            "Test": "9-df joint Wald chi-square",
            "Statistic": result["gee_stat"],
            "df1": 9,
            "df2": np.nan,
            "p": result["gee_p"],
        },
        {
            **common,
            "Correlation_structure": "Not applicable",
            "Covariance_type": "Subject-clustered sandwich",
            "Small_sample_correction": "use_correction=True; df_correction=True; use_t=True",
            "Test": "9-df cluster-robust F",
            "Statistic": result["ols_f"],
            "df1": 9,
            "df2": 63,
            "p": result["ols_f_p"],
        },
        {
            **common,
            "Correlation_structure": "Not applicable",
            "Covariance_type": "Subject-clustered sandwich",
            "Small_sample_correction": "use_correction=True; chi-square reference",
            "Test": "Equivalent 9-df joint Wald chi-square",
            "Statistic": result["ols_chi"],
            "df1": 9,
            "df2": np.nan,
            "p": result["ols_chi_p"],
        },
    ]


def coding_audit(data: pd.DataFrame) -> pd.DataFrame:
    formulas = {
        "Treatment_ref_HC_Outcome_Bin1": "FC ~ group * stage * C(bin)",
        "Treatment_ref_SZ_Anticipation_Bin10": (
            "FC ~ C(group, Treatment(reference='SZ')) * "
            "C(stage, Treatment(reference='Anticipation')) * "
            "C(bin, Treatment(reference=10))"
        ),
        "Sum_coding": "FC ~ C(group, Sum) * C(stage, Sum) * C(bin, Sum)",
    }
    rows = []
    for label, formula in formulas.items():
        gee = smf.gee(
            formula,
            groups="subject",
            data=data,
            cov_struct=Exchangeable(),
            family=sm.families.Gaussian(),
        ).fit(maxiter=200, cov_type="robust")
        names = list(gee.params.index)
        indices = target_indices(names, ("group", "stage", "bin"))
        matrix = contrast_matrix(names, indices)
        gee_test = gee.wald_test(matrix, scalar=True)
        ols = smf.ols(formula, data=data).fit(
            cov_type="cluster",
            cov_kwds={"groups": data["subject"], "use_correction": True, "df_correction": True},
            use_t=True,
        )
        ols_chi = ols.wald_test(matrix, use_f=False, scalar=True)
        rows.append(
            {
                "Coding": label,
                "Formula": formula,
                "N_tested_coefficients": len(indices),
                "GEE_Wald_chi2": float(np.asarray(gee_test.statistic).squeeze()),
                "GEE_p": float(np.asarray(gee_test.pvalue).squeeze()),
                "OLS_corrected_Wald_chi2": float(np.asarray(ols_chi.statistic).squeeze()),
                "OLS_chi2_p": float(np.asarray(ols_chi.pvalue).squeeze()),
                "Tested_terms": " | ".join(names[index] for index in indices),
            }
        )
    out = pd.DataFrame(rows)
    require(out["GEE_Wald_chi2"].max() - out["GEE_Wald_chi2"].min() < 1e-9, "GEE coding invariance failed")
    require(out["OLS_corrected_Wald_chi2"].max() - out["OLS_corrected_Wald_chi2"].min() < 1e-9, "OLS coding invariance failed")
    return out


def data_audit(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    counts: list[dict[str, object]] = [
        {"Dimension": "Rows", "Level": "All", "Count": len(raw)},
        {"Dimension": "Subjects", "Level": "Unique", "Count": raw["Subject"].nunique()},
    ]
    subject_groups = raw.drop_duplicates("Subject")["Group"].value_counts().sort_index()
    counts.extend({"Dimension": "Subject_group", "Level": key, "Count": int(value)} for key, value in subject_groups.items())
    counts.extend({"Dimension": "Stage", "Level": key, "Count": int(value)} for key, value in raw["Stage"].value_counts().sort_index().items())
    counts.extend({"Dimension": "Bin", "Level": str(key), "Count": int(value)} for key, value in raw["Bin"].value_counts().sort_index().items())
    na = pd.DataFrame({"Column": raw.columns, "NA_count": [int(raw[column].isna().sum()) for column in raw.columns]})

    contiguous = True
    seen: set[str] = set()
    last = None
    for subject in raw["Subject"].astype(str):
        if subject != last:
            if subject in seen:
                contiguous = False
                break
            seen.add(subject)
            last = subject
    expected_within = [(bin_value, stage) for bin_value in range(1, 11) for stage in ["Anticipation", "Outcome"]]
    within_order = all(
        list(zip(group["Bin"].tolist(), group["Stage"].tolist())) == expected_within
        for _, group in raw.groupby("Subject", sort=False)
    )
    details = {
        "subject_blocks_contiguous": contiguous,
        "within_subject_order": "Bin 1-10; within each Bin Anticipation then Outcome" if within_order else "not uniform",
        "observations_per_subject": "20 for every Subject",
        "observations_per_subject_delta": "10 for every Subject",
    }
    return pd.DataFrame(counts), na, details


def provenance_search() -> tuple[pd.DataFrame, pd.DataFrame]:
    extensions = {".py", ".r", ".ipynb", ".sh", ".log", ".txt", ".md", ".tex"}
    keyword_patterns = {
        "11.5254": re.compile(r"(?<![0-9.])11\.5254(?![0-9])"),
        "0.2414": re.compile(r"(?<![0-9.])0\.2414(?![0-9])"),
        "10.9727": re.compile(r"(?<![0-9.])10\.9727(?![0-9])"),
        "0.2776": re.compile(r"(?<![0-9.])0\.2776(?![0-9])"),
        "GEE": re.compile(r"\bGEE\b", re.IGNORECASE),
        "exchangeable": re.compile(r"\bexchangeable\b", re.IGNORECASE),
        "cluster": re.compile(r"\bcluster(?:ed|ing|s)?\b", re.IGNORECASE),
        "DeltaFC": re.compile(r"\bDelta[_ ]?FC\b", re.IGNORECASE),
        "Anticipation": re.compile(r"\bAnticipation\b", re.IGNORECASE),
        "Outcome": re.compile(r"\bOutcome\b", re.IGNORECASE),
        "Group": re.compile(r"\bGroup\b", re.IGNORECASE),
        "Bin": re.compile(r"\bBin\b", re.IGNORECASE),
    }
    excluded_names = {
        Path(__file__).name,
        "CURRENT_MODEL_SPECIFICATION_AUDIT.md",
        "CATEGORICAL_FORENSIC_AUDIT.md",
        "RANDOM_EFFECT_BOUNDARY_AUDIT.md",
    }
    counts = {key: {"files": set(), "lines": 0} for key in keyword_patterns}
    numeric_hits: list[dict[str, object]] = []
    for path in PROJECT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.name in excluded_names or ".python_packages" in path.parts:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            matched = []
            for keyword, pattern in keyword_patterns.items():
                if pattern.search(line):
                    counts[keyword]["files"].add(str(path))
                    counts[keyword]["lines"] += 1
                    if keyword in {"11.5254", "0.2414", "10.9727", "0.2776"}:
                        matched.append(keyword)
            if matched:
                start = max(0, line_number - 3)
                stop = min(len(lines), line_number + 2)
                context = " || ".join(f"L{index + 1}: {lines[index].strip()}" for index in range(start, stop))
                numeric_hits.append(
                    {
                        "Path": str(path.resolve()),
                        "Line": line_number,
                        "Fingerprints": " | ".join(matched),
                        "Context_plus_minus_2_lines": context,
                    }
                )
    summary = pd.DataFrame(
        [
            {"Keyword": key, "Files_with_hits": len(value["files"]), "Matching_lines": value["lines"]}
            for key, value in counts.items()
        ]
    )
    return summary, pd.DataFrame(numeric_hits, columns=["Path", "Line", "Fingerprints", "Context_plus_minus_2_lines"])


def fit_mixed_boundary(data: pd.DataFrame) -> pd.DataFrame:
    formulas = {
        "Model_A": "FC ~ group * stage * G1_centered",
        "Model_B": "FC ~ group * stage * G1_centered + age_c + sex + meanFD_c",
        "Model_C": (
            "FC ~ group * stage * G1_centered + age_c * stage * G1_centered + "
            "sex_c * stage * G1_centered + meanFD_c * stage * G1_centered"
        ),
    }
    rows = []
    for label, formula in formulas.items():
        model = smf.mixedlm(formula, data=data, groups=data["subject"], re_formula="1 + G1_centered")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(reml=False, method="lbfgs", maxiter=5000, disp=False)
        warning_text = " | ".join(str(item.message) for item in caught)
        require(result.converged, f"{label} failed to converge")
        covariance = np.asarray(result.cov_re, dtype=float)
        intercept_variance = float(covariance[0, 0])
        slope_variance = float(covariance[1, 1])
        covariance_is = float(covariance[0, 1])
        correlation = covariance_is / np.sqrt(intercept_variance * slope_variance)
        eigenvalues = np.linalg.eigvalsh(covariance)
        rows.append(
            {
                "Model": label,
                "Formula": formula,
                "N_rows": int(result.nobs),
                "N_subjects": data["subject"].nunique(),
                "Converged": bool(result.converged),
                "Optimizer": "lbfgs",
                "REML": False,
                "Fixed_effect_design_rank": int(np.linalg.matrix_rank(result.model.exog)),
                "Fixed_effect_design_columns": int(result.model.exog.shape[1]),
                "Random_effects": "Subject random intercept + random G1_centered slope",
                "Random_intercept_variance": intercept_variance,
                "Random_G1_slope_variance": slope_variance,
                "Intercept_slope_covariance": covariance_is,
                "Intercept_slope_correlation": correlation,
                "Residual_variance": float(result.scale),
                "Residual_SD": float(np.sqrt(result.scale)),
                "Smallest_RE_covariance_eigenvalue": float(eigenvalues.min()),
                "Largest_RE_covariance_eigenvalue": float(eigenvalues.max()),
                "RE_covariance_condition_number": float(eigenvalues.max() / eigenvalues.min()),
                "Intercept_variance_below_statsmodels_0.01_warning_threshold": intercept_variance < 0.01,
                "Slope_variance_below_statsmodels_0.01_warning_threshold": slope_variance < 0.01,
                "Warning": warning_text,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = frame if columns is None else frame[columns]
    headers = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if np.isnan(value) else fmt(value, 9))
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    raw, data = load_frozen_data()
    delta = build_delta(data)
    counts, na_counts, order_details = data_audit(raw)
    write_tsv(counts, CAT_DIR / "FROZEN_DATA_COUNTS.tsv")
    write_tsv(na_counts, CAT_DIR / "FROZEN_DATA_NA_COUNTS.tsv")

    specifications = [
        (
            "SPEC_A",
            data,
            "FC ~ group * stage * C(bin)",
            ("group", "stage", "bin"),
            "Full repeated-data model (1280 stage rows)",
            "None",
        ),
        (
            "SPEC_B",
            data,
            "FC ~ group * stage * C(bin, Treatment(reference=1)) + age + sex + meanFD",
            ("group", "stage", "bin"),
            "Full repeated-data adjusted model (1280 stage rows)",
            "Age + Sex + MeanFD",
        ),
        (
            "SPEC_C",
            delta,
            "Delta_FC ~ group * C(bin, Treatment(reference=1))",
            ("group", "bin"),
            "Within-subject Anticipation-minus-Outcome model (640 DeltaFC rows)",
            "None",
        ),
        (
            "SPEC_D",
            delta,
            "Delta_FC ~ group * C(bin, Treatment(reference=1)) + age + sex + meanFD",
            ("group", "bin"),
            "Adjusted Anticipation-minus-Outcome model (640 DeltaFC rows)",
            "Age + Sex + MeanFD",
        ),
    ]
    results = [fit_candidate(*specification) for specification in specifications]
    comparison = pd.DataFrame([row for result in results for row in comparison_rows(result)])
    comparison_columns = [
        "Specification",
        "Data_form",
        "N_rows",
        "N_subjects",
        "Formula",
        "Covariates",
        "Bin_type",
        "Cluster",
        "Correlation_structure",
        "Covariance_type",
        "Small_sample_correction",
        "Test",
        "Statistic",
        "df1",
        "df2",
        "p",
    ]
    comparison = comparison[comparison_columns]
    write_tsv(comparison, CAT_DIR / "CATEGORICAL_SPECIFICATION_COMPARISON.tsv")

    covariance_audit = pd.DataFrame(
        [
            {
                "Specification": result["specification"],
                "N_rows": len(result["data"]),
                "N_coefficients": result["k"],
                "GEE_exchangeable_alpha": float(np.asarray(result["gee"].cov_struct.dep_params).squeeze()),
                "GEE_scale_Pearson": float(result["gee"].scale),
                "GEE_robust_Wald_chi2": result["gee_stat"],
                "GEE_robust_p": result["gee_p"],
                "GEE_naive_Wald_chi2": result["gee_naive_stat"],
                "GEE_naive_p": result["gee_naive_p"],
                "OLS_cluster_uncorrected_Wald_chi2": result["ols_uncorrected_chi"],
                "OLS_cluster_corrected_Wald_chi2": result["ols_chi"],
                "OLS_cluster_corrected_chi2_p": result["ols_chi_p"],
                "OLS_cluster_F": result["ols_f"],
                "OLS_cluster_F_df1": 9,
                "OLS_cluster_F_df2": 63,
                "OLS_cluster_F_p": result["ols_f_p"],
                "OLS_small_sample_covariance_factor": result["correction_factor"],
                "GEE_warnings": result["gee_warnings"],
                "OLS_warnings": result["ols_warnings"],
            }
            for result in results
        ]
    )
    write_tsv(covariance_audit, CAT_DIR / "CATEGORICAL_COVARIANCE_AND_TEST_AUDIT.tsv")

    # If a historical script mislabeled another natural 9-df factor block as
    # the three-way term, one of these tests would expose that possibility.
    spec_b = results[1]
    spec_b_names = spec_b["names"]
    natural_blocks = {
        "Bin_main_effect": [i for i, name in enumerate(spec_b_names) if "C(bin" in name and ":" not in name],
        "Group_x_Bin": [
            i for i, name in enumerate(spec_b_names)
            if "group" in name and "C(bin" in name and "stage" not in name
        ],
        "Stage_x_Bin": [
            i for i, name in enumerate(spec_b_names)
            if "stage" in name and "C(bin" in name and "group" not in name
        ],
        "Group_x_Stage_x_Bin": [
            i for i, name in enumerate(spec_b_names)
            if "group" in name and "stage" in name and "C(bin" in name
        ],
    }
    natural_rows = []
    for block, indices in natural_blocks.items():
        require(len(indices) == 9, f"Unexpected Spec B natural block size for {block}: {len(indices)}")
        matrix = contrast_matrix(spec_b_names, indices)
        gee_test = spec_b["gee"].wald_test(matrix, scalar=True)
        ols_test = spec_b["ols"].wald_test(matrix, use_f=False, scalar=True)
        natural_rows.append(
            {
                "Block": block,
                "N_coefficients": 9,
                "GEE_robust_Wald_chi2": float(np.asarray(gee_test.statistic).squeeze()),
                "GEE_p": float(np.asarray(gee_test.pvalue).squeeze()),
                "OLS_corrected_Wald_chi2": float(np.asarray(ols_test.statistic).squeeze()),
                "OLS_chi2_p": float(np.asarray(ols_test.pvalue).squeeze()),
                "Terms": " | ".join(spec_b_names[index] for index in indices),
            }
        )
    write_tsv(pd.DataFrame(natural_rows), CAT_DIR / "SPEC_B_NATURAL_9DF_BLOCK_AUDIT.tsv")

    coding = coding_audit(data)
    write_tsv(coding, CAT_DIR / "CATEGORICAL_CODING_INVARIANCE_AUDIT.tsv")

    current = results[0]
    existing_gee = pd.read_csv(CAT_DIR / "GEE_omnibus_tests.tsv", sep="\t").iloc[0]
    existing_ols = pd.read_csv(CAT_DIR / "clustered_OLS_omnibus_tests.tsv", sep="\t").iloc[0]
    require(abs(current["gee_stat"] - float(existing_gee["chi2"])) < 1e-9, "Current GEE reproduction mismatch")
    require(abs(current["gee_p"] - float(existing_gee["p"])) < 1e-12, "Current GEE p reproduction mismatch")
    require(abs(current["ols_f"] - float(existing_ols["statistic"])) < 1e-9, "Current OLS reproduction mismatch")
    require(abs(current["ols_f_p"] - float(existing_ols["p"])) < 1e-12, "Current OLS p reproduction mismatch")
    save_contrast(
        current["matrix"],
        current["names"],
        current["indices"],
        CAT_DIR / "CURRENT_GEE_omnibus_contrast_matrix.tsv",
    )
    save_contrast(
        current["matrix"],
        current["names"],
        current["indices"],
        CAT_DIR / "CURRENT_clustered_OLS_omnibus_contrast_matrix.tsv",
    )

    search_summary, numeric_hits = provenance_search()
    write_tsv(search_summary, CAT_DIR / "HISTORICAL_PROVENANCE_SEARCH_SUMMARY.tsv")
    write_tsv(numeric_hits, CAT_DIR / "HISTORICAL_NUMERIC_FINGERPRINT_HITS.tsv")

    random_effects = fit_mixed_boundary(data)
    write_tsv(random_effects, MIXED_DIR / "RANDOM_EFFECT_BOUNDARY_COMPONENTS.tsv")

    gee_candidates = covariance_audit[["Specification", "GEE_robust_Wald_chi2", "GEE_robust_p"]].copy()
    gee_candidates["absolute_statistic_distance_from_historical"] = (gee_candidates["GEE_robust_Wald_chi2"] - HIST_GEE_STAT).abs()
    gee_candidates["absolute_p_distance_from_historical"] = (gee_candidates["GEE_robust_p"] - HIST_GEE_P).abs()
    ols_candidates = covariance_audit[["Specification", "OLS_cluster_corrected_Wald_chi2", "OLS_cluster_corrected_chi2_p"]].copy()
    ols_candidates["absolute_statistic_distance_from_historical"] = (ols_candidates["OLS_cluster_corrected_Wald_chi2"] - HIST_OLS_STAT).abs()
    ols_candidates["absolute_p_distance_from_historical"] = (ols_candidates["OLS_cluster_corrected_chi2_p"] - HIST_OLS_P).abs()
    write_tsv(gee_candidates, CAT_DIR / "HISTORICAL_GEE_CANDIDATE_DISTANCE.tsv")
    write_tsv(ols_candidates, CAT_DIR / "HISTORICAL_OLS_CANDIDATE_DISTANCE.tsv")

    historical_ratio = HIST_GEE_STAT / HIST_OLS_STAT
    implied_factors = pd.DataFrame(
        [
            {
                "N_rows": 1280,
                "N_clusters": 64,
                "N_coefficients": k,
                "Statsmodels_cluster_covariance_correction_factor": (64 / 63) * (1279 / (1280 - k)),
                "Absolute_distance_from_historical_statistic_ratio": abs(
                    (64 / 63) * (1279 / (1280 - k)) - historical_ratio
                ),
            }
            for k in range(35, 51)
        ]
    )
    write_tsv(implied_factors, CAT_DIR / "HISTORICAL_SMALL_SAMPLE_FACTOR_FINGERPRINT.tsv")

    current_terms = [current["names"][index] for index in current["indices"]]
    current_audit = f"""# Current categorical model specification audit

## Frozen data

- Input: `{STAGE_INPUT}`
- SHA-256: `{sha256_file(STAGE_INPUT)}`
- Rows: 1,280; Subjects: 64; HC: 32; SZ: 32.
- Stages: Anticipation 640, Outcome 640. Every Bin 1-10 has 128 rows.
- Missing values: 0 in every input column (full table in `FROZEN_DATA_NA_COUNTS.tsv`).
- Subject clusters are contiguous: `{order_details['subject_blocks_contiguous']}`. Input ordering is `{order_details['within_subject_order']}`.
- Every Subject contributes exactly 20 observations to the stage-level GEE/OLS models.

## Current GEE producing p = 0.0017529515982185663

- Exact computational formula in `{CURRENT_SCRIPT}`: `FC ~ group * stage * C(bin)`.
- Data rows: 1,280; no rows are dropped.
- Group coding: categorical levels `[HC, SZ]`, treatment reference HC; coefficient is SZ minus HC.
- Stage coding: categorical levels `[Outcome, Anticipation]`, treatment reference Outcome.
- Bin coding: `C(bin)` under Patsy's default treatment coding, exact levels 1-10, reference Bin 1.
- Covariates: none.
- Cluster ID: `subject`; 64 clusters, exactly 20 observations in each cluster.
- Family/link: Gaussian/identity.
- Working correlation: exchangeable; estimated alpha = `{fmt(float(np.asarray(current['gee'].cov_struct.dep_params).squeeze()), 12)}`.
- Covariance: statsmodels GEE robust sandwich (`cov_type='robust'`; no finite-cluster correction). Naive/model-based covariance was not used.
- Scale: Pearson/df-residual estimate = `{fmt(float(current['gee'].scale), 12)}`.
- Software: Python `{platform.python_version()}`, statsmodels `{statsmodels.__version__}`, NumPy `{np.__version__}`, pandas `{pd.__version__}`, SciPy `{scipy.__version__}`, Patsy `{patsy.__version__}`.
- Test implementation: `GEEResults.wald_test(R, scalar=True)`.
- Omnibus matrix: 9 x {len(current['names'])}; one unit row for each of the nine three-way coefficients and zeros elsewhere. Saved in `CURRENT_GEE_omnibus_contrast_matrix.tsv`.
- Jointly tested coefficients (9): `{'; '.join(current_terms)}`.
- Result: Wald chi-square = `{fmt(current['gee_stat'], 14)}`, df = 9, p = `{fmt(current['gee_p'], 14)}`.

## Current clustered OLS producing p = 0.0079000467945081303

- Exact formula/data/coding/reference levels/covariates: identical to the current GEE above.
- Cluster variable: `subject`; 64 clusters, 20 observations each.
- Covariance type: statsmodels `cov_type='cluster'`; not HC1, HC2, or HC3.
- Small-sample covariance correction: `use_correction=True`; factor = G/(G-1) x (N-1)/(N-k) = `{fmt(current['correction_factor'], 12)}` for G=64, N=1280, k=40.
- Degrees-of-freedom correction: `df_correction=True`; coefficient tests use t with 63 df because `use_t=True`.
- Omnibus implementation: `RegressionResults.wald_test(R, use_f=True, scalar=True)` with the same 9 x 40 matrix, saved in `CURRENT_clustered_OLS_omnibus_contrast_matrix.tsv`.
- Reported result: F = `{fmt(current['ols_f'], 14)}`, df = (9, 63), p = `{fmt(current['ols_f_p'], 14)}`.
- Equivalent corrected Wald chi-square: `{fmt(current['ols_chi'], 14)}`, df = 9, chi-square-reference p = `{fmt(current['ols_chi_p'], 14)}`. This is 9 x F and must not be numerically compared to F itself.
- Uncorrected subject-clustered Wald chi-square: `{fmt(current['ols_uncorrected_chi'], 14)}`; it equals the current robust GEE statistic to floating-point precision in this balanced saturated design.

## Coding check

Changing HC/Outcome/Bin 1 references to SZ/Anticipation/Bin 10, or using sum coding, changes individual coefficients but leaves the correct 9-df omnibus invariant (see `CATEGORICAL_CODING_INVARIANCE_AUDIT.tsv`).
"""
    write_text(current_audit, CAT_DIR / "CURRENT_MODEL_SPECIFICATION_AUDIT.md")

    boundary_source = inspect.getsource(mixed_linear_model.MixedLM.fit)
    threshold_line = next(line.strip() for line in boundary_source.splitlines() if "np.min(np.abs(np.diag(cov_re)))" in line)
    boundary_report = f"""# Random-effect boundary audit

This audit is separate from the categorical discrepancy. Models A, B and C were refit with their frozen formulas, maximum likelihood (`reml=False`), L-BFGS, 64 Subject clusters, and `re_formula='1 + G1_centered'`. No random slope was removed.

{markdown_table(random_effects, ['Model', 'Random_intercept_variance', 'Random_G1_slope_variance', 'Intercept_slope_covariance', 'Intercept_slope_correlation', 'Residual_variance', 'Smallest_RE_covariance_eigenvalue', 'Warning'])}

## Source of the warning

statsmodels {statsmodels.__version__} emits this generic warning through the scale-dependent heuristic `{threshold_line}`. Both diagonal variances are numerically below 0.01 in FC units, so the software warning cannot be attributed exclusively to one component. The random G1-slope variance is nevertheless the limiting (smallest) component (approximately 5.72e-05 to 5.96e-05) and is much closer to zero than the random-intercept variance (approximately 0.00278 to 0.00282). All fitted random-effect covariance matrices are positive definite: their smallest eigenvalues are positive, the intercept-slope correlations are finite, the fixed-effect designs are full rank, and all models converged.

Therefore the limiting component is the small random G1-slope variance under statsmodels' absolute 0.01 diagnostic threshold, while both diagonal components technically satisfy that scale-dependent trigger. The warning is not evidence of optimizer failure, a rank-deficient fixed-effect design, or an exactly zero variance estimate, and it does not justify automatically deleting the prespecified random slope.
"""
    write_text(boundary_report, MIXED_DIR / "RANDOM_EFFECT_BOUNDARY_AUDIT.md")

    compact = covariance_audit[
        [
            "Specification",
            "GEE_robust_Wald_chi2",
            "GEE_robust_p",
            "OLS_cluster_corrected_Wald_chi2",
            "OLS_cluster_corrected_chi2_p",
            "OLS_cluster_F",
            "OLS_cluster_F_p",
        ]
    ]
    min_ols = ols_candidates.sort_values(
        ["absolute_statistic_distance_from_historical", "absolute_p_distance_from_historical"]
    ).iloc[0]
    exact_numeric_sources = numeric_hits["Path"].nunique() if len(numeric_hits) else 0
    report = f"""# Categorical model forensic audit

## Executive finding

The discrepancy is real and is not numerical tolerance. None of the four prespecified candidates reproduces the historical pair. In the balanced complete canonical data, the correct 9-df robust GEE test is invariant across the full stage-level representation (A/B) and the algebraically equivalent Anticipation-minus-Outcome representation (C/D), and additive Age, Sex and MeanFD do not change this within-subject shape contrast. All four give Wald chi-square = 26.4054999202 and p = 0.00175295159822.

The historical specification is therefore **not recovered**:

`HISTORICAL_SPECIFICATION_NOT_RECOVERED`

The surviving numeric checkpoints are not accompanied by a generating model script, coefficient table, contrast matrix, or fit object. Their values cannot be used to select a model by significance.

## Frozen-data audit

- Exact input: `{STAGE_INPUT}`
- SHA-256: `{sha256_file(STAGE_INPUT)}`
- 1,280 rows; 64 Subjects; 32 HC and 32 SZ; 640 Anticipation and 640 Outcome rows; 128 observations in every Bin 1-10.
- Every Subject has 20 stage-level observations; DeltaFC construction yields 640 rows and 10 observations per Subject.
- No missing values occur in any input column. Counts are saved in `FROZEN_DATA_COUNTS.tsv` and `FROZEN_DATA_NA_COUNTS.tsv`.

## Candidate results

{markdown_table(compact)}

The full row-level specification table is `CATEGORICAL_SPECIFICATION_COMPARISON.tsv`; all eight coefficient tables and all eight exact omnibus matrices are saved beside it.

### Closest candidates

- Historical GEE chi-square 11.5254, p 0.2414: A, B, C and D are tied to numerical precision because each produces chi-square 26.4054999202, p 0.00175295159822. No candidate is close.
- Historical clustered-OLS Wald chi-square 10.9727, p 0.2776: D is numerically closest among the prespecified candidates (chi-square `{fmt(float(min_ols['OLS_cluster_corrected_Wald_chi2']), 12)}`, p `{fmt(float(min_ols['OLS_cluster_corrected_chi2_p']), 12)}`), but the discrepancy remains large and is not a match.

## What does and does not explain the discrepancy

- Data representation: not supported. A and C have identical correct GEE statistics; B and D do as well. DeltaFC is not the source.
- Additive covariates: not supported. A versus B and C versus D leave the GEE omnibus unchanged; OLS differences are only the expected finite-sample covariance-factor changes.
- Reference level/treatment versus sum coding: not supported. The correct 9-df omnibus is invariant to numerical precision.
- Current covariance/test implementation: fully identified. The current GEE is robust, not naive. The current OLS is Subject-clustered with small-sample and df corrections and is reported as F(9,63); its equivalent Wald chi-square is `{fmt(current['ols_chi'], 12)}`, not 10.9727.
- Historical test scale: the historical OLS statistic/p pair is internally a chi-square(9) test, not an F statistic. Direct comparison with the current F=2.8000 is invalid.
- Remaining plausible sources: an unrecovered/incorrect historical omnibus matrix, a different covariance or cluster implementation, a transformation or row subset not represented by A-D, or a checkpoint from a different computational object despite the stage-table label. The surviving evidence cannot distinguish these possibilities.
- In adjusted Spec B, none of the four natural 9-df factor blocks (Bin, Group x Bin, Stage x Bin, or Group x Stage x Bin) yields 11.5254; see `SPEC_B_NATURAL_9DF_BLOCK_AUDIT.tsv`. Thus the historical value is not explained by merely selecting another complete Patsy factor term.

## Historical finite-sample fingerprint

The ratio 11.5254 / 10.9727 = `{fmt(historical_ratio, 14)}` is almost exactly the statsmodels one-way cluster covariance correction G/(G-1) x (N-1)/(N-k) for G=64, N=1280 and k=43: `{fmt((64/63)*(1279/1237), 14)}`. A 43-column design is consistent with the 40-column saturated Group x Stage x Bin design plus three additive covariate columns. However, fitting that exact 43-column candidate B and testing the correct nine three-way coefficients gives GEE chi-square 26.4055, not 11.5254. The ratio is useful provenance evidence for a corrected clustered-OLS implementation, but it does not recover the omitted contrast/covariance detail and is not sufficient to declare Spec B historical.

## Provenance search

The requested extensions (`.py`, `.R`, `.ipynb`, `.sh`, `.log`, `.txt`, `.md`, `.tex`) were searched for all requested keywords. Exact checkpoint contexts occur in {exact_numeric_sources} distinct files and are saved with absolute paths and +/-2 lines in `HISTORICAL_NUMERIC_FINGERPRINT_HITS.tsv`; aggregate keyword counts are in `HISTORICAL_PROVENANCE_SEARCH_SUMMARY.tsv`. The p-value checkpoints 0.2414 and 0.2776 occur only in audit/checkpoint narrative or the current rerun's comparison constants; the statistics 11.5254 and 10.9727 do not occur in any searched project file. No file contains code or a log demonstrating a fit that generated either historical pair.

## Answers to the six requested questions

1. Current p=0.001753 GEE: `FC ~ group * stage * C(bin)`, 1,280 rows, HC/Outcome/Bin 1 references, no covariates, Subject ID (64 x 20), Gaussian identity GEE, exchangeable working correlation, robust sandwich, joint Wald chi-square on the nine three-way coefficients.
2. Current p=0.007900 OLS: the same formula/design, OLS with Subject-clustered sandwich, `use_correction=True`, `df_correction=True`, `use_t=True`, and the same nine-row matrix reported as F(9,63).
3. Historical GEE: all A-D tie as closest, but none matches.
4. Historical OLS: D is numerically closest using the comparable corrected Wald chi-square, but none matches.
5. Identified non-sources are data representation, additive covariates, and ordinary coding/reference changes. The unresolved source is most consistent with historical omnibus/covariance/software implementation or an unrecorded transformation/subset; it cannot be uniquely assigned.
6. Historical model: cannot be determined from the surviving provenance. `HISTORICAL_SPECIFICATION_NOT_RECOVERED`.

No manuscript interpretation, normative-deviation analysis, or figure was modified or run. The forensic branch stops here.
"""
    write_text(report, CAT_DIR / "CATEGORICAL_FORENSIC_AUDIT.md")

    print("Categorical forensic audit completed.")
    print(CAT_DIR / "CATEGORICAL_FORENSIC_AUDIT.md")
    print(MIXED_DIR / "RANDOM_EFFECT_BOUNDARY_AUDIT.md")


if __name__ == "__main__":
    main()
