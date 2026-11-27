#!/usr/bin/env python3
"""Finite-sample stability audit for the frozen categorical robust GEE.

Only the prespecified Gaussian-identity exchangeable GEE mean model is used.
The script reuses the preceding audit's analysis-data constructor and verifies
its fingerprint before fitting bias-reduced GEE, delete-one-cluster GEE,
cluster jackknife covariance, and a null Rademacher wild cluster bootstrap.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import multiprocessing as mp
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path


# Avoid BLAS over-subscription when bootstrap fits run in separate processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

AUDIT_DIR = Path(os.environ["STRIATAL_OUTPUT_ROOT"])
PROJECT_ROOT = Path(os.environ["STRIATAL_PROJECT_ROOT"])
PREVIOUS_DIR = Path(os.environ["STRIATAL_MIXEDLM_GEE_OUTPUT"])
PREVIOUS_SCRIPT = Path(__file__).resolve().parents[1] / "mixedlm_vs_gee" / "01_same_data_method_audit.py"
PREVIOUS_FINGERPRINT = PREVIOUS_DIR / "analysis_dataset_fingerprint.json"
VENDOR = Path(os.environ["STRIATAL_PYTHON_VENDOR"]) if os.environ.get("STRIATAL_PYTHON_VENDOR") else None
if VENDOR is not None and VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
import patsy
import PIL
from PIL import Image, ImageDraw, ImageFont
import scipy
from scipy import linalg
from scipy.stats import chi2, pearsonr, spearmanr
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE


FULL_FORMULA = "FC ~ Group * Stage * C(Bin) + age_c + sex_c + meanFD_c"
REDUCED_FORMULA = (
    "FC ~ Group * Stage + Group * C(Bin) + Stage * C(Bin) "
    "+ age_c + sex_c + meanFD_c"
)
EXPECTED_CHI2 = float(os.environ["STRIATAL_EXPECTED_CHI2"])
EXPECTED_P = float(os.environ["STRIATAL_EXPECTED_P"])
STAT_TOL = 1e-8
P_TOL = 1e-10
SEED = 20260915
BOOTSTRAP_B = int(os.environ.get("GEE_AUDIT_BOOTSTRAP_B", "4999"))
N_WORKERS = max(1, min(int(os.environ.get("GEE_AUDIT_WORKERS", "8")), os.cpu_count() or 1))

_BOOT_X = None
_BOOT_GROUPS = None
_BOOT_MU0 = None
_BOOT_RESID = None
_BOOT_TARGET = None
_BOOT_START = None


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


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
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


def load_previous_module():
    require(PREVIOUS_SCRIPT.is_file(), f"Previous audit script missing: {PREVIOUS_SCRIPT}")
    spec = importlib.util.spec_from_file_location("same_data_method_audit", PREVIOUS_SCRIPT)
    require(spec is not None and spec.loader is not None, "Could not import previous audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_locked_analysis_data() -> tuple[pd.DataFrame, dict, object]:
    previous = load_previous_module()
    analysis_dat, calculated = previous.load_analysis_data()
    require(PREVIOUS_FINGERPRINT.is_file(), f"Previous fingerprint missing: {PREVIOUS_FINGERPRINT}")
    saved = json.loads(PREVIOUS_FINGERPRINT.read_text(encoding="utf-8"))
    for key in [
        "absolute_input_path",
        "input_sha256",
        "analysis_dataframe_hash_pandas_sha256",
        "row_signature_sha256",
        "N_rows",
        "N_subjects",
        "N_HC_subjects",
        "N_SZ_subjects",
        "centering_constants",
        "analysis_row_means",
    ]:
        require(calculated[key] == saved[key], f"Previous analysis-data fingerprint mismatch for {key}")
    require(len(analysis_dat) == 1280 and analysis_dat["Subject"].nunique() == 64, "Locked analysis data shape changed")
    fingerprint = dict(saved)
    fingerprint.update(
        {
            "reuse_status": "EXACT_ANALYSIS_DAT_REUSED_FROM_PREVIOUS_AUDIT_CODE_AND_FINGERPRINT_VERIFIED",
            "previous_fingerprint_path": str(PREVIOUS_FINGERPRINT.resolve()),
            "previous_fingerprint_sha256": sha256_file(PREVIOUS_FINGERPRINT),
            "previous_analysis_script": str(PREVIOUS_SCRIPT.resolve()),
            "previous_analysis_script_sha256": sha256_file(PREVIOUS_SCRIPT),
            "covariates_recentered_in_this_audit": False,
        }
    )
    return analysis_dat, fingerprint, previous


def model_row_ids(model) -> list[int]:
    return [int(value) for value in list(model.data.row_labels)]


def threeway_terms(model) -> list[str]:
    design_info = getattr(model.data, "design_info", None)
    if design_info is None:
        design_info = getattr(model.data.orig_exog, "design_info", None)
    require(design_info is not None, "Patsy DesignInfo unavailable")
    target_names = [
        name for name in design_info.term_names
        if name.count(":") == 2 and all(token in name.lower() for token in ["group", "stage", "bin"])
    ]
    require(len(target_names) == 1, f"Three-way Patsy term not unique: {target_names}")
    target_slice = design_info.term_name_slices[target_names[0]]
    names = list(model.exog_names[target_slice])
    require(len(names) == 9, f"Expected 9 three-way columns, got {names}")
    return names


def quadratic(beta: np.ndarray, covariance: np.ndarray) -> float:
    beta = np.asarray(beta, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    require(covariance.shape == (len(beta), len(beta)), "Quadratic covariance shape mismatch")
    require(np.isfinite(beta).all() and np.isfinite(covariance).all(), "Quadratic input non-finite")
    return float(beta @ np.linalg.pinv(covariance, hermitian=True) @ beta)


def fit_formula_gee(formula: str, data: pd.DataFrame, cov_type: str, label: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.gee(
            formula,
            groups="Subject",
            data=data,
            family=sm.families.Gaussian(),
            cov_struct=Exchangeable(),
        )
        result = model.fit(maxiter=500, cov_type=cov_type)
    warning_text = " | ".join(str(item.message) for item in caught)
    require(model_row_ids(result.model) == data["row_id"].tolist(), f"{label} changed the locked row index")
    require(bool(result.converged), f"{label} did not converge")
    return result, warning_text


def submatrix(covariance: np.ndarray, indices: list[int]) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    return covariance[np.ix_(indices, indices)]


def matrix_frame(matrix: np.ndarray, terms: list[str], short_terms: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, columns=short_terms)
    frame.insert(0, "term", terms)
    frame.insert(1, "short_term", short_terms)
    return frame


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    se = np.sqrt(np.diag(matrix))
    return matrix / np.outer(se, se)


def covariance_summary_row(name: str, matrix: np.ndarray, beta: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    variances = np.diag(matrix)
    ses = np.sqrt(variances)
    sign, logdet = np.linalg.slogdet(matrix)
    weights = np.maximum(eigenvalues, 0) / np.maximum(np.sum(np.maximum(eigenvalues, 0)), np.finfo(float).tiny)
    positive = weights > 0
    effective_rank = float(np.exp(-np.sum(weights[positive] * np.log(weights[positive]))))
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    statistic = quadratic(beta, matrix)
    return {
        "covariance": name,
        "wald_chi2": statistic,
        "df": len(beta),
        "chi2_reference_p": float(chi2.sf(statistic, len(beta))),
        "diagonal_variances": " | ".join(f"{value:.17g}" for value in variances),
        "standard_errors": " | ".join(f"{value:.17g}" for value in ses),
        "min_diagonal_variance": float(variances.min()),
        "max_diagonal_variance": float(variances.max()),
        "mean_diagonal_variance": float(variances.mean()),
        "min_SE": float(ses.min()),
        "max_SE": float(ses.max()),
        "mean_SE": float(ses.mean()),
        "mean_absolute_off_diagonal_covariance": float(np.mean(np.abs(matrix[mask]))),
        "eigenvalues_ascending": " | ".join(f"{value:.17g}" for value in eigenvalues),
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(np.linalg.cond(matrix)),
        "determinant_sign": float(sign),
        "log_absolute_determinant": float(logdet),
        "determinant": float(np.linalg.det(matrix)),
        "numeric_rank": int(np.linalg.matrix_rank(matrix)),
        "effective_rank_entropy": effective_rank,
    }


def generalized_geometry(v_robust: np.ndarray, v_naive: np.ndarray, beta: np.ndarray, short_terms: list[str]):
    eigenvalues, eigenvectors = linalg.eigh(v_robust, v_naive, check_finite=True)
    projections = eigenvectors.T @ beta
    robust_contributions = projections**2 / eigenvalues
    naive_contributions = projections**2
    require(abs(float(robust_contributions.sum()) - quadratic(beta, v_robust)) < 1e-8, "Generalized robust decomposition failed")
    require(abs(float(naive_contributions.sum()) - quadratic(beta, v_naive)) < 1e-8, "Generalized naive decomposition failed")
    rows = []
    for index in range(len(eigenvalues)):
        row = {
            "component": index + 1,
            "generalized_eigenvalue_robust_vs_naive": float(eigenvalues[index]),
            "beta_projection": float(projections[index]),
            "robust_Wald_contribution": float(robust_contributions[index]),
            "naive_Wald_contribution": float(naive_contributions[index]),
            "relative_contribution_to_robust_Wald": float(robust_contributions[index] / robust_contributions.sum()),
        }
        for term, loading in zip(short_terms, eigenvectors[:, index], strict=True):
            row[f"loading_{term}"] = float(loading)
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, eigenvalues, eigenvectors


def fit_array_gee(y: np.ndarray, x: np.ndarray, groups: np.ndarray, start_params: np.ndarray | None = None, maxiter: int = 200):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = GEE(
            np.asarray(y, dtype=float),
            np.asarray(x, dtype=float),
            groups=np.asarray(groups),
            family=sm.families.Gaussian(),
            cov_struct=Exchangeable(),
        )
        result = model.fit(maxiter=maxiter, cov_type="robust", start_params=start_params)
    return result, " | ".join(str(item.message) for item in caught)


def leave_one_subject_out(
    y: np.ndarray, x: np.ndarray, groups: np.ndarray, target: list[int],
    subject_labels: list[str], subject_groups: dict[str, str], observed: float, start: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for cluster_index, subject in enumerate(subject_labels):
        keep = groups != cluster_index
        try:
            result, warning_text = fit_array_gee(y[keep], x[keep], groups[keep], start_params=start, maxiter=300)
            beta = np.asarray(result.params, dtype=float)[target]
            covariance = submatrix(result.cov_params(), target)
            statistic = quadratic(beta, covariance)
            p_value = float(chi2.sf(statistic, 9))
            converged = bool(result.converged)
            status = "OK" if converged else "NONCONVERGED"
            error = ""
        except Exception as exc:
            beta = np.full(9, np.nan)
            statistic = np.nan
            p_value = np.nan
            converged = False
            status = "ERROR"
            warning_text = ""
            error = repr(exc)
        row = {
            "subject_removed": subject,
            "group": subject_groups[subject],
            "N_subjects": 63,
            "N_rows": int(keep.sum()),
            "chi2": statistic,
            "df": 9,
            "p": p_value,
            "delta_chi2_from_full": statistic - observed if np.isfinite(statistic) else np.nan,
            "absolute_delta_chi2": abs(statistic - observed) if np.isfinite(statistic) else np.nan,
            "nine_beta_vector": " | ".join(f"{value:.17g}" for value in beta),
            "converged": converged,
            "status": status,
            "warnings": warning_text,
            "error": error,
        }
        for bin_value, value in zip(range(2, 11), beta, strict=True):
            row[f"beta_Bin{bin_value}"] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    require(frame["converged"].all(), f"LOSO GEE failure(s): {frame.loc[~frame['converged'], ['subject_removed', 'status', 'error']].to_dict('records')}")
    return frame


def cluster_score_influence(result, target: list[int], subject_labels: list[str]) -> pd.DataFrame:
    """Mirror statsmodels GEE._covmat cluster scores and verify their sum."""
    model = result.model
    rows = []
    meat_sum = np.zeros((model.exog.shape[1], model.exog.shape[1]), dtype=float)
    for index in range(model.num_group):
        expval, lpr = model.cached_means[index]
        resid = model.endog_li[index] - expval
        dmat = model.mean_deriv(model.exog_li[index], lpr)
        sdev = np.sqrt(model.family.variance(expval))
        solved = model.cov_struct.covariance_matrix_solve(expval, index, sdev, (dmat, resid))
        require(solved is not None, f"GEE covariance solve failed for cluster {index}")
        _, vinv_resid = tuple(solved)
        score = np.dot(dmat.T, vinv_resid)
        meat = np.outer(score, score)
        meat_sum += meat
        target_score = score[target]
        rows.append(
            {
                "subject": subject_labels[index],
                "cluster_score_norm_full": float(np.linalg.norm(score)),
                "cluster_score_norm_threeway_coordinates": float(np.linalg.norm(target_score)),
                "cluster_meat_contribution_frobenius_norm": float(np.linalg.norm(meat, ord="fro")),
            }
        )
    _, _, official_center = model._covmat()
    require(np.max(np.abs(meat_sum - official_center)) < 1e-10, "Cluster meat reconstruction does not match statsmodels GEE._covmat")
    return pd.DataFrame(rows)


def init_bootstrap_worker(x, groups, mu0, resid, target, start):
    global _BOOT_X, _BOOT_GROUPS, _BOOT_MU0, _BOOT_RESID, _BOOT_TARGET, _BOOT_START
    _BOOT_X = x
    _BOOT_GROUPS = groups
    _BOOT_MU0 = mu0
    _BOOT_RESID = resid
    _BOOT_TARGET = target
    _BOOT_START = start


def bootstrap_worker(task):
    bootstrap_id, cluster_weights = task
    y_star = _BOOT_MU0 + _BOOT_RESID * cluster_weights[_BOOT_GROUPS]
    last_error = ""
    all_warnings = ""
    for attempt, (start, maxiter) in enumerate([(_BOOT_START, 200), (None, 500)], start=1):
        try:
            result, warning_text = fit_array_gee(y_star, _BOOT_X, _BOOT_GROUPS, start_params=start, maxiter=maxiter)
            all_warnings = warning_text
            if bool(result.converged):
                beta = np.asarray(result.params, dtype=float)[_BOOT_TARGET]
                covariance = submatrix(result.cov_params(), _BOOT_TARGET)
                statistic = quadratic(beta, covariance)
                return {
                    "bootstrap_id": bootstrap_id,
                    "wald_chi2": statistic,
                    "converged": True,
                    "status": "OK" if attempt == 1 else "OK_RETRY",
                    "warnings": warning_text,
                    "error": "",
                }
            last_error = "GEE returned converged=False"
        except Exception as exc:
            last_error = repr(exc)
    return {
        "bootstrap_id": bootstrap_id,
        "wald_chi2": np.nan,
        "converged": False,
        "status": "FAILED",
        "warnings": all_warnings,
        "error": last_error,
    }


def run_bootstrap(
    x: np.ndarray, groups: np.ndarray, mu0: np.ndarray, resid: np.ndarray,
    target: list[int], start: np.ndarray,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    cluster_weights = rng.choice(np.array([-1, 1], dtype=np.int8), size=(BOOTSTRAP_B, len(np.unique(groups))))
    tasks = ((index + 1, cluster_weights[index]) for index in range(BOOTSTRAP_B))
    rows = []
    progress_path = AUDIT_DIR / "bootstrap_progress.json"
    write_json({"completed": 0, "total": BOOTSTRAP_B, "status": "RUNNING"}, progress_path)
    context = mp.get_context("spawn")
    with context.Pool(
        processes=N_WORKERS,
        initializer=init_bootstrap_worker,
        initargs=(x, groups, mu0, resid, target, start),
        maxtasksperchild=1000,
    ) as pool:
        for row in pool.imap_unordered(bootstrap_worker, tasks, chunksize=10):
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == BOOTSTRAP_B:
                write_json(
                    {
                        "completed": len(rows),
                        "total": BOOTSTRAP_B,
                        "status": "RUNNING" if len(rows) < BOOTSTRAP_B else "COMPLETED",
                        "failures_so_far": int(sum(not item["converged"] for item in rows)),
                    },
                    progress_path,
                )
    return pd.DataFrame(rows).sort_values("bootstrap_id").reset_index(drop=True)


def bootstrap_figure(statistics: np.ndarray, observed: float, q95: float) -> None:
    # Pillow is used because the locked CPython 3.12 statistics runtime does
    # not contain matplotlib.  The plot is a direct rendering of the saved
    # bootstrap statistics and is exported at 300 dpi to both PNG and PDF.
    width, height = 2160, 1380
    left, right, top, bottom = 220, 90, 170, 230
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_paths = [Path(os.environ.get("STRIATAL_FONT_DIR", ".")) / Path("arial.ttf"), Path(os.environ.get("STRIATAL_FONT_DIR", ".")) / Path("calibri.ttf")]
    bold_paths = [Path(os.environ.get("STRIATAL_FONT_DIR", ".")) / Path("arialbd.ttf"), Path(os.environ.get("STRIATAL_FONT_DIR", ".")) / Path("calibrib.ttf")]
    font_path = next((path for path in font_paths if path.is_file()), None)
    bold_path = next((path for path in bold_paths if path.is_file()), font_path)
    regular = ImageFont.truetype(str(font_path), 38) if font_path else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 32) if font_path else ImageFont.load_default()
    title_font = ImageFont.truetype(str(bold_path), 48) if bold_path else regular

    counts, edges = np.histogram(statistics, bins=55)
    x_max = float(max(edges[-1], observed, q95) * 1.04)
    x_min = 0.0
    y_max = int(max(counts) * 1.12) or 1

    def xpix(value: float) -> int:
        return int(left + (value - x_min) / (x_max - x_min) * plot_w)

    def ypix(value: float) -> int:
        return int(top + plot_h - value / y_max * plot_h)

    draw.line((left, top, left, top + plot_h), fill="#333333", width=3)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#333333", width=3)
    for count, lo, hi in zip(counts, edges[:-1], edges[1:], strict=True):
        if count == 0:
            continue
        x0, x1 = xpix(float(lo)), xpix(float(hi))
        y0 = ypix(float(count))
        draw.rectangle((x0 + 1, y0, max(x0 + 2, x1 - 1), top + plot_h - 1), fill="#6688AA")
    for fraction in np.linspace(0, 1, 6):
        value = x_min + fraction * (x_max - x_min)
        x = xpix(value)
        draw.line((x, top + plot_h, x, top + plot_h + 12), fill="#333333", width=2)
        label = f"{value:.0f}"
        box = draw.textbbox((0, 0), label, font=small)
        draw.text((x - (box[2] - box[0]) / 2, top + plot_h + 22), label, fill="#222222", font=small)
    for fraction in np.linspace(0, 1, 5):
        value = int(round(fraction * y_max))
        y_tick = ypix(value)
        draw.line((left - 12, y_tick, left, y_tick), fill="#333333", width=2)
        label = str(value)
        box = draw.textbbox((0, 0), label, font=small)
        draw.text((left - 25 - (box[2] - box[0]), y_tick - (box[3] - box[1]) / 2), label, fill="#222222", font=small)

    x_q95 = xpix(q95)
    for y_dash in range(top, top + plot_h, 28):
        draw.line((x_q95, y_dash, x_q95, min(y_dash + 16, top + plot_h)), fill="#D18F00", width=6)
    x_observed = xpix(observed)
    draw.line((x_observed, top, x_observed, top + plot_h), fill="#A43131", width=7)

    title = "Subject-level Rademacher wild cluster bootstrap null"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 55), title, fill="#111111", font=title_font)
    xlabel = "9-df robust Wald statistic under H0"
    x_box = draw.textbbox((0, 0), xlabel, font=regular)
    draw.text(((width - (x_box[2] - x_box[0])) / 2, height - 95), xlabel, fill="#222222", font=regular)
    draw.text((28, 58), "Bootstrap replicates", fill="#222222", font=regular)
    legend_y = top + 35
    draw.line((left + 40, legend_y, left + 115, legend_y), fill="#A43131", width=7)
    draw.text((left + 135, legend_y - 22), f"Observed Wald = {observed:.2f}", fill="#222222", font=small)
    draw.line((left + 580, legend_y, left + 655, legend_y), fill="#D18F00", width=6)
    draw.text((left + 675, legend_y - 22), f"Null 95th percentile = {q95:.2f}", fill="#222222", font=small)

    image.save(AUDIT_DIR / "Figure_wild_cluster_bootstrap_null.png", dpi=(300, 300), optimize=True)
    image.save(AUDIT_DIR / "Figure_wild_cluster_bootstrap_null.pdf", "PDF", resolution=300.0)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    require(BOOTSTRAP_B >= 1, "Bootstrap replicate count must be positive")
    analysis_dat, fingerprint, previous = load_locked_analysis_data()
    write_json(fingerprint, AUDIT_DIR / "analysis_dataset_fingerprint.json")

    robust, robust_warnings = fit_formula_gee(FULL_FORMULA, analysis_dat, "robust", "CANONICAL_ROBUST")
    parameter_names = list(robust.params.index)
    terms = threeway_terms(robust.model)
    target = [parameter_names.index(term) for term in terms]
    short_terms = [f"Bin{value}" for value in range(2, 11)]
    beta_full = robust.params.loc[terms].to_numpy(dtype=float)
    v_robust = submatrix(robust.cov_params(), target)
    v_naive = submatrix(robust.cov_naive, target)
    t_robust = quadratic(beta_full, v_robust)
    p_robust = float(chi2.sf(t_robust, 9))
    t_naive = quadratic(beta_full, v_naive)
    p_naive = float(chi2.sf(t_naive, 9))
    canonical_reproduced = abs(t_robust - EXPECTED_CHI2) < STAT_TOL and abs(p_robust - EXPECTED_P) < P_TOL
    require(canonical_reproduced, f"STOP: canonical robust GEE mismatch: chi2={t_robust}, p={p_robust}")

    # Confirm that the installed statsmodels API formally advertises bias_reduced.
    fit_doc = GEE.fit.__doc__ or ""
    fit_source = inspect.getsource(GEE.fit)
    bias_api_supported = "bias_reduced" in fit_doc and "cov_type == \"bias_reduced\"" in fit_source
    bias_available = False
    bias_error = ""
    bias_warnings = ""
    bias_result = None
    if bias_api_supported:
        try:
            bias_result, bias_warnings = fit_formula_gee(FULL_FORMULA, analysis_dat, "bias_reduced", "BIAS_REDUCED")
            beta_bias = bias_result.params.loc[terms].to_numpy(dtype=float)
            v_bias = submatrix(bias_result.cov_params(), target)
            t_bias = quadratic(beta_bias, v_bias)
            p_bias = float(chi2.sf(t_bias, 9))
            bias_beta_difference = float(np.max(np.abs(beta_bias - beta_full)))
            bias_available = True
        except Exception as exc:
            bias_error = repr(exc)
    if not bias_available:
        beta_bias = np.full(9, np.nan)
        v_bias = np.full((9, 9), np.nan)
        t_bias = np.nan
        p_bias = np.nan
        bias_beta_difference = np.nan

    write_tsv(matrix_frame(v_naive, terms, short_terms), AUDIT_DIR / "threeway_covariance_matrix_naive.tsv")
    write_tsv(matrix_frame(v_robust, terms, short_terms), AUDIT_DIR / "threeway_covariance_matrix_robust.tsv")
    write_tsv(matrix_frame(v_bias, terms, short_terms), AUDIT_DIR / "threeway_covariance_matrix_bias_reduced.tsv")
    write_tsv(matrix_frame(correlation_matrix(v_naive), terms, short_terms), AUDIT_DIR / "threeway_correlation_matrix_naive.tsv")
    write_tsv(matrix_frame(correlation_matrix(v_robust), terms, short_terms), AUDIT_DIR / "threeway_correlation_matrix_robust.tsv")
    if bias_available:
        bias_correlation = correlation_matrix(v_bias)
    else:
        bias_correlation = v_bias.copy()
    write_tsv(matrix_frame(bias_correlation, terms, short_terms), AUDIT_DIR / "threeway_correlation_matrix_bias_reduced.tsv")

    generalized, generalized_eigenvalues, generalized_eigenvectors = generalized_geometry(v_robust, v_naive, beta_full, short_terms)
    write_tsv(generalized, AUDIT_DIR / "generalized_covariance_eigenanalysis.tsv")
    dominant = generalized.loc[generalized["relative_contribution_to_robust_Wald"].idxmax()]

    y = analysis_dat["FC"].to_numpy(dtype=float)
    x = np.asarray(robust.model.exog, dtype=float)
    group_codes, subject_uniques = pd.factorize(analysis_dat["Subject"], sort=False)
    groups = np.asarray(group_codes, dtype=int)
    subject_labels = [str(item) for item in subject_uniques]
    require(len(subject_labels) == 64 and np.bincount(groups).tolist() == [20] * 64, "Cluster coding changed")
    subject_groups = {
        str(row.Subject): str(row.Group)
        for row in analysis_dat.drop_duplicates("Subject")[["Subject", "Group"]].itertuples(index=False)
    }
    direct_observed, direct_warnings = fit_array_gee(y, x, groups, start_params=np.asarray(robust.params, dtype=float), maxiter=300)
    direct_beta = np.asarray(direct_observed.params, dtype=float)[target]
    direct_v = submatrix(direct_observed.cov_params(), target)
    direct_t = quadratic(direct_beta, direct_v)
    require(abs(direct_t - t_robust) < 1e-8, "Direct-array GEE does not reproduce formula GEE")

    loso = leave_one_subject_out(
        y, x, groups, target, subject_labels, subject_groups, t_robust, np.asarray(robust.params, dtype=float)
    )
    write_tsv(loso, AUDIT_DIR / "leave_one_subject_out_GEE.tsv")
    beta_columns = [f"beta_Bin{value}" for value in range(2, 11)]
    beta_minus = loso[beta_columns].to_numpy(dtype=float)
    beta_minus_mean = beta_minus.mean(axis=0)
    centered = beta_minus - beta_minus_mean
    g = len(beta_minus)
    v_jackknife = ((g - 1) / g) * centered.T @ centered
    t_jackknife = quadratic(beta_full, v_jackknife)
    p_jackknife = float(chi2.sf(t_jackknife, 9))
    write_tsv(matrix_frame(v_jackknife, terms, short_terms), AUDIT_DIR / "jackknife_covariance.tsv")
    write_tsv(matrix_frame(correlation_matrix(v_jackknife), terms, short_terms), AUDIT_DIR / "jackknife_correlation.tsv")

    score_influence = cluster_score_influence(robust, target, subject_labels)
    score_influence = score_influence.merge(
        loso[["subject_removed", "group", "delta_chi2_from_full", "absolute_delta_chi2"]],
        left_on="subject", right_on="subject_removed", how="left", validate="one_to_one",
    ).drop(columns="subject_removed")
    score_spearman = spearmanr(score_influence["cluster_meat_contribution_frobenius_norm"], score_influence["absolute_delta_chi2"])
    score_pearson = pearsonr(score_influence["cluster_meat_contribution_frobenius_norm"], score_influence["absolute_delta_chi2"])
    score_influence["rank_meat_norm_desc"] = score_influence["cluster_meat_contribution_frobenius_norm"].rank(method="min", ascending=False).astype(int)
    score_influence["rank_abs_delta_chi2_desc"] = score_influence["absolute_delta_chi2"].rank(method="min", ascending=False).astype(int)
    write_tsv(score_influence.sort_values("rank_abs_delta_chi2_desc"), AUDIT_DIR / "cluster_score_influence.tsv")

    reduced, reduced_warnings = fit_formula_gee(REDUCED_FORMULA, analysis_dat, "robust", "REDUCED_NULL")
    mu0 = np.asarray(reduced.fittedvalues, dtype=float)
    residual = y - mu0
    within_sequences = []
    for subject in subject_labels:
        sub = analysis_dat.loc[analysis_dat["Subject"].astype(str) == subject, ["Bin", "Stage"]]
        within_sequences.append(tuple((int(row.Bin), str(row.Stage)) for row in sub.itertuples(index=False)))
    require(len(set(within_sequences)) == 1, "Within-subject residual ordering is not identical")
    within_order = " | ".join(f"Bin{bin_value}:{stage}" for bin_value, stage in within_sequences[0])

    bootstrap = run_bootstrap(x, groups, mu0, residual, target, np.asarray(robust.params, dtype=float))
    write_tsv(bootstrap, AUDIT_DIR / "wild_cluster_bootstrap_statistics.tsv")
    failures = int((~bootstrap["converged"]).sum())
    failure_rate = failures / BOOTSTRAP_B
    require(failure_rate <= 0.01, f"STOP: bootstrap failure rate {failure_rate:.3%} exceeds 1%")
    successful_stats = bootstrap.loc[bootstrap["converged"], "wald_chi2"].to_numpy(dtype=float)
    require(len(successful_stats) > 0, "No successful bootstrap statistics")
    exceedances = int(np.sum(successful_stats >= t_robust))
    if failures == 0:
        p_boot = (1 + exceedances) / (BOOTSTRAP_B + 1)
    else:
        p_boot = (1 + exceedances) / (len(successful_stats) + 1)
    quantile_probs = [0.025, 0.05, 0.90, 0.95, 0.975, 0.99]
    quantiles = np.quantile(successful_stats, quantile_probs)
    bootstrap_summary = {
        "B_requested": BOOTSTRAP_B,
        "B_successful": int(len(successful_stats)),
        "failures": failures,
        "failure_rate": failure_rate,
        "seed": SEED,
        "weights": "Rademacher {-1,+1}, one weight per Subject cluster",
        "observed_chi2": t_robust,
        "exceedances_at_or_above_observed": exceedances,
        "empirical_p_plus_one": p_boot,
        "mean": float(np.mean(successful_stats)),
        "SD": float(np.std(successful_stats, ddof=1)),
        "median": float(np.median(successful_stats)),
        "percentile_2_5": float(quantiles[0]),
        "percentile_5": float(quantiles[1]),
        "percentile_90": float(quantiles[2]),
        "percentile_95": float(quantiles[3]),
        "percentile_97_5": float(quantiles[4]),
        "percentile_99": float(quantiles[5]),
        "max": float(np.max(successful_stats)),
        "observed_percentile_rank_percent": float(100 * np.mean(successful_stats <= t_robust)),
        "denominator_note": "(B+1) because all fits succeeded" if failures == 0 else "successful fits + 1; failures retained and validation classified inconclusive",
    }
    write_json(bootstrap_summary, AUDIT_DIR / "wild_cluster_bootstrap_summary.json")
    bootstrap_figure(successful_stats, t_robust, bootstrap_summary["percentile_95"])

    summary_rows = [
        covariance_summary_row("GEE_naive", v_naive, beta_full),
        covariance_summary_row("GEE_robust", v_robust, beta_full),
    ]
    if bias_available:
        summary_rows.append(covariance_summary_row("GEE_bias_reduced", v_bias, beta_full))
    else:
        summary_rows.append({"covariance": "GEE_bias_reduced", "wald_chi2": np.nan, "df": 9, "chi2_reference_p": np.nan})
    summary_rows.append(covariance_summary_row("Delete_1_cluster_jackknife", v_jackknife, beta_full))
    covariance_summary = pd.DataFrame(summary_rows)
    write_tsv(covariance_summary, AUDIT_DIR / "threeway_covariance_summary.tsv")

    method_comparison = pd.DataFrame(
        [
            {
                "method": "GEE naive/model-based",
                "cov_type": "naive",
                "chi2": t_naive,
                "df": 9,
                "p_chi2_reference": p_naive,
                "beta_vector": " | ".join(f"{value:.17g}" for value in beta_full),
                "max_abs_beta_difference_from_robust_fit": 0.0,
                "status": "DIAGNOSTIC_ONLY",
            },
            {
                "method": "GEE robust sandwich",
                "cov_type": "robust",
                "chi2": t_robust,
                "df": 9,
                "p_chi2_reference": p_robust,
                "beta_vector": " | ".join(f"{value:.17g}" for value in beta_full),
                "max_abs_beta_difference_from_robust_fit": 0.0,
                "status": "CANONICAL_REPRODUCED",
            },
            {
                "method": "GEE bias-reduced sandwich",
                "cov_type": "bias_reduced" if bias_available else "not_available",
                "chi2": t_bias,
                "df": 9,
                "p_chi2_reference": p_bias,
                "beta_vector": " | ".join(f"{value:.17g}" for value in beta_bias) if bias_available else "NA",
                "max_abs_beta_difference_from_robust_fit": bias_beta_difference,
                "status": "AVAILABLE" if bias_available else "BIAS_REDUCED_GEE_NOT_AVAILABLE",
            },
            {
                "method": "Delete-1-cluster jackknife covariance",
                "cov_type": "jackknife",
                "chi2": t_jackknife,
                "df": 9,
                "p_chi2_reference": p_jackknife,
                "beta_vector": " | ".join(f"{value:.17g}" for value in beta_full),
                "max_abs_beta_difference_from_robust_fit": 0.0,
                "status": "DIAGNOSTIC_ONLY",
            },
        ]
    )
    write_tsv(method_comparison, AUDIT_DIR / "gee_covariance_method_comparison.tsv")

    loso_summary = {
        "min_chi2": float(loso["chi2"].min()),
        "max_chi2": float(loso["chi2"].max()),
        "median_chi2": float(loso["chi2"].median()),
        "min_p": float(loso["p"].min()),
        "max_p": float(loso["p"].max()),
        "median_p": float(loso["p"].median()),
        "N_p_lt_0_05": int((loso["p"] < 0.05).sum()),
        "N_p_lt_0_01": int((loso["p"] < 0.01).sum()),
        "N_p_lt_0_005": int((loso["p"] < 0.005).sum()),
        "largest_absolute_delta_chi2": float(loso["absolute_delta_chi2"].max()),
    }
    top5 = loso.nlargest(5, "absolute_delta_chi2")[
        ["subject_removed", "group", "chi2", "p", "delta_chi2_from_full", "absolute_delta_chi2"]
    ].copy()

    diag_ratio = np.diag(v_robust) / np.diag(v_naive)
    corr_difference = correlation_matrix(v_robust) - correlation_matrix(v_naive)
    corr_offdiag = corr_difference[~np.eye(9, dtype=bool)]
    number_small_generalized = int(np.sum(generalized_eigenvalues < 0.5))
    contribution_sorted = generalized.sort_values("relative_contribution_to_robust_Wald", ascending=False)
    cumulative = contribution_sorted["relative_contribution_to_robust_Wald"].cumsum().to_numpy()
    components_for_80 = int(np.searchsorted(cumulative, 0.80) + 1)
    dominant_component = int(dominant["component"])
    dominant_lambda = float(dominant["generalized_eigenvalue_robust_vs_naive"])
    dominant_relative = float(dominant["relative_contribution_to_robust_Wald"])
    geometry_explanation = (
        f"Robust/naive diagonal variance ratios range {diag_ratio.min():.4g}-{diag_ratio.max():.4g} "
        f"(mean {diag_ratio.mean():.4g}); all nine are {'below' if np.all(diag_ratio < 1) else 'not uniformly below'} 1. "
        f"The maximum absolute off-diagonal correlation change is {np.max(np.abs(corr_offdiag)):.4g}. "
        f"Generalized eigenvalues span {generalized_eigenvalues.min():.4g}-{generalized_eigenvalues.max():.4g}, "
        f"with {number_small_generalized}/9 below 0.5. Component {dominant_component} (lambda={dominant_lambda:.4g}) "
        f"contributes {100 * dominant_relative:.2f}% of the robust Wald, and {components_for_80} component(s) account for 80% of it."
    )

    loso_stable = bool((loso["p"] < 0.05).all())
    bias_significant = bool(bias_available and p_bias < 0.05)
    bootstrap_significant = bool(p_boot < 0.05)
    conflict = bool(bias_available and (bias_significant != bootstrap_significant))
    if failures > 0 or not bias_available:
        finite_classification = "FINITE_SAMPLE_VALIDATION_INCONCLUSIVE"
    elif conflict:
        finite_classification = "FINITE_SAMPLE_VALIDATION_INCONCLUSIVE"
    elif bias_significant and bootstrap_significant and loso_stable:
        finite_classification = "ROBUST_GEE_SIGNAL_SURVIVES_FINITE_SAMPLE_CHECKS"
    elif (not bias_significant) or (not bootstrap_significant):
        finite_classification = "ASYMPTOTIC_ROBUST_GEE_NOT_STABLE"
    else:
        finite_classification = "FINITE_SAMPLE_VALIDATION_INCONCLUSIVE"

    if finite_classification == "ROBUST_GEE_SIGNAL_SURVIVES_FINITE_SAMPLE_CHECKS":
        manuscript_implication = "The categorical interaction may be treated as supportive sensitivity evidence from robust GEE finite-sample checks, but it remains framework-sensitive because MixedLM and model-based covariance do not support it."
    elif finite_classification == "ASYMPTOTIC_ROBUST_GEE_NOT_STABLE":
        manuscript_implication = "The asymptotic robust-GEE p-value should not be treated as robust confirmatory evidence; at most, the result is framework-sensitive exploratory evidence and should be excluded from confirmatory inferential conclusions."
    else:
        manuscript_implication = "Finite-sample validation is inconclusive; the categorical interaction should be described only as framework-sensitive exploratory evidence, not confirmatory evidence."

    covariance_display = covariance_summary[
        ["covariance", "wald_chi2", "chi2_reference_p", "min_eigenvalue", "max_eigenvalue", "condition_number", "effective_rank_entropy"]
    ].copy()
    generalized_display = generalized[
        ["component", "generalized_eigenvalue_robust_vs_naive", "beta_projection", "robust_Wald_contribution", "relative_contribution_to_robust_Wald"]
    ].copy()
    report = f"""# GEE robust sandwich finite-sample stability audit

## A. Frozen data verification

- N = {fingerprint['N_subjects']} subjects: {fingerprint['N_HC_subjects']} HC and {fingerprint['N_SZ_subjects']} SZ.
- Rows = {fingerprint['N_rows']}; every subject has 20 rows.
- Absolute input = `{fingerprint['absolute_input_path']}`.
- SHA-256 = `{fingerprint['input_sha256']}`.
- Stages = {fingerprint['stage_counts']}; Bins 1-10 each have 128 rows; missing values = 0.
- `age_c`, `sex_c`, and `meanFD_c` were not independently re-centered. The preceding audit's `load_analysis_data()` was imported, and both its saved dataframe hash and centering constants were verified exactly. Status: **{fingerprint['reuse_status']}**.
- Frozen within-subject row order used for residual vectors: `{within_order}`. One Rademacher weight multiplies all 20 residuals of a subject, so the complete frozen within-subject vector is preserved.

## B. Original canonical result

Mean model: `{FULL_FORMULA}`; Gaussian identity GEE; exchangeable working correlation; cluster = Subject; robust sandwich covariance.

- GEE robust: chi-square(9) = **{fmt(t_robust)}**, p = **{fmt(p_robust)}**.
- Reproduces the frozen canonical statistic and p value: **YES**.
- Exchangeable alpha = {fmt(float(np.asarray(robust.cov_struct.dep_params).squeeze()))}; scale = {fmt(float(robust.scale))}; warnings = `{robust_warnings or 'none'}`.

## C. Model-based comparator

- GEE naive/model-based: chi-square(9) = **{fmt(t_naive)}**, p = **{fmt(p_naive)}**.
- This is a diagnostic comparator, not a replacement for the canonical robust result.

## D. Bias-reduced GEE

- Installed statsmodels version = {statsmodels.__version__}. `GEE.fit` formally documents `cov_type` values `robust`, `naive`, and `bias_reduced`, and the installed source contains the bias-reduced branch and `_bc_covmat` implementation.
- Status = **{'AVAILABLE' if bias_available else 'BIAS_REDUCED_GEE_NOT_AVAILABLE'}**.
- Bias-reduced Wald chi-square(9) = **{fmt(t_bias)}**, p = **{fmt(p_bias)}**.
- Maximum absolute robust-fit versus bias-reduced-fit beta difference = {fmt(bias_beta_difference)}.
- Warnings = `{bias_warnings or 'none'}`; error = `{bias_error or 'none'}`.

## E. Covariance geometry

{markdown_table(covariance_display, ['covariance', 'wald_chi2', 'chi2_reference_p', 'min_eigenvalue', 'max_eigenvalue', 'condition_number', 'effective_rank_entropy'])}

Effective rank is the entropy effective rank of the non-negative covariance eigenvalues; numeric ranks are also in `threeway_covariance_summary.tsv`. Full covariance and correlation matrices are saved separately for naive, robust, bias-reduced, and jackknife estimators.

Quantitative explanation of the model-based and sandwich Wald statistics: {geometry_explanation}

The generalized decomposition below uses eigenvectors normalized so that v-prime V-naive v = 1. In that basis, the robust Wald contribution is `(v-prime beta)^2 / lambda`, and the contributions sum exactly to {fmt(t_robust)}.

{markdown_table(generalized_display, ['component', 'generalized_eigenvalue_robust_vs_naive', 'beta_projection', 'robust_Wald_contribution', 'relative_contribution_to_robust_Wald'])}

Thus the observed increase is attributed to: (A) the measured diagonal variance ratios above; (B) the measured correlation reconfiguration; and (C) the concentration of beta in the reported low-variance generalized directions. Cluster-level influence is evaluated separately below rather than inferred from covariance geometry alone.

## F. Leave-one-subject-out stability

- Chi-square range = {fmt(loso_summary['min_chi2'])} to {fmt(loso_summary['max_chi2'])}; median = {fmt(loso_summary['median_chi2'])}.
- p range = {fmt(loso_summary['min_p'])} to {fmt(loso_summary['max_p'])}; median = {fmt(loso_summary['median_p'])}.
- N/64 with p < .05 = {loso_summary['N_p_lt_0_05']}; p < .01 = {loso_summary['N_p_lt_0_01']}; p < .005 = {loso_summary['N_p_lt_0_005']}.
- Largest absolute delta chi-square = {fmt(loso_summary['largest_absolute_delta_chi2'])}.

Five largest absolute changes:

{markdown_table(top5, ['subject_removed', 'group', 'chi2', 'p', 'delta_chi2_from_full', 'absolute_delta_chi2'])}

The full delete-one-cluster table retains all 64 subjects and all nine beta vectors. These exclusions are influence diagnostics only; none is used to redefine the formal analysis.

Official-internal score reconstruction check: the sum of reconstructed cluster meat contributions matches statsmodels `GEE._covmat()` to <1e-10. Meat-norm versus absolute delete-one chi-square change has Spearman rho = {fmt(float(score_spearman.statistic))}, p = {fmt(float(score_spearman.pvalue))}; Pearson r = {fmt(float(score_pearson.statistic))}, p = {fmt(float(score_pearson.pvalue))}. Full rankings are in `cluster_score_influence.tsv`.

## G. Jackknife covariance

- Delete-one-cluster jackknife Wald using the original full-sample beta: chi-square(9) = **{fmt(t_jackknife)}**.
- Chi-square-reference diagnostic p = **{fmt(p_jackknife)}**.
- This p-value is a sensitivity reference and is not presented as resolving all small-sample degrees-of-freedom issues.

## H. Wild cluster bootstrap under H0

- Reduced null mean model: `{REDUCED_FORMULA}`; only the Group × Stage × Bin term is removed.
- B = {BOOTSTRAP_B}; successful = {len(successful_stats)}; failures = {failures} ({100 * failure_rate:.3f}%); seed = {SEED}.
- Weights: subject-level Rademacher {{-1,+1}}; Group, Stage, Bin, covariates, and complete within-subject residual vectors are unchanged.
- Observed T = {fmt(t_robust)}.
- Empirical plus-one p = **{fmt(p_boot)}**; exceedances = {exceedances}.
- Bootstrap 95th percentile = {fmt(bootstrap_summary['percentile_95'])}; 99th percentile = {fmt(bootstrap_summary['percentile_99'])}.
- Mean = {fmt(bootstrap_summary['mean'])}; SD = {fmt(bootstrap_summary['SD'])}; median = {fmt(bootstrap_summary['median'])}; 2.5th = {fmt(bootstrap_summary['percentile_2_5'])}; 5th = {fmt(bootstrap_summary['percentile_5'])}; 90th = {fmt(bootstrap_summary['percentile_90'])}; 97.5th = {fmt(bootstrap_summary['percentile_97_5'])}; maximum = {fmt(bootstrap_summary['max'])}.
- Observed percentile rank = {fmt(bootstrap_summary['observed_percentile_rank_percent'])}%.

## I. Final classification

- Cross-framework: **INFERENCE_FRAMEWORK_SENSITIVE**.
- Finite-sample robust-GEE validation: **{finite_classification}**.
- Bias-reduced significant at .05: {bias_significant}; bootstrap significant at .05: {bootstrap_significant}; all 64 LOSO fits significant at .05: {loso_stable}; bias/bootstrap conflict: {conflict}.

## J. Manuscript implication

Methodological implication only: {manuscript_implication}

No manuscript file was changed. No bin post-hoc, subfamily analysis, deviation analysis, alternative mean model, or specification search was performed.
"""
    write_text(report, AUDIT_DIR / "FINAL_GEE_FINITE_SAMPLE_AUDIT.md")

    output_names = [
        "02_validate_finite_sample_audit.py",
        "analysis_dataset_fingerprint.json",
        "gee_covariance_method_comparison.tsv",
        "threeway_covariance_matrix_naive.tsv",
        "threeway_covariance_matrix_robust.tsv",
        "threeway_covariance_matrix_bias_reduced.tsv",
        "threeway_correlation_matrix_naive.tsv",
        "threeway_correlation_matrix_robust.tsv",
        "threeway_correlation_matrix_bias_reduced.tsv",
        "threeway_covariance_summary.tsv",
        "generalized_covariance_eigenanalysis.tsv",
        "leave_one_subject_out_GEE.tsv",
        "jackknife_covariance.tsv",
        "jackknife_correlation.tsv",
        "wild_cluster_bootstrap_statistics.tsv",
        "wild_cluster_bootstrap_summary.json",
        "bootstrap_progress.json",
        "cluster_score_influence.tsv",
        "Figure_wild_cluster_bootstrap_null.png",
        "Figure_wild_cluster_bootstrap_null.pdf",
        "FINAL_GEE_FINITE_SAMPLE_AUDIT.md",
    ]
    provenance = {
        "audit": "GEE robust sandwich finite-sample stability audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input": fingerprint["absolute_input_path"],
        "input_sha256": fingerprint["input_sha256"],
        "previous_analysis_script": str(PREVIOUS_SCRIPT.resolve()),
        "previous_analysis_script_sha256": sha256_file(PREVIOUS_SCRIPT),
        "previous_fingerprint": str(PREVIOUS_FINGERPRINT.resolve()),
        "previous_fingerprint_sha256": sha256_file(PREVIOUS_FINGERPRINT),
        "analysis_dataframe_hash_pandas_sha256": fingerprint["analysis_dataframe_hash_pandas_sha256"],
        "row_signature_sha256": fingerprint["row_signature_sha256"],
        "formulas": {"FULL": FULL_FORMULA, "REDUCED_H0": REDUCED_FORMULA},
        "GEE": {
            "family": "Gaussian",
            "link": "identity",
            "working_correlation": "exchangeable",
            "cluster": "Subject",
            "canonical_covariance": "robust sandwich",
            "bias_reduced_api_supported": bias_api_supported,
            "bias_reduced_available": bias_available,
            "GEE_fit_signature": str(inspect.signature(GEE.fit)),
            "GEE_fit_source_sha256": sha256_text(fit_source),
            "GEE_bc_covmat_source_sha256": sha256_text(inspect.getsource(GEE._bc_covmat)),
            "robust_warnings": robust_warnings,
            "bias_reduced_warnings": bias_warnings,
            "reduced_warnings": reduced_warnings,
            "direct_array_warnings": direct_warnings,
        },
        "bootstrap": {
            "B": BOOTSTRAP_B,
            "seed": SEED,
            "weight_distribution": "Rademacher {-1,+1}",
            "weight_level": "Subject cluster; same weight for all 20 observations",
            "workers": N_WORKERS,
            "failure_rate": failure_rate,
        },
        "score_contribution_implementation": "Exact per-cluster expressions from installed statsmodels GEE._covmat; sum verified against official center matrix",
        "canonical_reproduced": canonical_reproduced,
        "cross_framework_classification": "INFERENCE_FRAMEWORK_SENSITIVE",
        "finite_sample_classification": finite_classification,
        "outputs": {name: sha256_file(AUDIT_DIR / name) for name in output_names},
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "patsy": patsy.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "Pillow": PIL.__version__,
        },
        "scope_exclusions": [
            "no manuscript changes",
            "no bin post-hoc",
            "no subfamily analysis",
            "no deviation analysis",
            "no mean-model change",
            "no specification search",
        ],
    }
    write_json(provenance, AUDIT_DIR / "provenance.json")

    bias_summary = next((row for row in summary_rows if row["covariance"] == "GEE_bias_reduced"), None)
    print("=" * 70)
    print("GEE FINITE-SAMPLE STABILITY AUDIT")
    print("=" * 70)
    print()
    print(f"Subjects = {fingerprint['N_subjects']}")
    print(f"Rows = {fingerprint['N_rows']}")
    print()
    print("-" * 70)
    print("ORIGINAL GEE ROBUST")
    print("-" * 70)
    print()
    print(f"chi2(9) = {t_robust:.12f}")
    print(f"p = {p_robust:.12g}")
    print()
    print("-" * 70)
    print("GEE NAIVE")
    print("-" * 70)
    print()
    print(f"chi2(9) = {t_naive:.12f}")
    print(f"p = {p_naive:.12g}")
    print()
    print("-" * 70)
    print("GEE BIAS-REDUCED")
    print("-" * 70)
    print()
    print(f"chi2(9) = {t_bias:.12f}" if bias_available else "chi2(9) = NOT AVAILABLE")
    print(f"p = {p_bias:.12g}" if bias_available else "p = NOT AVAILABLE")
    print()
    print("-" * 70)
    print("COVARIANCE GEOMETRY")
    print("-" * 70)
    print()
    naive_summary = summary_rows[0]
    robust_summary = summary_rows[1]
    print(f"naive min/max eigenvalue = {naive_summary['min_eigenvalue']:.12g} / {naive_summary['max_eigenvalue']:.12g}")
    print(f"robust min/max eigenvalue = {robust_summary['min_eigenvalue']:.12g} / {robust_summary['max_eigenvalue']:.12g}")
    if bias_available and bias_summary is not None:
        print(f"bias-reduced min/max eigenvalue = {bias_summary['min_eigenvalue']:.12g} / {bias_summary['max_eigenvalue']:.12g}")
    else:
        print("bias-reduced min/max eigenvalue = NOT AVAILABLE")
    print()
    print(f"smallest generalized eigenvalue robust vs naive = {generalized_eigenvalues.min():.12g}")
    print(f"largest generalized eigenvalue robust vs naive = {generalized_eigenvalues.max():.12g}")
    print()
    print(
        f"dominant Wald direction: component {dominant_component}, lambda={dominant_lambda:.6g}, "
        f"contribution={100 * dominant_relative:.2f}%"
    )
    print()
    print("-" * 70)
    print("LEAVE-ONE-SUBJECT-OUT")
    print("-" * 70)
    print()
    print(f"chi2 range = {loso_summary['min_chi2']:.12g} to {loso_summary['max_chi2']:.12g}")
    print(f"p range = {loso_summary['min_p']:.12g} to {loso_summary['max_p']:.12g}")
    print()
    print(f"N / 64 with p < .05 = {loso_summary['N_p_lt_0_05']}")
    print(f"N / 64 with p < .01 = {loso_summary['N_p_lt_0_01']}")
    print()
    print(f"largest absolute delta chi2 = {loso_summary['largest_absolute_delta_chi2']:.12g}")
    print()
    print("-" * 70)
    print("JACKKNIFE")
    print("-" * 70)
    print()
    print(f"chi2(9) = {t_jackknife:.12f}")
    print(f"p diagnostic = {p_jackknife:.12g}")
    print()
    print("-" * 70)
    print("NULL WILD CLUSTER BOOTSTRAP")
    print("-" * 70)
    print()
    print(f"B = {BOOTSTRAP_B}")
    print()
    print(f"Observed chi2 = {t_robust:.12f}")
    print()
    print(f"Bootstrap 95th percentile = {bootstrap_summary['percentile_95']:.12g}")
    print(f"Bootstrap 99th percentile = {bootstrap_summary['percentile_99']:.12g}")
    print()
    print(f"Empirical p = {p_boot:.12g}")
    print()
    print("-" * 70)
    print("FINAL CLASSIFICATION")
    print("-" * 70)
    print()
    print("Cross-framework:")
    print("INFERENCE_FRAMEWORK_SENSITIVE")
    print()
    print("Finite-sample GEE:")
    print(finite_classification)
    print()
    print("Primary explanation for robust-vs-naive discrepancy:")
    print(geometry_explanation)
    print()
    print("=" * 70)


if __name__ == "__main__":
    mp.freeze_support()
    main()
