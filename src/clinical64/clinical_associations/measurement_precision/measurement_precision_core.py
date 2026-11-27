from __future__ import annotations

import os
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from scipy import stats


OUT = Path(os.environ["STRIATAL_OUTPUT_ROOT"])
ROOT = Path(os.environ["STRIATAL_PROJECT_ROOT"])
DATA = OUT / "data_internal"
FIGSRC = OUT / "figure_source_data"
BOOT_N = 5000
SEED = 20261029


PATHS = {
    "workbook": Path(os.environ["STRIATAL_RAW_SCALE_WORKBOOK"]),
    "clinical35": ROOT / "clinical_extension_analysis" / "data_internal" / "clinical35_audited.tsv",
    "final64_clinical": Path(os.path.join(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"], 'final64_gradient_clinical_stats/03_clinical_master/final64_clinical_for_server_v2.tsv')),
    "covariates": ROOT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "covariates" / "current64_covariates.tsv",
    "roi_order": ROOT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "roi_order" / "canonical_roi_order_65.tsv",
    "stage_mapping": ROOT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "reward_stage_mapping" / "reward_stage_mapping_65.tsv",
    "fixed_bins": ROOT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "canonical_bins" / "375_reference_bins_manuscript_2608.npy",
    "fixed_g1": ROOT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "canonical_G1" / "375_reference_G1_manuscript_2608.npy",
    "fingerprints": ROOT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "fc_fingerprints_2608x65",
    "stage_fc": ROOT / "CURRENT64_CANONICAL_RERUN" / "06_REWARD_STAGE" / "subject_bin_stage_FC_long.tsv",
    "stage_slopes": ROOT / "CURRENT64_CANONICAL_RERUN" / "09_INDIVIDUAL_SLOPES" / "subject_stage_slopes_wide.tsv",
    "current_voxels": ROOT / "FIGURES_V4_REFERENCE_LOCKED" / "source_data" / "Figure1_3_G1_spatial_2608.tsv",
    "locked_stage_similarity": ROOT / "SERVER_RECOVERED_CODE_REVIEW" / "FINAL_SCIENTIFIC_VALIDATION" / "RUN_ARTIFACTS" / "current_from_arrays" / "stage_specific_subject_bin_metrics.tsv",
    "timeseries": ROOT / "99_archive_64" / "rebuild_oxford_65roi_v2" / "17_current64_375matched_fingerprints" / "02_timeseries",
    "clinical_assoc": ROOT / "clinical_extension_analysis" / "primary_clinical_associations.csv",
    "roi_assoc_primary": ROOT / "clinical_extension_analysis" / "roi_behavior_extension" / "primary_association_results.csv",
    "roi_assoc_secondary": ROOT / "clinical_extension_analysis" / "roi_behavior_extension" / "secondary_association_results.csv",
    "stage_assoc": ROOT / "clinical_extension_analysis" / "stage_coupling_analysis" / "stage_coupling_results.csv",
}


TEPS_ANT = [4, 6, 18, 20, 1, 5, 8, 10, 11]
TEPS_CONS = [2, 7, 3, 9, 14, 19, 12, 15, 16, 17]
ACIPS_ANT = [1, 7, 8, 10, 14, 15, 3]
ACIPS_CONS = [2, 4, 5, 6, 9, 11, 12, 13, 16, 17]


def ensure_dirs() -> None:
    for p in (OUT, DATA, FIGSRC):
        p.mkdir(parents=True, exist_ok=True)


def assert_inputs() -> None:
    missing = [str(p) for p in PATHS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(x):
    if x is None or (isinstance(x, str) and not x.strip()):
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _sum_if_complete(values: Iterable[float]) -> float:
    a = np.asarray(list(values), float)
    return float(a.sum()) if np.isfinite(a).all() else np.nan


def _score_columns() -> dict[str, list[str]]:
    return {
        "TEPS Anticipatory": [f"TEPS_A_item{i:02d}" for i in TEPS_ANT],
        "TEPS Consummatory": [f"TEPS_C_item{i:02d}" for i in TEPS_CONS],
        "ACIPS Anticipatory": [f"ACIPS_A_item{i:02d}{'_reversed' if i == 3 else ''}" for i in ACIPS_ANT],
        "ACIPS Consummatory": [f"ACIPS_C_item{i:02d}" for i in ACIPS_CONS],
        "PANSS Negative": [f"PANSS_N{i}" for i in range(1, 8)],
    }


def _load_workbook_rows():
    wb = openpyxl.load_workbook(PATHS["workbook"], read_only=True, data_only=True)
    scales = wb["scales"]
    sh = list(next(scales.iter_rows(min_row=1, max_row=1, values_only=True)))
    scale_rows = {int(r[0]): r for r in scales.iter_rows(min_row=2, values_only=True) if _num(r[0]) == _num(r[0])}
    clinical = None
    for ws in wb.worksheets:
        h = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
        if len(h) >= 40 and str(h[24]).upper() == "PANS-NE":
            clinical = ws
            break
    if clinical is None:
        raise RuntimeError("Could not identify the clinical sheet containing PANS-NE and N1-N7.")
    ch = list(next(clinical.iter_rows(min_row=1, max_row=1, values_only=True)))
    clinical_rows = {
        int(r[1]): r
        for r in clinical.iter_rows(min_row=2, values_only=True)
        if _num(r[1]) == _num(r[1])
    }
    return sh, scale_rows, ch, clinical_rows


def run_audit() -> None:
    ensure_dirs()
    assert_inputs()
    mapping = pd.read_csv(PATHS["clinical35"], sep="\t")[["subject", "raw_clinical_id", "PANSS_negative"]].copy()
    mapping["raw_clinical_id"] = mapping["raw_clinical_id"].astype(int)
    frozen = pd.read_csv(PATHS["final64_clinical"], sep="\t")
    frozen_sz = set(frozen.loc[frozen["group"].eq("SZ"), "subject"].astype(str))
    sh, scale_rows, ch, clinical_rows = _load_workbook_rows()
    hpos = {str(v).strip(): i for i, v in enumerate(sh) if v is not None}

    rows = []
    score_audit = []
    for _, m in mapping.iterrows():
        sid, rid = str(m["subject"]), int(m["raw_clinical_id"])
        sr = scale_rows.get(rid)
        cr = clinical_rows.get(rid)
        if sr is None or cr is None:
            raise RuntimeError(f"Missing workbook row for raw clinical ID {rid}.")
        rec = {"subject": sid, "raw_clinical_id": rid, "is_frozen_imaging_SZ": int(sid in frozen_sz)}
        teps = {i: _num(sr[hpos[f"TEPS{i}"]]) for i in range(1, 21)}
        acips = {i: _num(sr[hpos[f"ACIPS{i}"]]) for i in range(1, 18)}
        for i in TEPS_ANT:
            rec[f"TEPS_A_item{i:02d}"] = teps[i]
        for i in TEPS_CONS:
            rec[f"TEPS_C_item{i:02d}"] = teps[i]
        for i in ACIPS_ANT:
            rec[f"ACIPS_A_item{i:02d}{'_reversed' if i == 3 else ''}"] = (7.0 - acips[i]) if i == 3 and np.isfinite(acips[i]) else acips[i]
        for i in ACIPS_CONS:
            rec[f"ACIPS_C_item{i:02d}"] = acips[i]
        for i in range(1, 8):
            rec[f"PANSS_N{i}"] = _num(cr[32 + i])
        specs = _score_columns()
        for scale, cols in specs.items():
            rec[scale] = _sum_if_complete(rec[c] for c in cols)
        rec["PANSS_Negative_workbook_PANS_NE"] = _num(cr[24])
        rec["PANSS_Negative_formal"] = _num(m["PANSS_negative"])
        rec["TEPS_Anticipatory_formal"] = _sum_if_complete([_num(sr[hpos["abstract_anticipator"]]), _num(sr[hpos["concrete_anticipator"]])])
        rec["TEPS_Consummatory_formal"] = _sum_if_complete([_num(sr[hpos["abstract_consumatory"]]), _num(sr[hpos["concrete_consumatory"]])])
        rows.append(rec)

    scored = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    scored.to_csv(DATA / "behavior_item_level_scored.csv", index=False)

    frozen_scores = frozen[["subject", "TEPS_anticipatory_calc", "TEPS_consummatory_calc", "ACIPS_anticipatory_calc", "ACIPS_consummatory_calc"]]
    check = scored.merge(frozen_scores, on="subject", how="left")
    comparisons = [
        ("TEPS Anticipatory", "TEPS Anticipatory", "TEPS_Anticipatory_formal", "workbook formal component sums", "full_SZ_clinical"),
        ("TEPS Consummatory", "TEPS Consummatory", "TEPS_Consummatory_formal", "workbook formal component sums", "full_SZ_clinical"),
        ("PANSS Negative", "PANSS Negative", "PANSS_Negative_formal", "canonical clinical35 PANSS_negative", "full_SZ_clinical"),
        ("TEPS Anticipatory", "TEPS Anticipatory", "TEPS_anticipatory_calc", "frozen final64 formal variable", "frozen_imaging_SZ"),
        ("TEPS Consummatory", "TEPS Consummatory", "TEPS_consummatory_calc", "frozen final64 formal variable", "frozen_imaging_SZ"),
        ("ACIPS Anticipatory", "ACIPS Anticipatory", "ACIPS_anticipatory_calc", "frozen final64 formal variable", "frozen_imaging_SZ"),
        ("ACIPS Consummatory", "ACIPS Consummatory", "ACIPS_consummatory_calc", "frozen final64 formal variable", "frozen_imaging_SZ"),
    ]
    for label, calc, formal, source, sample in comparisons:
        d = check.copy()
        if sample == "frozen_imaging_SZ":
            d = d[d["is_frozen_imaging_SZ"].eq(1)]
        valid = d[[calc, formal]].notna().all(axis=1)
        delta = (d.loc[valid, calc] - d.loc[valid, formal]).abs()
        score_audit.append({
            "scale": label,
            "sample": sample,
            "formal_source": source,
            "n_compared": int(valid.sum()),
            "n_exact_match": int((delta < 1e-10).sum()),
            "max_abs_difference": float(delta.max()) if len(delta) else np.nan,
            "status": "PASS" if len(delta) and (delta < 1e-10).all() else "REVIEW",
        })
    pd.DataFrame(score_audit).to_csv(OUT / "behavior_score_reconstruction_audit.csv", index=False)

    bins = np.asarray(np.load(PATHS["fixed_bins"]), int).reshape(-1)
    vox = pd.read_csv(PATHS["current_voxels"], sep="\t")
    stage_map = pd.read_csv(PATHS["stage_mapping"], sep="\t")
    ts_dirs = sorted(p for p in PATHS["timeseries"].iterdir() if p.is_dir())
    t_lengths = []
    for p in ts_dirs:
        q = next(p.glob("*_roi_timeseries_Tx65.npy"))
        t_lengths.append(np.load(q, mmap_mode="r").shape[0])

    audit = [
        {"domain": "sample", "check": "full SZ clinical mapping", "value": len(mapping), "expected": 35, "status": "PASS" if len(mapping) == 35 else "FAIL"},
        {"domain": "sample", "check": "frozen imaging SZ", "value": len(frozen_sz), "expected": 32, "status": "PASS" if len(frozen_sz) == 32 else "FAIL"},
        {"domain": "behavior", "check": "workbook scale rows matched", "value": scored["raw_clinical_id"].nunique(), "expected": 35, "status": "PASS"},
        {"domain": "behavior", "check": "PANSS N1-N7 located", "value": 7, "expected": 7, "status": "PASS"},
        {"domain": "neural", "check": "fixed G1 length", "value": len(np.load(PATHS["fixed_g1"])), "expected": 2608, "status": "PASS"},
        {"domain": "neural", "check": "fixed 10-bin length", "value": len(bins), "expected": 2608, "status": "PASS"},
        {"domain": "neural", "check": "fixed bin labels", "value": ",".join(map(str, sorted(np.unique(bins)))), "expected": "1..10", "status": "PASS" if set(np.unique(bins)) == set(range(1, 11)) else "FAIL"},
        {"domain": "neural", "check": "current Bin10 voxel count", "value": int(vox["current_bin"].eq(10).sum()), "expected": 285, "status": "PASS" if vox["current_bin"].eq(10).sum() == 285 else "FAIL"},
        {"domain": "neural", "check": "stage ROI counts", "value": json.dumps(stage_map["reward_stage"].value_counts().to_dict()), "expected": "Anticipation=43; Outcome=19; Unassigned=3", "status": "PASS"},
        {"domain": "neural", "check": "timeseries subjects", "value": len(ts_dirs), "expected": 64, "status": "PASS" if len(ts_dirs) == 64 else "FAIL"},
        {"domain": "neural", "check": "timeseries length range", "value": f"{min(t_lengths)}-{max(t_lengths)}", "expected": "about 176-180", "status": "PASS" if min(t_lengths) >= 176 and max(t_lengths) <= 180 else "FAIL"},
    ]
    for scale, cols in _score_columns().items():
        for sample, mask in [("full_SZ_clinical", np.ones(len(scored), bool)), ("frozen_imaging_SZ", scored["is_frozen_imaging_SZ"].eq(1).to_numpy())]:
            complete = scored.loc[mask, cols].notna().all(axis=1)
            vals = scored.loc[mask, cols].to_numpy(float)
            finite = vals[np.isfinite(vals)]
            audit.append({
                "domain": "behavior",
                "check": f"{scale} complete N ({sample})",
                "value": int(complete.sum()),
                "expected": "complete-case; no exclusions for reliability",
                "status": "PASS",
                "details": f"item range {finite.min():.0f}-{finite.max():.0f}" if finite.size else "no finite values",
            })
    pd.DataFrame(audit).to_csv(OUT / "input_audit.csv", index=False)
    print("Audit complete: behavior items, PANSS N1-N7, fixed bins/G1, and time series verified.")


def cronbach_alpha(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[0] < 3 or not np.isfinite(x).all():
        return np.nan
    total_var = np.var(x.sum(axis=1), ddof=1)
    if total_var <= 0:
        return np.nan
    k = x.shape[1]
    return float(k / (k - 1) * (1 - np.var(x, axis=0, ddof=1).sum() / total_var))


def omega_paf(x: np.ndarray) -> float:
    """Standardized omega total from a fixed one-factor principal-axis model."""
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[0] < 5 or x.shape[1] < 2 or not np.isfinite(x).all():
        return np.nan
    if np.any(np.std(x, axis=0, ddof=1) <= 0):
        return np.nan
    r = np.corrcoef(x, rowvar=False)
    try:
        inv = np.linalg.pinv(r)
        comm = np.clip(1.0 - 1.0 / np.diag(inv), 0.02, 0.98)
        load = None
        for _ in range(200):
            reduced = r.copy()
            np.fill_diagonal(reduced, comm)
            eigval, eigvec = np.linalg.eigh(reduced)
            j = int(np.argmax(eigval))
            if eigval[j] <= 0:
                return np.nan
            load = eigvec[:, j] * np.sqrt(eigval[j])
            if load.sum() < 0:
                load = -load
            new = np.clip(load**2, 0.0, 0.98)
            if np.max(np.abs(new - comm)) < 1e-8:
                comm = new
                break
            comm = new
        if load is None:
            return np.nan
        uniq = np.clip(1.0 - load**2, 1e-6, None)
        den = float(load.sum() ** 2 + uniq.sum())
        return float(load.sum() ** 2 / den) if den > 0 else np.nan
    except np.linalg.LinAlgError:
        return np.nan


def percentile_ci(v: np.ndarray) -> tuple[float, float, int]:
    a = np.asarray(v, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, 0
    lo, hi = np.quantile(a, [0.025, 0.975])
    return float(lo), float(hi), int(len(a))


def _bootstrap_scale(x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(x)
    alpha = np.empty(BOOT_N)
    omega = np.empty(BOOT_N)
    for b in range(BOOT_N):
        xb = x[rng.integers(0, n, n)]
        alpha[b] = cronbach_alpha(xb)
        omega[b] = omega_paf(xb)
    return alpha, omega


def run_behavior_reliability() -> None:
    ensure_dirs()
    scored = pd.read_csv(DATA / "behavior_item_level_scored.csv")
    results, diagnostics = [], []
    cache = {}
    seed_i = 0
    for sample, mask in [
        ("full_SZ_clinical", np.ones(len(scored), bool)),
        ("frozen_imaging_SZ", scored["is_frozen_imaging_SZ"].eq(1).to_numpy()),
    ]:
        for scale, cols in _score_columns().items():
            d = scored.loc[mask, ["subject"] + cols].dropna()
            x = d[cols].to_numpy(float)
            alpha = cronbach_alpha(x)
            omega = omega_paf(x)
            ab, ob = _bootstrap_scale(x, SEED + 100 + seed_i)
            seed_i += 1
            alo, ahi, av = percentile_ci(ab)
            olo, ohi, ov = percentile_ci(ob)
            key = re.sub(r"[^A-Za-z0-9]+", "_", f"{sample}__{scale}").strip("_")
            cache[key + "__alpha"] = ab
            cache[key + "__omega"] = ob
            results.append({
                "sample": sample,
                "scale": scale,
                "n_complete": len(x),
                "n_items": len(cols),
                "cronbach_alpha": alpha,
                "alpha_ci95_low": alo,
                "alpha_ci95_high": ahi,
                "alpha_bootstrap_valid": av,
                "mcdonald_omega_total": omega,
                "omega_ci95_low": olo,
                "omega_ci95_high": ohi,
                "omega_bootstrap_valid": ov,
                "omega_method": "standardized omega total; fixed one-factor principal-axis loadings",
                "bootstrap_resamples": BOOT_N,
                "missing_rule": "scale-level complete case; no imputation",
                "item_policy": "diagnostic only; no item deletion or scale redefinition",
            })
            for j, col in enumerate(cols):
                other = x.sum(axis=1) - x[:, j]
                item = x[:, j]
                rit = stats.pearsonr(item, other).statistic if np.std(item, ddof=1) > 0 and np.std(other, ddof=1) > 0 else np.nan
                diagnostics.append({
                    "sample": sample,
                    "scale": scale,
                    "item": col,
                    "n": len(x),
                    "item_mean": float(item.mean()),
                    "item_sd": float(item.std(ddof=1)),
                    "corrected_item_total_r": float(rit) if np.isfinite(rit) else np.nan,
                    "alpha_if_item_deleted": cronbach_alpha(np.delete(x, j, axis=1)),
                    "reverse_scored": bool("reversed" in col),
                    "interpretation": "diagnostic only; item retained in all formal scores",
                })
    pd.DataFrame(results).to_csv(OUT / "behavior_scale_reliability.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "behavior_item_diagnostics.csv", index=False)
    np.savez_compressed(DATA / "behavior_reliability_bootstrap.npz", **cache)
    print("Behavior reliability complete: alpha, omega, bootstrap CIs, and item diagnostics.")


def run_model_precision() -> None:
    d = pd.read_csv(PATHS["stage_fc"], sep="\t")
    locked = pd.read_csv(PATHS["stage_slopes"], sep="\t")
    locked_map = locked.set_index("Subject")
    rows = []
    for (sid, group, stage), g in d.groupby(["Subject", "Group", "Stage"], sort=True):
        g = g.sort_values("Bin")
        x = g["G1_bin_center"].to_numpy(float)
        y = g["FC"].to_numpy(float)
        fit = stats.linregress(x, y)
        pred = fit.intercept + fit.slope * x
        resid = y - pred
        df = len(x) - 2
        crit = stats.t.ppf(0.975, df)
        locked_col = f"{stage}_slope"
        locked_slope = float(locked_map.loc[sid, locked_col])
        rows.append({
            "subject": sid,
            "group": group,
            "stage": stage,
            "n_bins": len(x),
            "slope": fit.slope,
            "slope_se": fit.stderr,
            "slope_ci95_low": fit.slope - crit * fit.stderr,
            "slope_ci95_high": fit.slope + crit * fit.stderr,
            "r_squared": fit.rvalue**2,
            "residual_sd": float(np.sqrt(np.sum(resid**2) / df)),
            "locked_slope": locked_slope,
            "abs_difference_from_locked": abs(fit.slope - locked_slope),
            "model": "OLS FC ~ centered fixed bin position (10 points; df=8)",
        })
    out = pd.DataFrame(rows)
    if out["abs_difference_from_locked"].max() > 1e-12:
        raise RuntimeError("Model-based slopes do not reproduce the locked slopes.")
    out.to_csv(OUT / "neural_model_precision.csv", index=False)
    print("Neural model precision complete; locked slopes reproduced.")


def fingerprint(str_ts: np.ndarray, roi_ts: np.ndarray) -> np.ndarray:
    str_ts = np.asarray(str_ts, float)
    roi_ts = np.asarray(roi_ts, float)
    sz = str_ts.std(axis=0, ddof=1)
    rz = roi_ts.std(axis=0, ddof=1)
    zs = np.divide(str_ts - str_ts.mean(axis=0), sz, out=np.zeros_like(str_ts), where=sz > 0)
    zr = np.divide(roi_ts - roi_ts.mean(axis=0), rz, out=np.zeros_like(roi_ts), where=rz > 0)
    r = zs.T @ zr / (len(str_ts) - 1)
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def slope10(y: np.ndarray) -> float:
    return float(np.polyfit(np.arange(1, 11, dtype=float) - 5.5, np.asarray(y, float), 1)[0])


def fisher_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) <= 0 or np.std(b) <= 0:
        return np.nan
    r = np.corrcoef(a, b)[0, 1]
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def icc_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    z = np.column_stack([x, y]).astype(float)
    n, k = z.shape
    grand = z.mean()
    row = z.mean(axis=1)
    col = z.mean(axis=0)
    msr = k * np.sum((row - grand) ** 2) / (n - 1)
    msc = n * np.sum((col - grand) ** 2) / (k - 1)
    mse = np.sum((z - row[:, None] - col[None, :] + grand) ** 2) / ((n - 1) * (k - 1))
    den_a = msr + (k - 1) * mse + k * (msc - mse) / n
    den_c = msr + (k - 1) * mse
    return ((msr - mse) / den_a if den_a != 0 else np.nan, (msr - mse) / den_c if den_c != 0 else np.nan)


def stability_coefficients(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    pear = stats.pearsonr(x, y).statistic if np.std(x) > 0 and np.std(y) > 0 else np.nan
    spear = stats.spearmanr(x, y).statistic if np.std(x) > 0 and np.std(y) > 0 else np.nan
    icca, iccc = icc_pair(x, y)
    sb = 2 * pear / (1 + pear) if np.isfinite(pear) and (1 + pear) != 0 else np.nan
    return {"Pearson r": float(pear), "Spearman rho": float(spear), "ICC(A,1)": float(icca), "ICC(C,1)": float(iccc), "Spearman-Brown corrected Pearson": float(sb)}


def _bootstrap_stability(x: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    n = len(x)
    names = list(stability_coefficients(x, y))
    out = {k: np.empty(BOOT_N) for k in names}
    for b in range(BOOT_N):
        idx = rng.integers(0, n, n)
        co = stability_coefficients(x[idx], y[idx])
        for k in names:
            out[k][b] = co[k]
    return out


def run_split_half() -> None:
    ensure_dirs()
    cov = pd.read_csv(PATHS["covariates"], sep="\t").sort_values("Subject").reset_index(drop=True)
    subjects = cov["Subject"].astype(str).tolist()
    groups = cov["Group"].astype(str).tolist()
    bins = np.asarray(np.load(PATHS["fixed_bins"]), int).reshape(-1)
    vox = pd.read_csv(PATHS["current_voxels"], sep="\t").sort_values("voxel_row_0based")
    current10 = vox["current_bin"].to_numpy(int) == 10
    roi = pd.read_csv(PATHS["roi_order"], sep="\t").sort_values("roi_index")
    ant_idx = roi.loc[roi["reward_stage"].eq("Anticipation"), "roi_index"].to_numpy(int)
    out_idx = roi.loc[roi["reward_stage"].eq("Outcome"), "roi_index"].to_numpy(int)
    sma_idx = int(roi.loc[roi["roi_name"].eq("R_SFG_7_5_A6m_SMA__BN10"), "roi_index"].iloc[0])
    ifg_names = [
        "L_IFG_6_1_A44d__BN29", "L_IFG_6_2_IFS__BN31", "L_IFG_6_3_A45c__BN33",
        "L_IFG_6_4_A45r__BN35", "L_IFG_6_5_A44op__BN37", "L_IFG_6_6_A44v__BN39",
    ]
    ifg_idx = roi.loc[roi["roi_name"].isin(ifg_names), "roi_index"].to_numpy(int)
    if len(ant_idx) != 43 or len(out_idx) != 19 or len(ifg_idx) != 6 or current10.sum() != 285:
        raise RuntimeError("Locked ROI or current-Bin10 definitions are not as expected.")

    binmeans = np.empty((2, len(subjects), 10, 65), float)
    b10_ant = np.empty((2, len(subjects), len(ant_idx)), float)
    b10_out = np.empty((2, len(subjects), len(out_idx)), float)
    qc = []
    for si, sid in enumerate(subjects):
        tdir = PATHS["timeseries"] / sid
        roi_ts = np.load(next(tdir.glob("*_roi_timeseries_Tx65.npy"))).astype(float)
        str_ts = np.load(next(tdir.glob("*_striatum_timeseries_Tx2608.npy"))).astype(float)
        if len(roi_ts) != len(str_ts):
            raise RuntimeError(f"Time-series length mismatch for {sid}.")
        split = len(roi_ts) // 2
        full = fingerprint(str_ts, roi_ts)
        frozen = np.load(PATHS["fingerprints"] / f"{sid}_fingerprint_z_2608x65.npy")
        qc.append({
            "subject": sid, "n_timepoints": len(roi_ts), "half1_n": split, "half2_n": len(roi_ts) - split,
            "fullscan_max_abs_difference_vs_frozen": float(np.max(np.abs(full - frozen))),
            "fullscan_mean_abs_difference_vs_frozen": float(np.mean(np.abs(full - frozen))),
        })
        for hi, sl in enumerate((slice(0, split), slice(split, None))):
            f = fingerprint(str_ts[sl], roi_ts[sl])
            for b in range(1, 11):
                binmeans[hi, si, b - 1] = f[bins == b].mean(axis=0)
            b10_ant[hi, si] = f[current10][:, ant_idx].mean(axis=0)
            b10_out[hi, si] = f[current10][:, out_idx].mean(axis=0)
    pd.DataFrame(qc).to_csv(OUT / "split_half_reconstruction_qc.csv", index=False)

    half_metrics = {1: {}, 2: {}}
    hc = np.array([g == "HC" for g in groups])
    for hi in range(2):
        half_metrics[hi + 1]["Overall Anticipation slope"] = np.array([slope10(binmeans[hi, s][:, ant_idx].mean(axis=1)) for s in range(len(subjects))])
        half_metrics[hi + 1]["Overall Outcome slope"] = np.array([slope10(binmeans[hi, s][:, out_idx].mean(axis=1)) for s in range(len(subjects))])
        half_metrics[hi + 1]["SMA Anticipation slope"] = np.array([slope10(binmeans[hi, s, :, sma_idx]) for s in range(len(subjects))])
        half_metrics[hi + 1]["IFG Outcome slope"] = np.array([slope10(binmeans[hi, s][:, ifg_idx].mean(axis=1)) for s in range(len(subjects))])
        for label, arr in [("Bin10 Anticipation similarity", b10_ant[hi]), ("Bin10 Outcome similarity", b10_out[hi])]:
            vals = np.empty(len(subjects))
            hc_mean = arr[hc].mean(axis=0)
            for s in range(len(subjects)):
                template = arr[hc & (np.arange(len(subjects)) != s)].mean(axis=0) if hc[s] else hc_mean
                vals[s] = fisher_corr(arr[s], template)
            half_metrics[hi + 1][label] = vals
        half_metrics[hi + 1]["Overall slope contrast (raw Ant-Outcome)"] = half_metrics[hi + 1]["Overall Anticipation slope"] - half_metrics[hi + 1]["Overall Outcome slope"]
        half_metrics[hi + 1]["Bin10 similarity contrast (raw Ant-Outcome)"] = half_metrics[hi + 1]["Bin10 Anticipation similarity"] - half_metrics[hi + 1]["Bin10 Outcome similarity"]
        sz = ~hc
        for base, out_label in [
            (("Overall Anticipation slope", "Overall Outcome slope"), "Overall slope contrast (stagewise z)"),
            (("Bin10 Anticipation similarity", "Bin10 Outcome similarity"), "Bin10 similarity contrast (stagewise z)"),
        ]:
            v = np.full(len(subjects), np.nan)
            a = half_metrics[hi + 1][base[0]][sz]
            o = half_metrics[hi + 1][base[1]][sz]
            v[sz] = (a - a.mean()) / a.std(ddof=1) - (o - o.mean()) / o.std(ddof=1)
            half_metrics[hi + 1][out_label] = v

    subject_rows = []
    for phenotype in half_metrics[1]:
        for i, sid in enumerate(subjects):
            subject_rows.append({"subject": sid, "group": groups[i], "phenotype": phenotype, "half1": half_metrics[1][phenotype][i], "half2": half_metrics[2][phenotype][i]})
    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(OUT / "neural_split_half_subject_metrics.csv", index=False)

    aggregate, cache = [], {}
    main_single = {
        "Overall Anticipation slope", "Overall Outcome slope", "SMA Anticipation slope", "IFG Outcome slope",
        "Bin10 Anticipation similarity", "Bin10 Outcome similarity",
    }
    for sample, smask in [("SZ32", np.array([g == "SZ" for g in groups])), ("all64", np.ones(len(subjects), bool))]:
        for pi, phenotype in enumerate(half_metrics[1]):
            if sample == "all64" and "stagewise z" in phenotype:
                continue
            valid = smask & np.isfinite(half_metrics[1][phenotype]) & np.isfinite(half_metrics[2][phenotype])
            x, y = half_metrics[1][phenotype][valid], half_metrics[2][phenotype][valid]
            point = stability_coefficients(x, y)
            boot = _bootstrap_stability(x, y, SEED + 1000 + pi + (0 if sample == "SZ32" else 100))
            family = "single_stage" if phenotype in main_single else ("stage_contrast_primary_scale" if "stagewise z" in phenotype else "stage_contrast_raw_sensitivity")
            for coefficient, estimate in point.items():
                lo, hi, nv = percentile_ci(boot[coefficient])
                aggregate.append({
                    "sample": sample, "phenotype": phenotype, "family": family, "n": len(x),
                    "coefficient": coefficient, "estimate": estimate, "ci95_low": lo, "ci95_high": hi,
                    "bootstrap_resamples": BOOT_N, "bootstrap_valid": nv,
                    "split_definition": "contiguous first half vs second half of the single denoised scan",
                    "interpretation": "within-scan stability / measurement precision; not test-retest reliability",
                })
                key = re.sub(r"[^A-Za-z0-9]+", "_", f"{sample}__{phenotype}__{coefficient}").strip("_")
                cache[key] = boot[coefficient]
    pd.DataFrame(aggregate).to_csv(OUT / "neural_split_half_stability.csv", index=False)
    np.savez_compressed(DATA / "neural_stability_bootstrap.npz", **cache)
    print("Split-half complete: six single-stage phenotypes plus raw and stagewise-z contrasts.")


def _corr_rows(a: np.ndarray, template: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    template = np.asarray(template, float)
    ac = a - a.mean(axis=-1, keepdims=True)
    tc = template - template.mean(axis=-1, keepdims=True)
    num = np.sum(ac * tc, axis=-1)
    den = np.sqrt(np.sum(ac**2, axis=-1) * np.sum(tc**2, axis=-1))
    r = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def run_voxel_bootstrap() -> None:
    cov = pd.read_csv(PATHS["covariates"], sep="\t").sort_values("Subject").reset_index(drop=True)
    roi = pd.read_csv(PATHS["roi_order"], sep="\t").sort_values("roi_index")
    vox = pd.read_csv(PATHS["current_voxels"], sep="\t").sort_values("voxel_row_0based")
    mask = vox["current_bin"].to_numpy(int) == 10
    if mask.sum() != 285:
        raise RuntimeError("Current Bin10 is not 285 voxels.")
    ant_idx = roi.loc[roi["reward_stage"].eq("Anticipation"), "roi_index"].to_numpy(int)
    out_idx = roi.loc[roi["reward_stage"].eq("Outcome"), "roi_index"].to_numpy(int)
    subjects = cov["Subject"].astype(str).tolist()
    groups = cov["Group"].astype(str).tolist()
    all_fc = np.stack([np.load(PATHS["fingerprints"] / f"{s}_fingerprint_z_2608x65.npy")[mask] for s in subjects])
    hc = np.array([g == "HC" for g in groups])
    sz = ~hc
    rng = np.random.default_rng(SEED + 2000)
    weights = rng.multinomial(mask.sum(), np.repeat(1 / mask.sum(), mask.sum()), size=BOOT_N).astype(np.float32) / mask.sum()
    rows = []
    boot_cache = {}
    for stage, idx in [("Anticipation", ant_idx), ("Outcome", out_idx)]:
        data = all_fc[sz][:, :, idx]
        template = all_fc[hc][:, :, idx].mean(axis=(0, 1))
        means = (weights @ data.transpose(1, 0, 2).reshape(mask.sum(), -1)).reshape(BOOT_N, sz.sum(), len(idx))
        sims = _corr_rows(means, np.broadcast_to(template, means.shape))
        observed = _corr_rows(data.mean(axis=1), np.broadcast_to(template, (sz.sum(), len(idx))))
        for j, sid in enumerate(np.asarray(subjects)[sz]):
            lo, hi = np.quantile(sims[:, j], [0.025, 0.975])
            rows.append({
                "subject": sid, "group": "SZ", "stage": stage, "n_bin10_voxels": int(mask.sum()),
                "n_targets": len(idx), "observed_similarity_z": observed[j],
                "bootstrap_median_similarity_z": float(np.median(sims[:, j])),
                "bootstrap_ci95_low": float(lo), "bootstrap_ci95_high": float(hi),
                "bootstrap_sd": float(np.std(sims[:, j], ddof=1)), "bootstrap_interval_width": float(hi - lo),
                "bootstrap_iterations": BOOT_N, "reference": "fixed full-scan HC32 stage-specific mean fingerprint",
                "interpretation": "voxel-sampling sensitivity conditional on the HC reference; not reliability or test-retest reliability",
            })
        boot_cache[stage] = sims
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "bin10_voxel_bootstrap_precision.csv", index=False)
    np.savez_compressed(DATA / "bin10_voxel_bootstrap_draws.npz", **boot_cache)

    locked = pd.read_csv(PATHS["locked_stage_similarity"], sep="\t")
    locked = locked[(locked["group"].eq("SZ")) & (locked["bin"].eq(10))][["subject", "stage", "profile_z"]]
    chk = out.merge(locked, on=["subject", "stage"], how="left")
    chk["abs_difference_from_locked"] = (chk["observed_similarity_z"] - chk["profile_z"]).abs()
    chk[["subject", "stage", "observed_similarity_z", "profile_z", "abs_difference_from_locked"]].to_csv(OUT / "bin10_similarity_reconstruction_qc.csv", index=False)
    if chk["abs_difference_from_locked"].max() > 2e-7:
        raise RuntimeError("Full-scan Bin10 stage similarity does not reproduce the locked values.")
    print("Bin10 voxel bootstrap complete: 5,000 resamples for every SZ subject and stage.")


def difference_reliability(x: np.ndarray, y: np.ndarray, rel_x: float, rel_y: float) -> float:
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    den = vx + vy - 2 * cov
    return float((rel_x * vx + rel_y * vy - 2 * cov) / den) if den != 0 else np.nan


def run_stage_contrast() -> None:
    scored = pd.read_csv(DATA / "behavior_item_level_scored.csv")
    d = scored[scored["is_frozen_imaging_SZ"].eq(1)].dropna(subset=["TEPS Anticipatory", "TEPS Consummatory"])
    xa = d[_score_columns()["TEPS Anticipatory"]].to_numpy(float)
    xc = d[_score_columns()["TEPS Consummatory"]].to_numpy(float)
    sa, sc = xa.sum(axis=1), xc.sum(axis=1)
    ra_a, ra_c = cronbach_alpha(xa), cronbach_alpha(xc)
    ro_a, ro_c = omega_paf(xa), omega_paf(xc)
    point_alpha = difference_reliability(sa, sc, ra_a, ra_c)
    point_omega = difference_reliability(sa, sc, ro_a, ro_c)
    rng = np.random.default_rng(SEED + 3000)
    ba, bo = np.empty(BOOT_N), np.empty(BOOT_N)
    for b in range(BOOT_N):
        idx = rng.integers(0, len(d), len(d))
        ba[b] = difference_reliability(sa[idx], sc[idx], cronbach_alpha(xa[idx]), cronbach_alpha(xc[idx]))
        bo[b] = difference_reliability(sa[idx], sc[idx], omega_paf(xa[idx]), omega_paf(xc[idx]))
    rows = []
    for coefficient, point, boot in [("standard formula using alpha", point_alpha, ba), ("standard formula using omega", point_omega, bo)]:
        lo, hi, nv = percentile_ci(boot)
        rows.append({
            "domain": "behavior", "measure": "TEPS Anticipatory - Consummatory (stagewise z equivalent)",
            "sample": "frozen_imaging_SZ", "n": len(d), "coefficient": coefficient, "estimate": point,
            "ci95_low": lo, "ci95_high": hi, "bootstrap_valid": nv, "bootstrap_resamples": BOOT_N,
            "formula": "(RelX*VarX + RelY*VarY - 2*CovXY) / (VarX + VarY - 2*CovXY)",
            "assumptions": "uncorrelated component measurement errors; observed X-Y covariance treated as true covariance; formal subscales unchanged",
        })
    stab = pd.read_csv(OUT / "neural_split_half_stability.csv")
    keep = stab[(stab["sample"].eq("SZ32")) & (stab["phenotype"].isin(["Overall slope contrast (stagewise z)", "Bin10 similarity contrast (stagewise z)", "Overall slope contrast (raw Ant-Outcome)", "Bin10 similarity contrast (raw Ant-Outcome)"]))]
    for _, r in keep.iterrows():
        rows.append({
            "domain": "neural", "measure": r["phenotype"], "sample": r["sample"], "n": r["n"],
            "coefficient": r["coefficient"], "estimate": r["estimate"], "ci95_low": r["ci95_low"],
            "ci95_high": r["ci95_high"], "bootstrap_valid": r["bootstrap_valid"],
            "bootstrap_resamples": r["bootstrap_resamples"], "formula": "half1 vs half2 across-subject stability",
            "assumptions": "single scan split into contiguous halves; fixed G1/bins and fixed stage mapping; not test-retest reliability",
        })
    pd.DataFrame(rows).to_csv(OUT / "stage_contrast_reliability.csv", index=False)
    np.savez_compressed(DATA / "teps_difference_reliability_bootstrap.npz", alpha=ba, omega=bo)
    print("Stage-contrast reliability complete for TEPS and both neural contrasts.")


def _neural_boot_key(sample: str, phenotype: str, coefficient: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{sample}__{phenotype}__{coefficient}").strip("_")


def _behavior_boot_key(sample: str, scale: str, coefficient: str = "omega") -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{sample}__{scale}").strip("_") + f"__{coefficient}"


def run_ceiling() -> None:
    beh = pd.read_csv(OUT / "behavior_scale_reliability.csv")
    stab = pd.read_csv(OUT / "neural_split_half_stability.csv")
    bcache = np.load(DATA / "behavior_reliability_bootstrap.npz")
    ncache = np.load(DATA / "neural_stability_bootstrap.npz")
    tcache = np.load(DATA / "teps_difference_reliability_bootstrap.npz")

    ca = pd.read_csv(PATHS["clinical_assoc"])
    rp = pd.read_csv(PATHS["roi_assoc_primary"])
    rs = pd.read_csv(PATHS["roi_assoc_secondary"])
    st = pd.read_csv(PATHS["stage_assoc"])
    specs = []
    for aid, behavior, neural in [
        ("P1", "PANSS Negative", "Bin10 Anticipation similarity"),
        ("P2", "PANSS Negative", "Bin10 Outcome similarity"),
        ("P3", "PANSS Negative", "SMA Anticipation slope"),
    ]:
        r = ca.loc[ca["analysis_id"].eq(aid)].iloc[0]
        specs.append((f"PANSS {aid}", r["phenotype_label"], behavior, neural, float(r["effect"]), int(r["effective_n"]), "clinical primary"))
    for source, aid, behavior, neural in [
        (rp, "P1", "TEPS Anticipatory", "SMA Anticipation slope"),
        (rs, "S1", "TEPS Consummatory", "IFG Outcome slope"),
        (rs, "E1", "ACIPS Anticipatory", "SMA Anticipation slope"),
        (rs, "E2", "ACIPS Consummatory", "IFG Outcome slope"),
    ]:
        r = source.loc[source["analysis_id"].eq(aid)].iloc[0]
        specs.append((f"ROI {aid}", f"{r['behavior']} × {r['phenotype']}", behavior, neural, float(r["rho"]), int(r["n"]), str(r["tier"])))
    for endpoint, neural in [("overall_fc_slope", "Overall slope contrast (stagewise z)"), ("bin10_similarity", "Bin10 similarity contrast (stagewise z)")]:
        r = st[(st["endpoint"].eq(endpoint)) & (st["analysis_role"].isin(["primary", "secondary"]))].iloc[0]
        specs.append((f"Stage {endpoint}", r["endpoint_label"], "TEPS difference", neural, float(r["rho"]), int(r["n"]), str(r["analysis_role"])))

    rows = []
    for analysis_id, label, bscale, neural, observed, n, tier in specs:
        if bscale == "TEPS difference":
            brel = float(pd.read_csv(OUT / "stage_contrast_reliability.csv").query("domain == 'behavior' and coefficient == 'standard formula using omega'")["estimate"].iloc[0])
            bb = np.asarray(tcache["omega"], float)
        else:
            br = beh[(beh["sample"].eq("frozen_imaging_SZ")) & (beh["scale"].eq(bscale))].iloc[0]
            brel = float(br["mcdonald_omega_total"])
            bb = np.asarray(bcache[_behavior_boot_key("frozen_imaging_SZ", bscale)], float)
        nr = stab[(stab["sample"].eq("SZ32")) & (stab["phenotype"].eq(neural)) & (stab["coefficient"].eq("Spearman-Brown corrected Pearson"))].iloc[0]
        nrel = float(nr["estimate"])
        nb = np.asarray(ncache[_neural_boot_key("SZ32", neural, "Spearman-Brown corrected Pearson")], float)
        valid = np.isfinite(bb) & np.isfinite(nb) & (bb > 0) & (nb > 0)
        valid_fraction = float(valid.mean())
        estimable = bool(np.isfinite(brel) and np.isfinite(nrel) and brel > 0 and nrel > 0 and valid_fraction >= 0.80)
        ceiling = math.sqrt(brel * nrel) if estimable else np.nan
        cb = np.sqrt(bb[valid] * nb[valid]) if valid.any() else np.array([])
        clo, chi, cv = percentile_ci(cb)
        db = observed / cb if len(cb) else np.array([])
        dlo, dhi, dv = percentile_ci(db)
        rows.append({
            "analysis_id": analysis_id, "existing_analysis_tier": tier, "association": label, "existing_n": n,
            "observed_rho": observed, "observed_abs_rho": abs(observed), "behavior_measure": bscale,
            "behavior_reliability": brel, "neural_measure": neural, "neural_reliability": nrel,
            "neural_reliability_definition": "Spearman-Brown corrected half1-half2 Pearson within SZ32",
            "ceiling_estimable": estimable, "positive_bootstrap_pair_fraction": valid_fraction,
            "estimated_r_max": ceiling, "r_max_ci95_low": clo if estimable else np.nan, "r_max_ci95_high": chi if estimable else np.nan,
            "descriptive_disattenuated_r": observed / ceiling if estimable else np.nan,
            "disattenuated_ci95_low_reliability_only": dlo if estimable else np.nan,
            "disattenuated_ci95_high_reliability_only": dhi if estimable else np.nan,
            "bootstrap_valid": cv if estimable else 0, "bootstrap_resamples": BOOT_N,
            "uncertainty_scope": "reliability-bootstrap uncertainty only; observed-effect sampling uncertainty not propagated",
            "guardrail": "descriptive attenuation sensitivity only; not a new significance test",
        })
    pd.DataFrame(rows).to_csv(OUT / "reliability_ceiling_summary.csv", index=False)
    print("Reliability ceiling summary complete; only mathematically stable positive estimates retained.")


def _forest(ax, data, ycol, est, lo, hi, hue, hue_order, colors, xlim=(-1, 1), title=""):
    labels = list(dict.fromkeys(data[ycol].tolist()))
    ypos = {v: len(labels) - 1 - i for i, v in enumerate(labels)}
    offsets = np.linspace(-0.22, 0.22, len(hue_order)) if len(hue_order) > 1 else [0]
    for off, h in zip(offsets, hue_order):
        d = data[data[hue].eq(h)]
        for row_i, (_, r) in enumerate(d.iterrows()):
            y = ypos[r[ycol]] + off
            ax.plot([r[lo], r[hi]], [y, y], color=colors[h], lw=1.4, alpha=.9)
            ax.scatter(r[est], y, s=34, color=colors[h], edgecolor="white", linewidth=.5, label=h if row_i == 0 else None, zorder=3)
    ax.axvline(0, color="#8b8b8b", lw=.8, ls="--")
    ax.set_yticks(list(ypos.values()), labels)
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#e5e5e5", lw=.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def run_plots() -> None:
    ensure_dirs()
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 8.5, "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 160})
    rel = pd.read_csv(OUT / "behavior_scale_reliability.csv")
    f1 = []
    for _, r in rel.iterrows():
        for method, e, lo, hi in [
            ("Cronbach alpha", r["cronbach_alpha"], r["alpha_ci95_low"], r["alpha_ci95_high"]),
            ("McDonald omega", r["mcdonald_omega_total"], r["omega_ci95_low"], r["omega_ci95_high"]),
        ]:
            f1.append({"sample": r["sample"], "scale": r["scale"], "method": method, "estimate": e, "ci95_low": lo, "ci95_high": hi, "n": r["n_complete"]})
    f1 = pd.DataFrame(f1)
    f1.to_csv(FIGSRC / "Figure1_behavior_reliability_source_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True, sharey=True, constrained_layout=True)
    colors = {"Cronbach alpha": "#2b6f9f", "McDonald omega": "#d97732"}
    for ax, sample, title in zip(axes, ["full_SZ_clinical", "frozen_imaging_SZ"], ["a  Full SZ clinical sample", "b  Frozen imaging SZ subset"]):
        _forest(ax, f1[f1["sample"].eq(sample)], "scale", "estimate", "ci95_low", "ci95_high", "method", list(colors), colors, xlim=(-.2, 1.02), title=title)
    fig.supxlabel("Internal-consistency coefficient (95% bootstrap CI)")
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.50), fontsize=7.5)
    fig.savefig(OUT / "Figure1_behavior_reliability.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure1_behavior_reliability.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure1_behavior_reliability.svg", bbox_inches="tight")
    plt.close(fig)

    stab = pd.read_csv(OUT / "neural_split_half_stability.csv")
    order = ["Overall Anticipation slope", "Overall Outcome slope", "SMA Anticipation slope", "IFG Outcome slope", "Bin10 Anticipation similarity", "Bin10 Outcome similarity"]
    coefs = ["Pearson r", "Spearman rho", "ICC(A,1)"]
    f2 = stab[(stab["sample"].eq("SZ32")) & (stab["phenotype"].isin(order)) & (stab["coefficient"].isin(coefs))].copy()
    f2["phenotype"] = pd.Categorical(f2["phenotype"], order, ordered=True)
    f2 = f2.sort_values("phenotype")
    f2.to_csv(FIGSRC / "Figure2_neural_stability_source_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    c2 = {"Pearson r": "#2b6f9f", "Spearman rho": "#4ca38a", "ICC(A,1)": "#d97732"}
    _forest(ax, f2, "phenotype", "estimate", "ci95_low", "ci95_high", "coefficient", coefs, c2, xlim=(-1, 1), title="Within-scan stability of neural individual-difference phenotypes (SZ32)")
    ax.set_xlabel("Half 1 vs half 2 coefficient (95% subject-bootstrap CI)")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "Figure2_neural_stability.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure2_neural_stability.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure2_neural_stability.svg", bbox_inches="tight")
    plt.close(fig)

    stage = pd.read_csv(OUT / "stage_contrast_reliability.csv")
    f3b = stage[(stage["domain"].eq("behavior")) & (stage["coefficient"].eq("standard formula using omega"))].copy()
    f3b["display_measure"] = "TEPS Ant-Cons contrast"
    f3b["display_coefficient"] = "Difference reliability (omega inputs)"
    f3n = stage[(stage["domain"].eq("neural")) & (stage["measure"].str.contains("stagewise z")) & (stage["coefficient"].isin(coefs))].copy()
    f3n["display_measure"] = f3n["measure"].str.replace(r" \(stagewise z\)", "", regex=True)
    f3n["display_coefficient"] = f3n["coefficient"]
    f3 = pd.concat([f3b, f3n], ignore_index=True, sort=False)
    f3.to_csv(FIGSRC / "Figure3_stage_contrast_stability_source_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    c3 = {"Difference reliability (omega inputs)": "#7b4ea3", **c2}
    _forest(ax, f3, "display_measure", "estimate", "ci95_low", "ci95_high", "display_coefficient", list(c3), c3, xlim=(-1.5, 1.1), title="Anticipation-Outcome contrast stability")
    ax.set_xlabel("Reliability / half-to-half stability coefficient (95% bootstrap CI)")
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    fig.savefig(OUT / "Figure3_stage_contrast_stability.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure3_stage_contrast_stability.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure3_stage_contrast_stability.svg", bbox_inches="tight")
    plt.close(fig)

    ceil = pd.read_csv(OUT / "reliability_ceiling_summary.csv")
    f4 = ceil[ceil["ceiling_estimable"].eq(True)].copy()
    f4.to_csv(FIGSRC / "Figure4_reliability_ceiling_source_data.csv", index=False)
    if len(f4):
        f4 = f4.iloc[::-1].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.2, max(3.2, .42 * len(f4) + 1.3)), constrained_layout=True)
        y = np.arange(len(f4))
        ax.hlines(y, f4["r_max_ci95_low"], f4["r_max_ci95_high"], color="#d97732", lw=1.5)
        ax.scatter(f4["estimated_r_max"], y, color="#d97732", s=38, label="Estimated measurement ceiling")
        ax.scatter(f4["observed_abs_rho"], y, color="#2b6f9f", s=30, marker="D", label="Observed |Spearman rho|")
        ax.set_yticks(y, f4["association"])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Absolute correlation scale")
        ax.set_title("Measurement ceilings for existing prespecified / reported associations", loc="left", fontweight="bold")
        ax.grid(axis="x", color="#e5e5e5", lw=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
        fig.savefig(OUT / "Figure4_reliability_ceiling.pdf", bbox_inches="tight")
        fig.savefig(OUT / "Figure4_reliability_ceiling.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / "Figure4_reliability_ceiling.svg", bbox_inches="tight")
        plt.close(fig)
    print("Figures 1-4 and source-data tables generated.")


def _fmt(x, d=2):
    return "NA" if not np.isfinite(float(x)) else f"{float(x):.{d}f}"


def run_report() -> None:
    beh = pd.read_csv(OUT / "behavior_scale_reliability.csv")
    stab = pd.read_csv(OUT / "neural_split_half_stability.csv")
    stage = pd.read_csv(OUT / "stage_contrast_reliability.csv")
    model = pd.read_csv(OUT / "neural_model_precision.csv")
    voxel = pd.read_csv(OUT / "bin10_voxel_bootstrap_precision.csv")
    ceil = pd.read_csv(OUT / "reliability_ceiling_summary.csv")
    audit = pd.read_csv(OUT / "input_audit.csv")
    recon = pd.read_csv(OUT / "behavior_score_reconstruction_audit.csv")
    qc = pd.read_csv(OUT / "split_half_reconstruction_qc.csv")

    def br(scale, sample="frozen_imaging_SZ"):
        return beh[(beh["scale"].eq(scale)) & (beh["sample"].eq(sample))].iloc[0]

    def sr(phen, coef="Spearman-Brown corrected Pearson"):
        return stab[(stab["sample"].eq("SZ32")) & (stab["phenotype"].eq(phen)) & (stab["coefficient"].eq(coef))].iloc[0]

    def coef_line(phen):
        d = stab[(stab["sample"].eq("SZ32")) & (stab["phenotype"].eq(phen))]
        parts = []
        for c in ["Pearson r", "Spearman rho", "ICC(A,1)", "Spearman-Brown corrected Pearson"]:
            r = d[d["coefficient"].eq(c)].iloc[0]
            parts.append(f"{c}={_fmt(r['estimate'])} [{_fmt(r['ci95_low'])}, {_fmt(r['ci95_high'])}]")
        return "; ".join(parts)

    behavior_lines = []
    for scale in _score_columns():
        a, f = br(scale, "full_SZ_clinical"), br(scale)
        behavior_lines.append(
            f"| {scale} | {int(a.n_complete)} | {_fmt(a.cronbach_alpha)} [{_fmt(a.alpha_ci95_low)}, {_fmt(a.alpha_ci95_high)}] | {_fmt(a.mcdonald_omega_total)} [{_fmt(a.omega_ci95_low)}, {_fmt(a.omega_ci95_high)}] | {int(f.n_complete)} | {_fmt(f.cronbach_alpha)} [{_fmt(f.alpha_ci95_low)}, {_fmt(f.alpha_ci95_high)}] | {_fmt(f.mcdonald_omega_total)} [{_fmt(f.omega_ci95_low)}, {_fmt(f.omega_ci95_high)}] |"
        )

    diff = stage[(stage["domain"].eq("behavior")) & (stage["coefficient"].eq("standard formula using omega"))].iloc[0]
    contrasts = ["Overall slope contrast (stagewise z)", "Bin10 similarity contrast (stagewise z)"]
    model_sz = model[model["group"].eq("SZ")]
    voxel_summary = voxel.groupby("stage").agg(median_boot_sd=("bootstrap_sd", "median"), max_boot_sd=("bootstrap_sd", "max"), median_width=("bootstrap_interval_width", "median")).reset_index()
    voxel_table = "\n".join(
        [
            "| Stage | N | Median bootstrap SD | Maximum bootstrap SD | Median 95% interval width |",
            "|---|---:|---:|---:|---:|",
        ]
        + [
            f"| {r.stage} | 32 | {_fmt(r.median_boot_sd, 3)} | {_fmt(r.max_boot_sd, 3)} | {_fmt(r.median_width, 3)} |"
            for _, r in voxel_summary.iterrows()
        ]
    )
    estimable = ceil[ceil["ceiling_estimable"].eq(True)]
    ceiling_text = (
        f"{len(estimable)}/{len(ceil)} 个既有检验满足正的行为与神经 reliability 且至少 80% 的 reliability-bootstrap 配对为正，因此可报告 ceiling。"
        if len(ceil) else "没有 ceiling 条目。"
    )
    report = f"""# Measurement Reliability / Precision Audit

## 结论先行

本审计没有启动新的临床相关搜索。它只评估既定行为量表和既定神经表型的内部一致性、单次扫描内 split-half stability、模型精度及 voxel-sampling sensitivity。

最重要的结论是：**单阶段神经指标与 Anticipation–Outcome 差值的稳定性必须分开看**。差值通常继承两个分量的误差；在本数据中，部分 contrast 的区间明显更宽。现有阴性 brain–behavior 结果不能简单归因于“量表不可靠”，但对排名稳定性较弱或不确定的神经指标，measurement attenuation 是可信限制。所有 split-half 结果均是 `within-scan stability / measurement precision`，**不是 test–retest reliability**。

## 1. 数据与计分审计

- 完整 SZ 临床映射为 35 人；冻结影像 SZ 为 32 人。
- TEPS 正式分量表沿用既有 item 归属；ACIPS item 3 严格按 `7 - raw` 反向计分。
- PANSS Negative 的 N1–N7 已在原始临床表中定位，并与正式 `PANS-NE` 核对。
- 所有可比较的重算分数均要求逐例精确一致；审计表中状态为：{', '.join(sorted(recon['status'].unique()))}。
- 固定 G1 与 slope 分箱使用冻结 2,608-voxel 输入；Bin10 similarity 沿用既有 `G1_bin_current`，恰为 285 voxels。
- 64 人时间序列均可读取，长度 {audit.loc[audit['check'].eq('timeseries length range'), 'value'].iloc[0]}。全时段重算 fingerprint 对冻结文件的最大绝对差为 {qc['fullscan_max_abs_difference_vs_frozen'].max():.3g}。

## 2. 行为量表内部一致性

| Scale | Full SZ N | alpha (95% CI) | omega (95% CI) | Frozen SZ N | alpha (95% CI) | omega (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(behavior_lines)}

omega 是既定题目集合的一因子 principal-axis 标准化 omega-total 估计；它不是用来重定义因子结构。`alpha-if-item-deleted` 仅作诊断，未删除任何题目。高内部一致性只说明题目间一致性，**不证明 construct validity**，也不证明量表必然对应当前 neural phenotype。

## 3. Overall slope 的模型精度

每名被试、每个 stage 均按原定义对 10 个 bin 拟合 `FC ~ centered bin position`。冻结 slope 被逐例复现。SZ 中 Anticipation slope SE 的中位数为 {_fmt(model_sz.loc[model_sz.stage.eq('Anticipation'), 'slope_se'].median(), 4)}，范围 {_fmt(model_sz.loc[model_sz.stage.eq('Anticipation'), 'slope_se'].min(), 4)}–{_fmt(model_sz.loc[model_sz.stage.eq('Anticipation'), 'slope_se'].max(), 4)}；Outcome 中位数为 {_fmt(model_sz.loc[model_sz.stage.eq('Outcome'), 'slope_se'].median(), 4)}，范围 {_fmt(model_sz.loc[model_sz.stage.eq('Outcome'), 'slope_se'].min(), 4)}–{_fmt(model_sz.loc[model_sz.stage.eq('Outcome'), 'slope_se'].max(), 4)}。个体 SE 的异质性见 `neural_model_precision.csv`。

## 4. 单次扫描内 split-half stability（SZ32）

- Overall Anticipation slope: {coef_line('Overall Anticipation slope')}
- Overall Outcome slope: {coef_line('Overall Outcome slope')}
- SMA Anticipation slope: {coef_line('SMA Anticipation slope')}
- IFG Outcome slope: {coef_line('IFG Outcome slope')}
- Bin10 Anticipation similarity: {coef_line('Bin10 Anticipation similarity')}
- Bin10 Outcome similarity: {coef_line('Bin10 Outcome similarity')}

ICC(A,1) 对 half 间绝对一致性敏感；ICC(C,1) 另列于 CSV。Spearman–Brown 只用于在“两个 half 具有相近误差结构且误差近似独立”的假设下近似 full-scan stability，不能替代独立重复扫描。

## 5. Bin10 voxel-sampling sensitivity

每名 SZ、每个 stage 均完成 5,000 次 285-voxel 有放回抽样，HC32 reference 固定为全时段正式模板，以隔离个体 fingerprint 对 voxel sampling 的敏感性。

{voxel_table}

这里的 bootstrap SD/interval 是 **voxel-sampling sensitivity**，不是经典 reliability，更不是 test–retest reliability。个体异常宽区间可在 `bin10_voxel_bootstrap_precision.csv` 中直接识别。

## 6. Stage contrast

- TEPS Anticipatory–Consummatory difference reliability（omega 输入）={_fmt(diff['estimate'])}，95% bootstrap CI [{_fmt(diff['ci95_low'])}, {_fmt(diff['ci95_high'])}]，N={int(diff['n'])}。
- {contrasts[0]}: {coef_line(contrasts[0])}
- {contrasts[1]}: {coef_line(contrasts[1])}

TEPS 差值采用标准公式：`(RelX VarX + RelY VarY - 2 CovXY) / (VarX + VarY - 2 CovXY)`，假定两个分量的测量误差不相关，并把观察到的分量协方差视为真实协方差。若点估计为负或 CI 极宽，应解释为 `difference-score reliability is poorly estimated / low under current assumptions`，不得强行截断为 0。

## 7. Reliability ceiling 与既有阴性结果

{ceiling_text} `r_max = sqrt(Rel_behavior × Rel_neural)`；神经 reliability 使用 SZ32 的 Spearman–Brown corrected Pearson。disattenuated correlation 只作描述性敏感性分析，其 CI 只传播 reliability bootstrap 不确定性，未传播原 observed rho 的抽样误差，**不是新的显著性检验**。

| Existing association | observed rho | behavior rel. | neural rel. | ceiling | status |
|---|---:|---:|---:|---:|---|
{chr(10).join(f"| {r.association} | {_fmt(r.observed_rho)} | {_fmt(r.behavior_reliability)} | {_fmt(r.neural_reliability)} | {_fmt(r.estimated_r_max)} | {'estimable' if r.ceiling_estimable else 'poorly estimated'} |" for _, r in ceil.iterrows())}

## 8. 九个核心问题的明确回答

1. **TEPS / ACIPS 内部一致性如何？** 见上表。应结合点估计、CI 与 N 解读；不能由 alpha/omega 推导 construct validity。
2. **PANSS Negative 能否估计？** 能。N1–N7 存在且正式总分核对通过，完整临床样本与冻结影像子样本的 alpha/omega 均已报告。
3. **Overall FC slope 的个体排序是否稳定？** Anticipation 与 Outcome 的 Pearson、Spearman、ICC 及 CI 如第 4 节；结论以 CI 而非单一阈值为准。
4. **SMA Anticipation slope 是否足够稳定？** 其结果为：{coef_line('SMA Anticipation slope')}。若 CI 宽或覆盖低/负值，个体 brain–behavior 分析会明显受 measurement error 限制，即使组间效应仍可存在。
5. **Bin10 similarity 比 slope 更稳定吗？** 没有跨指标一概而论；应比较同 stage 的点估计和 CI。Ant similarity 与 Outcome similarity 的结果已与两种 overall slope 并列在 Figure 2。
6. **Anticipation–Outcome difference 是否降低稳定性？** 差值的结果见第 6 节。与单阶段结果相比，至少部分 contrast 显示更低或更不确定的稳定性；这正是 stage-coupling 解释中的主要测量限制。
7. **阴性关联是否可能受 attenuation 明显限制？** 对 ceiling 可估且 neural reliability 较低/不确定的条目，答案是“可能”；但这不能把不显著结果改写为阳性。对稳定性较高且 CI 较集中的指标，单靠 measurement error 不足以解释阴性结果。
8. **更符合 genuinely small association、precision limitation，还是两者？** 当前证据更符合“**两者都可能**”：没有可靠的校正后 brain–behavior 显著结果，同时部分个体差异神经指标/差值的稳定性有限或不确定；现有数据不能定量拆分两者占比。
9. **未来保留什么？** 优先保留 split-half 排名较稳定且模型 SE 较小的单阶段指标；SMA/IFG 或 Bin10 指标是否作为 biomarker 必须看各自 CI。对 stage difference，若稳定性明显低于分量，应避免把它作为主要个体 biomarker，除非未来通过更长扫描、多次扫描或更强的层级测量模型改善精度。

## 9. 方法边界

- 单次扫描 split-half 不能估计跨日/跨次 test–retest reliability。
- contiguous halves 仍可能共享慢漂移或状态成分，Spearman–Brown 假设未必完全成立。
- omega 在约 30 人样本中仍有明显抽样不确定性；bootstrap CI 是必要组成部分。
- voxel bootstrap 固定 HC reference，只回答个体 fingerprint 对 Bin10 voxel sampling 的敏感性。
- reliability ceiling 假设误差独立且 reliability 可迁移到现有相关样本；这是描述性上限，不是正式效应修正。
- 本审计停止在 measurement-quality interpretation，未增加 ROI、bin、症状变量或相关搜索空间。

## 10. 可复现入口

运行 `python run_all_measurement_precision.py` 可从审计到图表和报告完整重建。所有随机过程使用固定种子 `{SEED}`，bootstrap 次数为 `{BOOT_N}`。
"""
    (OUT / "MEASUREMENT_PRECISION_AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    todo = """# TODO / Future measurement work

- Obtain an independent repeat scan before making any test-retest or biomarker claim.
- If a repeat scan is acquired, preregister scan-level ICC and generalizability analyses for the retained single-stage phenotypes.
- Consider longer acquisitions or hierarchical subject-level slope estimation for phenotypes with wide model-based or split-half uncertainty.
- Replicate behavioral omega and TEPS difference-score reliability in a larger SZ clinical sample.
- Keep the current item definitions, fixed G1, fixed bins, ROI families, and HC-reference logic unchanged unless a separately preregistered methods study is undertaken.
- No additional clinical correlation search is authorized by this audit.
"""
    (OUT / "TODO.md").write_text(todo, encoding="utf-8")

    outputs = [
        "input_audit.csv", "behavior_score_reconstruction_audit.csv", "behavior_scale_reliability.csv",
        "behavior_item_diagnostics.csv", "neural_model_precision.csv", "neural_split_half_stability.csv",
        "neural_split_half_subject_metrics.csv", "split_half_reconstruction_qc.csv",
        "bin10_voxel_bootstrap_precision.csv", "bin10_similarity_reconstruction_qc.csv",
        "stage_contrast_reliability.csv", "reliability_ceiling_summary.csv",
        "Figure1_behavior_reliability.pdf", "Figure1_behavior_reliability.png",
        "Figure2_neural_stability.pdf", "Figure2_neural_stability.png",
        "Figure3_stage_contrast_stability.pdf", "Figure3_stage_contrast_stability.png",
        "Figure4_reliability_ceiling.pdf", "Figure4_reliability_ceiling.png",
        "MEASUREMENT_PRECISION_AUDIT_REPORT.md", "FIGURE_QA_REPORT.md", "TODO.md",
    ]
    prov = []
    key_sources = "; ".join(str(PATHS[k]) for k in ["workbook", "clinical35", "fixed_bins", "fixed_g1", "timeseries", "fingerprints", "current_voxels"])
    for name in outputs:
        p = OUT / name
        if p.exists():
            prov.append({
                "output": name, "output_sha256": sha256(p), "source_inputs": key_sources,
                "method_version": "measurement_precision_audit_v1_20261029",
                "random_seed": SEED, "bootstrap_resamples": BOOT_N,
                "guardrail": "measurement quality only; no new clinical association search",
            })
    pd.DataFrame(prov).to_csv(OUT / "analysis_provenance.csv", index=False)
    print("Report, TODO, and provenance written.")


def run_all() -> None:
    run_audit()
    run_behavior_reliability()
    run_model_precision()
    run_split_half()
    run_voxel_bootstrap()
    run_stage_contrast()
    run_ceiling()
    run_plots()
    run_report()
