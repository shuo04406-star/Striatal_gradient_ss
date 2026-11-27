#!/usr/bin/env python3
"""Independent, no-overwrite verification for the 2026 conference abstract audit.

This script reads frozen inputs only and writes new audit outputs beside itself.
It refits continuous Models A/B/C and independently reconstructs the current-
Bin10 additive Pearson-r subfamily analysis from all 64 FC arrays.
"""

from __future__ import annotations

import os
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2, norm
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


PROJECT = Path(os.environ["STRIATAL_PROJECT_ROOT"])
REPO = Path(os.environ["STRIATAL_RELEASE_ROOT"])
AUDIT = Path(os.environ["STRIATAL_OUTPUT_ROOT"])
OUT = AUDIT / "reproduction_outputs" / "independent_core"

STAGE_INPUT = PROJECT / "CURRENT64_CANONICAL_RERUN" / "06_REWARD_STAGE" / "subject_bin_stage_FC_long.tsv"
COV_INPUT = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "covariates" / "current64_covariates.tsv"
ROI_META = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "reward_stage_mapping" / "roi_metadata_stage_subfamily_v2.tsv"
FC_ROOT = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "fc_fingerprints_2608x65"
MAPPING = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "voxel_order" / "row_mapping_2847_to_2608.npy"
DEF_A_G1 = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "canonical_G1" / "375_reference_G1_manuscript_2608.npy"
DEF_A_BINS = PROJECT / "CURRENT64_CANONICAL_RERUN" / "01_FROZEN_INPUTS" / "canonical_bins" / "375_reference_bins_manuscript_2608.npy"
GROUP_GRADIENTS = PROJECT / "SERVER_RECOVERED_CODE_REVIEW" / "STAGE2_SERVER_REFERENCE" / "server_reference" / "group_gradients.npy"
GROUP_BINS = PROJECT / "SERVER_RECOVERED_CODE_REVIEW" / "STAGE2_SERVER_REFERENCE" / "server_reference" / "voxel_G1_and_10bin.tsv"
CURRENT_BINS = Path(os.environ["STRIATAL_CURRENT_BIN_TABLE"])
REFERENCE375_MODEL_INPUT = PROJECT / "02_main_375" / "25_subject_level_stage_g1_model1" / "tables" / "subject_stage_G1_long.tsv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = frame.to_csv(sep="\t", index=False, float_format="%.17g", lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8").replace("\r\n", "\n") == content.replace("\r\n", "\n"):
            return
        raise FileExistsError(f"Refusing to replace differing audit output {path}")
    path.write_text(content, encoding="utf-8", newline="")


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8").replace("\r\n", "\n") == content.replace("\r\n", "\n"):
            return
        raise FileExistsError(f"Refusing to replace differing audit output {path}")
    path.write_text(content, encoding="utf-8", newline="")


def fit_mixed(formula: str, data: pd.DataFrame, label: str):
    model = smf.mixedlm(formula, data=data, groups=data["subject"], re_formula="1 + G1_centered")
    attempts = []
    for method in ("lbfgs", "powell", "cg"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = model.fit(reml=False, method=method, maxiter=5000, disp=False)
                attempts.append(
                    {
                        "model": label,
                        "optimizer": method,
                        "converged": bool(result.converged),
                        "logLik": float(result.llf),
                        "warnings": " | ".join(str(w.message) for w in caught),
                        "error": "",
                    }
                )
                if result.converged:
                    return result, pd.DataFrame(attempts)
            except Exception as exc:
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
    raise RuntimeError(f"{label} did not converge: {attempts}")


def normalize_stage_input() -> pd.DataFrame:
    raw = pd.read_csv(STAGE_INPUT, sep="\t")
    required = {"Subject", "Group", "Bin", "G1_bin_center", "Stage", "FC", "Age", "Sex", "MeanFD"}
    if not required.issubset(raw.columns):
        raise RuntimeError(f"Missing stage columns: {sorted(required - set(raw.columns))}")
    if len(raw) != 1280 or raw["Subject"].nunique() != 64:
        raise RuntimeError("Expected 1,280 rows and 64 subjects")
    if raw.duplicated(["Subject", "Bin", "Stage"]).any():
        raise RuntimeError("Duplicate Subject x Bin x Stage keys")
    if not np.isfinite(raw["FC"]).all():
        raise RuntimeError("Non-finite FC values")
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
    if people["group"].value_counts().to_dict() != {"HC": 32, "SZ": 32}:
        raise RuntimeError("Group balance check failed")
    if sorted(df["G1_centered"].unique().tolist()) != [x - 5.5 for x in range(1, 11)]:
        raise RuntimeError("G1_centered is not Bin - 5.5")
    df["age_c"] = df["age"] - people["age"].mean()
    df["meanFD_c"] = df["meanFD"] - people["meanFD"].mean()
    male = df["sex"].map({"Female": 0.0, "Male": 1.0}).astype(float)
    male_prop = float(people["sex"].map({"Female": 0.0, "Male": 1.0}).astype(float).mean())
    df["sex_c"] = male - male_prop
    return df


def find_term(names: list[str], required: tuple[str, ...], excluded: tuple[str, ...] = ()) -> str:
    hits = [
        name
        for name in names
        if all(token in name for token in required) and not any(token in name for token in excluded)
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Could not uniquely find term {required}: {hits}")
    return hits[0]


def linear_contrast(result, weights: dict[str, float], label: str, model: str) -> dict[str, object]:
    names = list(result.fe_params.index)
    vector = np.zeros(len(names), dtype=float)
    for term, value in weights.items():
        vector[names.index(term)] = value
    beta = np.asarray(result.fe_params, float)
    cov = np.asarray(result.cov_params().loc[names, names], float)
    estimate = float(vector @ beta)
    se = float(np.sqrt(max(float(vector @ cov @ vector.T), 0.0)))
    z = estimate / se
    return {
        "model": model,
        "record_type": "contrast",
        "term_or_contrast": label,
        "estimate": estimate,
        "SE": se,
        "CI95_low": estimate - 1.96 * se,
        "CI95_high": estimate + 1.96 * se,
        "statistic": z,
        "statistic_type": "Wald z",
        "df": np.nan,
        "p": float(2 * norm.sf(abs(z))),
        "formula": result.model.formula,
        "random_effects": "1 + G1_centered | subject",
        "estimation": "ML (reml=False)",
    }


def reproduce_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_stage_input()
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
    records = []
    attempts = []
    for label, (full_formula, reduced_formula) in formulas.items():
        full, a1 = fit_mixed(full_formula, df, f"Model_{label}_full")
        reduced, a0 = fit_mixed(reduced_formula, df, f"Model_{label}_reduced")
        attempts.extend([a1, a0])
        names = list(full.fe_params.index)
        term_g1 = "G1_centered"
        term_stage_g1 = "stage[T.Anticipation]:G1_centered"
        term_group_g1 = "group[T.SZ]:G1_centered"
        term_three = "group[T.SZ]:stage[T.Anticipation]:G1_centered"
        for required_term in (term_stage_g1, term_group_g1, term_three):
            if required_term not in names:
                raise RuntimeError(f"Missing expected fixed-effect term: {required_term}")
        contrasts = [
            ("HC Outcome slope", {term_g1: 1}),
            ("HC Anticipation slope", {term_g1: 1, term_stage_g1: 1}),
            ("SZ Outcome slope", {term_g1: 1, term_group_g1: 1}),
            ("SZ Anticipation slope", {term_g1: 1, term_group_g1: 1, term_stage_g1: 1, term_three: 1}),
            ("HC Anticipation minus Outcome slope difference", {term_stage_g1: 1}),
            ("SZ Anticipation minus Outcome slope difference", {term_stage_g1: 1, term_three: 1}),
            ("SZ minus HC difference in stage-gradient differentiation", {term_three: 1}),
        ]
        for contrast_label, weights in contrasts:
            records.append(linear_contrast(full, weights, contrast_label, f"Model_{label}"))
        lr = max(0.0, 2 * (float(full.llf) - float(reduced.llf)))
        records.append(
            {
                "model": f"Model_{label}",
                "record_type": "full_reduced_comparison",
                "term_or_contrast": "Group x Stage x G1",
                "estimate": np.nan,
                "SE": np.nan,
                "CI95_low": np.nan,
                "CI95_high": np.nan,
                "statistic": lr,
                "statistic_type": "Likelihood-ratio chi-square",
                "df": 1,
                "p": float(chi2.sf(lr, 1)),
                "formula": full_formula,
                "random_effects": "1 + G1_centered | subject",
                "estimation": "ML (reml=False)",
            }
        )
    return pd.DataFrame(records), pd.concat(attempts, ignore_index=True)


def prepare_covariates() -> pd.DataFrame:
    cov = pd.read_csv(COV_INPUT, sep="\t").rename(
        columns={"Subject": "subject", "Group": "group", "Age": "age", "Sex": "sex", "MeanFD": "meanFD"}
    )
    cov["subject"] = cov["subject"].astype(str)
    cov["group"] = cov["group"].astype(str).str.upper()
    cov["sex_num"] = cov["sex"].astype(str).str.lower().map({"female": 0, "f": 0, "male": 1, "m": 1})
    cov["group_sz"] = (cov["group"] == "SZ").astype(int)
    cov["age_c"] = cov["age"] - cov["age"].mean()
    cov["sex_c"] = cov["sex_num"] - cov["sex_num"].mean()
    cov["meanFD_c"] = cov["meanFD"] - cov["meanFD"].mean()
    if len(cov) != 64 or cov["subject"].nunique() != 64 or cov["group"].value_counts().to_dict() != {"HC": 32, "SZ": 32}:
        raise RuntimeError("Covariate QC failed")
    return cov


def reproduce_subfamilies() -> tuple[pd.DataFrame, dict]:
    cov = prepare_covariates()
    meta = pd.read_csv(ROI_META, sep="\t")
    if meta["stage"].value_counts().to_dict() != {"Anticipation": 43, "Outcome": 19, "Not_assigned": 3}:
        raise RuntimeError("ROI stage mapping changed")
    current = pd.read_csv(CURRENT_BINS, sep="\t")["G1_bin_current"].astype(int).to_numpy()
    mapping = np.load(MAPPING).reshape(-1).astype(int)
    current_common = current[mapping]
    if len(current_common) != 2608 or int(np.sum(current_common == 10)) != 285:
        raise RuntimeError("Current-bin mapping QC failed")
    arrays = []
    input_hashes = []
    for subject in cov["subject"]:
        path = FC_ROOT / f"{subject}_fingerprint_z_2608x65.npy"
        array = np.asarray(np.load(path), float)
        if array.shape == (65, 2608):
            array = array.T
        if array.shape != (2608, 65) or not np.isfinite(array).all():
            raise RuntimeError(f"Bad FC array: {path}")
        arrays.append(array)
        input_hashes.append({"subject": subject, "path": str(path), "sha256": sha256(path)})
    write_tsv(pd.DataFrame(input_hashes), OUT / "fc_input_hashes.tsv")
    fc = np.stack(arrays)
    fingerprints = np.empty((64, 10, 65), dtype=float)
    for bin_number in range(1, 11):
        fingerprints[:, bin_number - 1, :] = fc[:, current_common == bin_number, :].mean(axis=1)
    hc_idx = np.flatnonzero(cov["group"].to_numpy() == "HC")
    hc_mean = fingerprints[hc_idx].mean(axis=0)
    subject_rows = []
    roi_rows = []
    for i, row in cov.iterrows():
        template = fingerprints[hc_idx[hc_idx != i], 9, :].mean(axis=0) if row.group == "HC" else hc_mean[9]
        values = fingerprints[i, 9]
        z_values = (values - values.mean()) / values.std(ddof=1)
        z_template = (template - template.mean()) / template.std(ddof=1)
        contributions = z_values * z_template / 64.0
        direct_r = float(np.corrcoef(values, template)[0, 1])
        if abs(float(contributions.sum()) - direct_r) > 1e-12:
            raise RuntimeError("Pearson-r contribution identity failed")
        subject_rows.append(
            {
                "subject": row.subject,
                "group": row.group,
                "group_sz": row.group_sz,
                "profile_r": direct_r,
                "age_c": row.age_c,
                "sex_c": row.sex_c,
                "meanFD_c": row.meanFD_c,
            }
        )
        for roi_index, contribution in enumerate(contributions):
            roi_rows.append(
                {
                    "subject": row.subject,
                    "group": row.group,
                    "group_sz": row.group_sz,
                    "age_c": row.age_c,
                    "sex_c": row.sex_c,
                    "meanFD_c": row.meanFD_c,
                    "roi_index0": roi_index,
                    "r_contribution": contribution,
                }
            )
    subjects = pd.DataFrame(subject_rows)
    roi = pd.DataFrame(roi_rows).merge(meta[["roi_index0", "subfamily", "stage"]], on="roi_index0", validate="many_to_one")
    aggregate = (
        roi.groupby(["subject", "group", "group_sz", "age_c", "sex_c", "meanFD_c", "subfamily", "stage"], as_index=False)
        .agg(contribution_sum=("r_contribution", "sum"), N_ROIs=("roi_index0", "nunique"))
    )
    total_model = smf.ols("profile_r ~ group_sz + age_c + sex_c + meanFD_c", subjects).fit(cov_type="HC3")
    total_beta = float(total_model.params["group_sz"])
    results = []
    for (subfamily, stage), subset in aggregate.groupby(["subfamily", "stage"], sort=True):
        model = smf.ols("contribution_sum ~ group_sz + age_c + sex_c + meanFD_c", subset).fit(cov_type="HC3")
        beta = float(model.params["group_sz"])
        se = float(model.bse["group_sz"])
        results.append(
            {
                "subfamily": subfamily,
                "stage": stage,
                "N_ROIs": int(subset["N_ROIs"].iloc[0]),
                "beta_SZ_minus_HC": beta,
                "SE_HC3": se,
                "CI95_low": beta - 1.96 * se,
                "CI95_high": beta + 1.96 * se,
                "p_raw": float(model.pvalues["group_sz"]),
            }
        )
    out = pd.DataFrame(results)
    assigned = out["stage"] != "Not_assigned"
    out["q_FDR_13_assigned"] = np.nan
    out.loc[assigned, "q_FDR_13_assigned"] = multipletests(out.loc[assigned, "p_raw"].to_numpy(), method="fdr_bh")[1]
    out["signed_share_of_total_pct"] = 100 * out["beta_SZ_minus_HC"] / total_beta
    out = out.sort_values("beta_SZ_minus_HC").reset_index(drop=True)
    qc = {
        "current_bin_input_sha256": sha256(CURRENT_BINS),
        "mapping_sha256": sha256(MAPPING),
        "current_full_bin_counts": pd.Series(current).value_counts().sort_index().to_dict(),
        "current_common_bin_counts": pd.Series(current_common).value_counts().sort_index().to_dict(),
        "bin10_voxels": int(np.sum(current_common == 10)),
        "total_profile_r_group_beta": total_beta,
        "subfamily_beta_sum": float(out["beta_SZ_minus_HC"].sum()),
        "subfamily_sum_error": abs(float(out["beta_SZ_minus_HC"].sum()) - total_beta),
        "assigned_fdr_tests": int(assigned.sum()),
    }
    return out, qc


def input_qc() -> dict:
    stage = pd.read_csv(STAGE_INPUT, sep="\t")
    roi = pd.read_csv(ROI_META, sep="\t")
    gradients = np.load(GROUP_GRADIENTS)
    group_bins = pd.read_csv(GROUP_BINS, sep="\t")
    mapping = np.load(MAPPING).reshape(-1).astype(int)
    def_a_g1 = np.load(DEF_A_G1)
    def_a_bins = np.load(DEF_A_BINS)
    current = pd.read_csv(CURRENT_BINS, sep="\t")
    model375 = pd.read_csv(REFERENCE375_MODEL_INPUT, sep="\t")
    mapped_current_g1 = gradients[mapping, 0]
    sign_relation = min(
        float(np.max(np.abs(mapped_current_g1 - def_a_g1))),
        float(np.max(np.abs(mapped_current_g1 + def_a_g1))),
    )
    return {
        "paths": {
            "stage_input": str(STAGE_INPUT),
            "roi_metadata": str(ROI_META),
            "reference_gradients": str(GROUP_GRADIENTS),
            "reference_bins": str(GROUP_BINS),
            "definition_a_g1": str(DEF_A_G1),
            "definition_a_bins": str(DEF_A_BINS),
            "current_bins": str(CURRENT_BINS),
            "reference375_model_input": str(REFERENCE375_MODEL_INPUT),
        },
        "hashes": {str(path): sha256(path) for path in [STAGE_INPUT, COV_INPUT, ROI_META, MAPPING, DEF_A_G1, DEF_A_BINS, GROUP_GRADIENTS, GROUP_BINS, CURRENT_BINS, REFERENCE375_MODEL_INPUT]},
        "reference375_model_input": {
            "rows": len(model375),
            "subjects": int(model375["Subject"].nunique()),
            "bins": sorted(int(x) for x in model375["G1_bin"].unique()),
            "stages": sorted(str(x) for x in model375["Stage"].unique()),
        },
        "reference_gradient": {
            "shape": list(gradients.shape),
            "g1_min": float(gradients[:, 0].min()),
            "g1_max": float(gradients[:, 0].max()),
            "group_bin_rows": len(group_bins),
            "group_bin_counts": group_bins["G1_bin"].value_counts().sort_index().to_dict(),
        },
        "clinical_common_space": {
            "mapping_rows": len(mapping),
            "mapping_unique": int(np.unique(mapping).size),
            "definition_a_g1_rows": len(def_a_g1),
            "definition_a_g1_min": float(def_a_g1.min()),
            "definition_a_g1_max": float(def_a_g1.max()),
            "mapped_reference_sign_equivalence_max_abs_error": sign_relation,
            "definition_a_bin_counts": pd.Series(def_a_bins).value_counts().sort_index().to_dict(),
            "current_table_rows": len(current),
        },
        "clinical_model_input": {
            "rows": len(stage),
            "subjects": int(stage["Subject"].nunique()),
            "groups": stage.drop_duplicates("Subject")["Group"].value_counts().to_dict(),
            "stages": sorted(str(x) for x in stage["Stage"].unique()),
            "g1_predictor_values": sorted(float(x) for x in stage["G1_bin_center"].unique()),
            "duplicate_keys": int(stage.duplicated(["Subject", "Bin", "Stage"]).sum()),
        },
        "roi_mapping": {
            "rows": len(roi),
            "stage_counts": roi["stage"].value_counts().to_dict(),
            "subfamily_stage_counts": {
                f"{stage}|{subfamily}": int(count)
                for (stage, subfamily), count in roi.groupby(["stage", "subfamily"]).size().items()
            },
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    models, attempts = reproduce_models()
    subfamilies, subfamily_qc = reproduce_subfamilies()
    qc = input_qc()
    qc["software"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
    }
    qc["subfamily_qc"] = subfamily_qc
    write_tsv(models, OUT / "model_abc_independent_refit.tsv")
    write_tsv(attempts, OUT / "model_optimizer_attempts.tsv")
    write_tsv(subfamilies, OUT / "subfamily_independent_reconstruction.tsv")
    write_json(qc, OUT / "input_qc_and_provenance.json")
    print(json.dumps({"status": "SUCCESS", "output_dir": str(OUT), "model_rows": len(models), "subfamily_rows": len(subfamilies)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
