#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import norm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


# =============================================================================
# 0. PATHS
# =============================================================================

BASE = Path(
    os.path.join(os.environ["STRIATAL_HISTORICAL_ROOT"], "25_subject_level_stage_g1_model1")
)

INFILE = (
    BASE /
    "tables/subject_stage_G1_long.tsv"
)

OUT = BASE / "random_slope_model"

TAB = OUT / "tables"
QC = OUT / "qc"

OUT.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
QC.mkdir(exist_ok=True)


# =============================================================================
# 1. LOAD LONG TABLE
# =============================================================================

if not INFILE.exists():
    raise FileNotFoundError(INFILE)

df = pd.read_csv(
    INFILE,
    sep="\t"
)

required = {
    "Subject",
    "G1_bin",
    "G1c",
    "Stage",
    "FC",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )

print("=" * 100)
print("DATA")
print("=" * 100)

print(
    "Subjects     :",
    df["Subject"].nunique()
)

print(
    "Observations :",
    len(df)
)

print(
    "Stages       :",
    sorted(df["Stage"].unique())
)

print(
    "G1 bins      :",
    sorted(df["G1_bin"].unique())
)

print(
    "Missing FC   :",
    int(df["FC"].isna().sum())
)

if df["Subject"].nunique() != 375:
    raise RuntimeError(
        "Expected 375 subjects."
    )

if len(df) != 7500:
    raise RuntimeError(
        "Expected 7500 observations."
    )

if df["FC"].isna().any():
    raise RuntimeError(
        "NaN FC values detected."
    )


# =============================================================================
# 2. MODEL FORMULA
# =============================================================================

formula = (
    'FC ~ '
    'C(Stage, Treatment(reference="Outcome")) '
    '* G1c'
)

print("\n" + "=" * 100)
print("MODELS")
print("=" * 100)

print(
    "Model RI:"
)

print(
    "FC ~ Stage * G1c + (1 | Subject)"
)

print()

print(
    "Model RS:"
)

print(
    "FC ~ Stage * G1c + "
    "(1 + G1c | Subject)"
)

print()

print(
    "Reference stage = Outcome"
)

print(
    "G1c = G1_bin - 5.5"
)


# =============================================================================
# 3. HELPER: FIT MODEL WITH MULTIPLE OPTIMIZERS
# =============================================================================

def fit_mixed_model(
    model,
    label
):

    last_error = None

    for method in [
        "lbfgs",
        "bfgs",
        "powell"
    ]:

        print(
            f"{label}: trying optimizer "
            f"{method} ..."
        )

        try:

            fit = model.fit(
                reml=False,
                method=method,
                maxiter=5000,
                disp=False
            )

            print(
                f"{label}: optimizer = "
                f"{method}"
            )

            print(
                f"{label}: converged = "
                f"{fit.converged}"
            )

            if fit.converged:
                return fit, method

            last_error = RuntimeError(
                f"{method}: not converged"
            )

        except Exception as e:

            print(
                f"{label}: {method} failed:"
            )

            print(
                repr(e)
            )

            last_error = e

    raise RuntimeError(
        f"{label}: all optimizers failed. "
        f"Last error = {last_error}"
    )


# =============================================================================
# 4. REFIT RANDOM-INTERCEPT MODEL
#
# Same data + same ML method so AIC/BIC can be directly compared.
# =============================================================================

model_ri = smf.mixedlm(
    formula,
    data=df,
    groups=df["Subject"],
    re_formula="1"
)

fit_ri, optimizer_ri = fit_mixed_model(
    model_ri,
    "RI"
)


# =============================================================================
# 5. RANDOM-INTERCEPT + RANDOM-G1-SLOPE MODEL
# =============================================================================

model_rs = smf.mixedlm(
    formula,
    data=df,
    groups=df["Subject"],
    re_formula="1 + G1c"
)

fit_rs, optimizer_rs = fit_mixed_model(
    model_rs,
    "RS"
)


# =============================================================================
# 6. IDENTIFY FIXED-EFFECT TERMS
# =============================================================================

fe = fit_rs.fe_params

interaction_terms = [
    t
    for t in fe.index
    if ":G1c" in t
]

stage_terms = [
    t
    for t in fe.index
    if (
        "C(Stage" in t
        and ":G1c" not in t
    )
]

if len(interaction_terms) != 1:
    raise RuntimeError(
        "Could not uniquely identify "
        "Stage x G1 term."
    )

if len(stage_terms) != 1:
    raise RuntimeError(
        "Could not uniquely identify "
        "Stage term."
    )

interaction_term = interaction_terms[0]
stage_term = stage_terms[0]


# =============================================================================
# 7. FIXED EFFECT TABLE
# =============================================================================

bse = fit_rs.bse_fe

pvals = fit_rs.pvalues.loc[
    fe.index
]

ci = fit_rs.conf_int().loc[
    fe.index
]

fixed = pd.DataFrame({
    "term": fe.index,
    "beta": fe.values,
    "SE": bse.values,
    "CI95_low": ci[0].values,
    "CI95_high": ci[1].values,
    "p": pvals.values,
})


def interpretation(term):

    if term == "Intercept":
        return (
            "Outcome mean FC at G1 midpoint"
        )

    if term == "G1c":
        return (
            "Outcome G1 slope"
        )

    if term == stage_term:
        return (
            "Anticipation minus Outcome "
            "at G1 midpoint"
        )

    if term == interaction_term:
        return (
            "PRIMARY: Anticipation slope "
            "minus Outcome slope "
            "(Stage x G1)"
        )

    return ""


fixed["interpretation"] = (
    fixed["term"]
    .apply(interpretation)
)

fixed.to_csv(
    TAB / "RANDOM_SLOPE_fixed_effects.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 8. SIMPLE SLOPES
#
# Outcome:
#     beta_G1
#
# Anticipation:
#     beta_G1 + beta_Stage:G1
#
# Calculate proper SE / CI / p for both.
# =============================================================================

cov_all = fit_rs.cov_params()

cov_fe = cov_all.loc[
    fe.index,
    fe.index
]

b_g1 = fe["G1c"]

b_int = fe[interaction_term]


# -----------------------------------------------------------------------------
# Outcome slope
# -----------------------------------------------------------------------------

outcome_beta = b_g1

outcome_var = cov_fe.loc[
    "G1c",
    "G1c"
]

outcome_se = np.sqrt(
    outcome_var
)

outcome_z = (
    outcome_beta /
    outcome_se
)

outcome_p = (
    2 *
    norm.sf(
        abs(outcome_z)
    )
)

outcome_ci_low = (
    outcome_beta -
    1.96 * outcome_se
)

outcome_ci_high = (
    outcome_beta +
    1.96 * outcome_se
)


# -----------------------------------------------------------------------------
# Anticipation slope
# -----------------------------------------------------------------------------

anti_beta = (
    b_g1 +
    b_int
)

anti_var = (
    cov_fe.loc[
        "G1c",
        "G1c"
    ]
    +
    cov_fe.loc[
        interaction_term,
        interaction_term
    ]
    +
    2 *
    cov_fe.loc[
        "G1c",
        interaction_term
    ]
)

anti_se = np.sqrt(
    anti_var
)

anti_z = (
    anti_beta /
    anti_se
)

anti_p = (
    2 *
    norm.sf(
        abs(anti_z)
    )
)

anti_ci_low = (
    anti_beta -
    1.96 * anti_se
)

anti_ci_high = (
    anti_beta +
    1.96 * anti_se
)


# -----------------------------------------------------------------------------
# Interaction itself
# -----------------------------------------------------------------------------

interaction_beta = (
    b_int
)

interaction_se = (
    fit_rs.bse_fe[
        interaction_term
    ]
)

interaction_p = (
    fit_rs.pvalues[
        interaction_term
    ]
)

interaction_ci = (
    fit_rs.conf_int()
    .loc[
        interaction_term
    ]
    .to_numpy()
)


simple_slopes = pd.DataFrame([
    {
        "Stage": "Outcome",
        "slope": outcome_beta,
        "SE": outcome_se,
        "CI95_low": outcome_ci_low,
        "CI95_high": outcome_ci_high,
        "z": outcome_z,
        "p": outcome_p,
    },
    {
        "Stage": "Anticipation",
        "slope": anti_beta,
        "SE": anti_se,
        "CI95_low": anti_ci_low,
        "CI95_high": anti_ci_high,
        "z": anti_z,
        "p": anti_p,
    }
])

simple_slopes.to_csv(
    TAB / "RANDOM_SLOPE_stage_simple_slopes.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 9. RANDOM-EFFECT STRUCTURE
# =============================================================================

cov_re = np.asarray(
    fit_rs.cov_re,
    dtype=float
)

if cov_re.shape != (2, 2):
    raise RuntimeError(
        f"Unexpected random covariance "
        f"shape: {cov_re.shape}"
    )

var_intercept = (
    cov_re[0, 0]
)

var_g1 = (
    cov_re[1, 1]
)

cov_int_g1 = (
    cov_re[0, 1]
)

sd_intercept = (
    np.sqrt(
        max(
            var_intercept,
            0
        )
    )
)

sd_g1 = (
    np.sqrt(
        max(
            var_g1,
            0
        )
    )
)

if (
    sd_intercept > 0
    and
    sd_g1 > 0
):
    corr_int_g1 = (
        cov_int_g1 /
        (
            sd_intercept *
            sd_g1
        )
    )
else:
    corr_int_g1 = np.nan


eigvals = np.linalg.eigvalsh(
    cov_re
)

min_eig = float(
    np.min(eigvals)
)

max_eig = float(
    np.max(eigvals)
)

if max_eig > 0:
    eigen_ratio = (
        min_eig /
        max_eig
    )
else:
    eigen_ratio = np.nan


random_effects = pd.DataFrame([{
    "random_intercept_variance":
        var_intercept,

    "random_intercept_SD":
        sd_intercept,

    "random_G1_slope_variance":
        var_g1,

    "random_G1_slope_SD":
        sd_g1,

    "intercept_G1_covariance":
        cov_int_g1,

    "intercept_G1_correlation":
        corr_int_g1,

    "residual_variance":
        fit_rs.scale,

    "residual_SD":
        np.sqrt(fit_rs.scale),

    "random_cov_min_eigenvalue":
        min_eig,

    "random_cov_max_eigenvalue":
        max_eig,

    "min_max_eigen_ratio":
        eigen_ratio,
}])

random_effects.to_csv(
    TAB / "RANDOM_SLOPE_random_effects.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 10. MODEL COMPARISON
#
# Both are fit with ML and identical fixed effects.
# AIC/BIC are therefore comparable.
# =============================================================================

comparison = pd.DataFrame([
    {
        "model":
            "Random intercept",

        "random_structure":
            "(1 | Subject)",

        "optimizer":
            optimizer_ri,

        "converged":
            bool(
                fit_ri.converged
            ),

        "logLik":
            fit_ri.llf,

        "AIC":
            fit_ri.aic,

        "BIC":
            fit_ri.bic,

        "n_parameters":
            len(
                fit_ri.params
            ),
    },
    {
        "model":
            "Random intercept + G1 slope",

        "random_structure":
            "(1 + G1c | Subject)",

        "optimizer":
            optimizer_rs,

        "converged":
            bool(
                fit_rs.converged
            ),

        "logLik":
            fit_rs.llf,

        "AIC":
            fit_rs.aic,

        "BIC":
            fit_rs.bic,

        "n_parameters":
            len(
                fit_rs.params
            ),
    },
])

comparison.to_csv(
    TAB / "RI_vs_RS_model_comparison.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 11. DIRECT COMPARISON OF STAGE x G1 RESULT
# =============================================================================

ri_fe = fit_ri.fe_params

ri_int_terms = [
    t
    for t in ri_fe.index
    if ":G1c" in t
]

if len(ri_int_terms) != 1:
    raise RuntimeError(
        "Could not identify RI interaction."
    )

ri_int = ri_int_terms[0]

interaction_compare = pd.DataFrame([
    {
        "model":
            "Random intercept",

        "interaction_beta":
            fit_ri.fe_params[
                ri_int
            ],

        "interaction_SE":
            fit_ri.bse_fe[
                ri_int
            ],

        "CI95_low":
            fit_ri.conf_int()
            .loc[
                ri_int,
                0
            ],

        "CI95_high":
            fit_ri.conf_int()
            .loc[
                ri_int,
                1
            ],

        "p":
            fit_ri.pvalues[
                ri_int
            ],
    },
    {
        "model":
            "Random intercept + G1 slope",

        "interaction_beta":
            interaction_beta,

        "interaction_SE":
            interaction_se,

        "CI95_low":
            interaction_ci[0],

        "CI95_high":
            interaction_ci[1],

        "p":
            interaction_p,
    },
])

interaction_compare.to_csv(
    TAB / "Stage_x_G1_RI_vs_RS.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 12. KEY RESULT
# =============================================================================

key = pd.DataFrame([{
    "N_subjects":
        df["Subject"].nunique(),

    "N_observations":
        len(df),

    "Outcome_slope":
        outcome_beta,

    "Outcome_slope_SE":
        outcome_se,

    "Outcome_slope_CI95_low":
        outcome_ci_low,

    "Outcome_slope_CI95_high":
        outcome_ci_high,

    "Outcome_slope_p":
        outcome_p,

    "Anticipation_slope":
        anti_beta,

    "Anticipation_slope_SE":
        anti_se,

    "Anticipation_slope_CI95_low":
        anti_ci_low,

    "Anticipation_slope_CI95_high":
        anti_ci_high,

    "Anticipation_slope_p":
        anti_p,

    "Anti_minus_Outcome_slope_difference":
        interaction_beta,

    "interaction_SE":
        interaction_se,

    "interaction_CI95_low":
        interaction_ci[0],

    "interaction_CI95_high":
        interaction_ci[1],

    "Stage_x_G1_p":
        interaction_p,

    "random_G1_slope_variance":
        var_g1,

    "random_G1_slope_SD":
        sd_g1,

    "AIC_random_intercept":
        fit_ri.aic,

    "AIC_random_slope":
        fit_rs.aic,

    "BIC_random_intercept":
        fit_ri.bic,

    "BIC_random_slope":
        fit_rs.bic,

    "optimizer":
        optimizer_rs,

    "converged":
        bool(
            fit_rs.converged
        ),
}])

key.to_csv(
    TAB / "RANDOM_SLOPE_KEY_RESULT.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 13. SAVE FULL SUMMARY
# =============================================================================

summary_text = []

summary_text.append(
    "RANDOM-SLOPE MODEL"
)

summary_text.append(
    "=" * 100
)

summary_text.append(
    "FC ~ Stage * G1c + "
    "(1 + G1c | Subject)"
)

summary_text.append(
    "Outcome = reference stage"
)

summary_text.append(
    "G1c = G1_bin - 5.5"
)

summary_text.append(
    f"N subjects = "
    f"{df['Subject'].nunique()}"
)

summary_text.append(
    f"N observations = "
    f"{len(df)}"
)

summary_text.append(
    f"Optimizer = "
    f"{optimizer_rs}"
)

summary_text.append(
    f"Converged = "
    f"{fit_rs.converged}"
)

summary_text.append("")

summary_text.append(
    fit_rs.summary().as_text()
)

(
    OUT /
    "RANDOM_SLOPE_full_summary.txt"
).write_text(
    "\n".join(
        summary_text
    ),
    encoding="utf-8"
)


# =============================================================================
# 14. FINAL CONSOLE REPORT
# =============================================================================

print("\n" + "=" * 100)
print("RANDOM-SLOPE MODEL: FIXED EFFECTS")
print("=" * 100)

print(
    fixed.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("SIMPLE SLOPES")
print("=" * 100)

print(
    simple_slopes.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("PRIMARY STAGE x G1 INTERACTION")
print("=" * 100)

print(
    f"Outcome slope      = "
    f"{outcome_beta:.10f}"
)

print(
    f"Anticipation slope = "
    f"{anti_beta:.10f}"
)

print(
    f"Anti - Outcome     = "
    f"{interaction_beta:.10f}"
)

print(
    f"SE                 = "
    f"{interaction_se:.10f}"
)

print(
    f"95% CI             = "
    f"[{interaction_ci[0]:.10f}, "
    f"{interaction_ci[1]:.10f}]"
)

print(
    f"p                  = "
    f"{interaction_p:.12g}"
)


print("\n" + "=" * 100)
print("RANDOM EFFECTS")
print("=" * 100)

print(
    random_effects.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("RI vs RANDOM-SLOPE MODEL")
print("=" * 100)

print(
    comparison.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("STAGE x G1: BEFORE vs AFTER RANDOM SLOPE")
print("=" * 100)

print(
    interaction_compare.to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("INTERPRETATION CHECK")
print("=" * 100)

if not fit_rs.converged:

    print(
        "WARNING: random-slope model "
        "did not converge."
    )

else:

    print(
        "Model converged."
    )


if var_g1 <= 1e-10:

    print(
        "WARNING: random G1 slope "
        "variance is approximately zero."
    )

else:

    print(
        "Random G1 slope variance > 0."
    )


if (
    np.isfinite(
        eigen_ratio
    )
    and
    eigen_ratio < 1e-6
):

    print(
        "WARNING: random-effects "
        "covariance may be near singular."
    )

else:

    print(
        "No obvious near-singularity "
        "from eigenvalue ratio."
    )


if interaction_p < 0.05:

    print(
        "PRIMARY RESULT: "
        "Stage x G1 remains significant "
        "after allowing subject-specific "
        "G1 slopes."
    )

else:

    print(
        "PRIMARY RESULT: "
        "Stage x G1 is no longer "
        "significant in the random-slope "
        "model."
    )


print("\nOutputs:")
print(
    TAB /
    "RANDOM_SLOPE_KEY_RESULT.tsv"
)

print(
    TAB /
    "RANDOM_SLOPE_fixed_effects.tsv"
)

print(
    TAB /
    "RANDOM_SLOPE_stage_simple_slopes.tsv"
)

print(
    TAB /
    "RANDOM_SLOPE_random_effects.tsv"
)

print(
    TAB /
    "RI_vs_RS_model_comparison.tsv"
)

print(
    TAB /
    "Stage_x_G1_RI_vs_RS.tsv"
)

print(
    OUT /
    "RANDOM_SLOPE_full_summary.txt"
)

print("\nDONE.")
