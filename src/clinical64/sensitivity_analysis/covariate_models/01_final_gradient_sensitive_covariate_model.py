#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

BASE = Path(os.environ["STRIATAL_HISTORICAL_ROOT"])

IN = (
    BASE /
    "20_current64_reward_stage_fc" /
    "01_fixed375_stage_profiles_manuscript"
)

COVIN = (
    BASE /
    "20_current64_reward_stage_fc" /
    "05_covariate_adjusted"
)

OUT = (
    BASE /
    "20_current64_reward_stage_fc" /
    "06_gradient_sensitive_covariates"
)

OUT.mkdir(parents=True, exist_ok=True)

STAGE_PATH = IN / "subject_bin_stage_FC_long.tsv"
COV_PATH = COVIN / "covariate_master_64.tsv"

# =============================================================================
# HELPERS
# =============================================================================

def fit_mixed(formula, data, label):

    print()
    print(f"Fitting {label} ...")

    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["subject"],
        re_formula="1 + G1_centered"
    )

    attempts = []

    for method in ["lbfgs", "powell", "cg"]:

        try:

            result = model.fit(
                reml=False,
                method=method,
                maxiter=5000,
                disp=False
            )

            attempts.append(
                (method, result.converged, result.llf)
            )

            if result.converged:

                print(
                    f"Converged with {method} | "
                    f"logLik={result.llf:.10f}"
                )

                return result

        except Exception as e:

            attempts.append(
                (method, False, str(e))
            )

    print("FIT ATTEMPTS:")

    for x in attempts:
        print(x)

    raise RuntimeError(
        f"{label} failed to converge."
    )


def find_term(
    term_names,
    required_tokens,
    forbidden_tokens=None
):

    forbidden_tokens = forbidden_tokens or []

    matches = []

    for term in term_names:

        if all(
            token in term
            for token in required_tokens
        ):

            if not any(
                token in term
                for token in forbidden_tokens
            ):
                matches.append(term)

    if len(matches) != 1:

        print()
        print("AVAILABLE TERMS:")

        for t in term_names:
            print(" ", t)

        raise RuntimeError(
            f"Could not uniquely identify term.\n"
            f"required={required_tokens}\n"
            f"forbidden={forbidden_tokens}\n"
            f"matches={matches}"
        )

    return matches[0]


# =============================================================================
# STEP 1. LOAD DATA
# =============================================================================

print("=" * 110)
print("STEP 1. LOAD STAGE DATA AND VALIDATED COVARIATES")
print("=" * 110)

stage = pd.read_csv(
    STAGE_PATH,
    sep="\t"
)

cov = pd.read_csv(
    COV_PATH,
    sep="\t"
)

assert len(stage) == 1280
assert stage["subject"].nunique() == 64

assert len(cov) == 64
assert cov["subject"].nunique() == 64

print("Stage rows =", len(stage))
print("Subjects   =", stage["subject"].nunique())

print(
    "Groups =",
    stage[
        ["subject", "group"]
    ]
    .drop_duplicates()
    ["group"]
    .value_counts()
    .to_dict()
)

print()


# =============================================================================
# STEP 2. REBUILD CENTERED COVARIATES
# =============================================================================

print("=" * 110)
print("STEP 2. CENTER AGE, SEX, AND MEAN FD")
print("=" * 110)

# age and meanFD are already available in covariate_master_64,
# but we recenter them explicitly here for clarity.

cov["age"] = pd.to_numeric(
    cov["age"],
    errors="raise"
)

cov["meanFD"] = pd.to_numeric(
    cov["meanFD"],
    errors="raise"
)

# Binary coding:
# Female = 0
# Male   = 1
#
# Then center at the sample mean.
# This means sex_c = 0 corresponds to the average sex composition
# of the 64-subject sample rather than specifically Female.

sex_map = {
    "Female": 0.0,
    "Male": 1.0
}

cov["sex_num"] = cov["sex"].map(
    sex_map
)

if cov["sex_num"].isna().any():
    raise RuntimeError(
        "Unexpected sex coding found."
    )

cov["age_c"] = (
    cov["age"] -
    cov["age"].mean()
)

cov["meanFD_c"] = (
    cov["meanFD"] -
    cov["meanFD"].mean()
)

cov["sex_c"] = (
    cov["sex_num"] -
    cov["sex_num"].mean()
)

print(
    f"Age mean       = {cov['age'].mean():.6f}"
)

print(
    f"Mean FD mean   = {cov['meanFD'].mean():.8f}"
)

print(
    f"Male proportion= {cov['sex_num'].mean():.6f}"
)

print()

print("Centered means:")
print(
    "age_c    =",
    cov["age_c"].mean()
)
print(
    "sex_c    =",
    cov["sex_c"].mean()
)
print(
    "meanFD_c =",
    cov["meanFD_c"].mean()
)

print()


# =============================================================================
# STEP 3. MERGE
# =============================================================================

stage = stage.merge(
    cov[
        [
            "subject",
            "age_c",
            "sex_c",
            "meanFD_c"
        ]
    ],
    on="subject",
    how="left",
    validate="many_to_one"
)

assert len(stage) == 1280

assert not stage[
    [
        "age_c",
        "sex_c",
        "meanFD_c"
    ]
].isna().any().any()

stage["group"] = pd.Categorical(
    stage["group"],
    categories=["HC", "SZ"]
)

stage["stage"] = pd.Categorical(
    stage["stage"],
    categories=[
        "Outcome",
        "Anticipation"
    ]
)

# Save exact final analysis table
stage.to_csv(
    OUT / "final_gradient_sensitive_analysis_table.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 4. MODELS
# =============================================================================

print("=" * 110)
print("STEP 4. FIT FINAL GRADIENT-SENSITIVE COVARIATE MODELS")
print("=" * 110)

# -------------------------------------------------------------------------
# FULL MODEL
#
# Diagnosis:
#   Group × Stage × G1
#
# Covariates are explicitly allowed to affect:
#   FC level
#   reward-stage difference
#   G1 slope
#   Stage × G1 differentiation
#
# No Group × Covariate terms are introduced.
# -------------------------------------------------------------------------

FULL_FORMULA = """
FC ~ group * stage * G1_centered
   + age_c * stage * G1_centered
   + sex_c * stage * G1_centered
   + meanFD_c * stage * G1_centered
"""

# -------------------------------------------------------------------------
# REDUCED MODEL
#
# Exactly the same model except removal of:
#   Group × Stage × G1
#
# This gives the 1-df LRT for the diagnosis effect on
# reward-stage gradient differentiation.
# -------------------------------------------------------------------------

REDUCED_FORMULA = """
FC ~ group * stage
   + group * G1_centered
   + stage * G1_centered
   + age_c * stage * G1_centered
   + sex_c * stage * G1_centered
   + meanFD_c * stage * G1_centered
"""

full = fit_mixed(
    FULL_FORMULA,
    stage,
    "FULL"
)

reduced = fit_mixed(
    REDUCED_FORMULA,
    stage,
    "REDUCED"
)


# =============================================================================
# STEP 5. PRIMARY LRT
# =============================================================================

print()
print("=" * 110)
print("STEP 5. PRIMARY GROUP × STAGE × G1 TEST")
print("=" * 110)

lrt = 2 * (
    full.llf -
    reduced.llf
)

lrt_p = chi2.sf(
    lrt,
    1
)

print(
    f"Full logLik    = {full.llf:.10f}"
)

print(
    f"Reduced logLik = {reduced.llf:.10f}"
)

print()

print(
    f"LRT chi2(1) = {lrt:.10f}"
)

print(
    f"LRT p       = {lrt_p:.12g}"
)

print()


# =============================================================================
# STEP 6. FIXED EFFECT TABLE
# =============================================================================

print("=" * 110)
print("STEP 6. FIXED EFFECTS")
print("=" * 110)

fe = full.fe_params.copy()

cov_all = full.cov_params()

fe_names = list(
    fe.index
)

cov_fe = cov_all.loc[
    fe_names,
    fe_names
]

fixed_rows = []

for term in fe_names:

    beta = float(
        fe[term]
    )

    se = float(
        np.sqrt(
            cov_fe.loc[
                term,
                term
            ]
        )
    )

    z = beta / se

    p = 2 * norm.sf(
        abs(z)
    )

    fixed_rows.append({
        "term": term,
        "beta": beta,
        "SE": se,
        "CI95_low": beta - 1.96 * se,
        "CI95_high": beta + 1.96 * se,
        "z": z,
        "p": p
    })

fixed = pd.DataFrame(
    fixed_rows
)

fixed.to_csv(
    OUT /
    "final_gradient_sensitive_fixed_effects.tsv",
    sep="\t",
    index=False
)

print(
    fixed.to_string(
        index=False,
        float_format=lambda x: f"{x:.9f}"
    )
)

print()


# =============================================================================
# STEP 7. IDENTIFY KEY TERMS
# =============================================================================

TERM_G1 = "G1_centered"

TERM_GROUP_G1 = find_term(
    fe_names,
    [
        "group[T.SZ]",
        "G1_centered"
    ],
    [
        "stage[T.Anticipation]"
    ]
)

TERM_STAGE_G1 = find_term(
    fe_names,
    [
        "stage[T.Anticipation]",
        "G1_centered"
    ],
    [
        "group[T.SZ]",
        "age_c",
        "sex_c",
        "meanFD_c"
    ]
)

TERM_THREEWAY = find_term(
    fe_names,
    [
        "group[T.SZ]",
        "stage[T.Anticipation]",
        "G1_centered"
    ]
)


# =============================================================================
# STEP 8. CONTRAST FUNCTION
# =============================================================================

def contrast(name, weights):

    w = np.zeros(
        len(fe_names),
        dtype=float
    )

    for term, value in weights.items():

        w[
            fe_names.index(term)
        ] = value

    estimate = float(
        w @ fe.values
    )

    variance = float(
        w @ cov_fe.values @ w
    )

    se = np.sqrt(
        max(variance, 0)
    )

    z = estimate / se

    p = 2 * norm.sf(
        abs(z)
    )

    return {
        "contrast": name,
        "estimate": estimate,
        "SE": se,
        "CI95_low": estimate - 1.96 * se,
        "CI95_high": estimate + 1.96 * se,
        "z": z,
        "p": p
    }


# =============================================================================
# STEP 9. GROUP-SPECIFIC SLOPES AT MEAN COVARIATES
# =============================================================================

contrasts = []

contrasts.append(
    contrast(
        "HC Outcome slope",
        {
            TERM_G1: 1
        }
    )
)

contrasts.append(
    contrast(
        "HC Anticipation slope",
        {
            TERM_G1: 1,
            TERM_STAGE_G1: 1
        }
    )
)

contrasts.append(
    contrast(
        "SZ Outcome slope",
        {
            TERM_G1: 1,
            TERM_GROUP_G1: 1
        }
    )
)

contrasts.append(
    contrast(
        "SZ Anticipation slope",
        {
            TERM_G1: 1,
            TERM_GROUP_G1: 1,
            TERM_STAGE_G1: 1,
            TERM_THREEWAY: 1
        }
    )
)

contrasts.append(
    contrast(
        "HC Anticipation minus Outcome slope difference",
        {
            TERM_STAGE_G1: 1
        }
    )
)

contrasts.append(
    contrast(
        "SZ Anticipation minus Outcome slope difference",
        {
            TERM_STAGE_G1: 1,
            TERM_THREEWAY: 1
        }
    )
)

contrasts.append(
    contrast(
        "SZ minus HC difference in stage-gradient differentiation",
        {
            TERM_THREEWAY: 1
        }
    )
)

contrast_df = pd.DataFrame(
    contrasts
)

contrast_df.to_csv(
    OUT /
    "final_gradient_sensitive_group_slopes.tsv",
    sep="\t",
    index=False
)

print("=" * 110)
print("STEP 9. GROUP-SPECIFIC SLOPES AT MEAN AGE / SEX / FD")
print("=" * 110)

print(
    contrast_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.9f}"
    )
)

print()


# =============================================================================
# STEP 10. COVARIATE MODERATION TERMS
# =============================================================================

print("=" * 110)
print("STEP 10. COVARIATE MODERATION OF G1 AND STAGE × G1")
print("=" * 110)

moderator_rows = []

for covariate in [
    "age_c",
    "sex_c",
    "meanFD_c"
]:

    # Overall G1 slope moderation
    g1_term = find_term(
        fe_names,
        [
            covariate,
            "G1_centered"
        ],
        [
            "stage[T.Anticipation]"
        ]
    )

    # Stage × G1 differentiation moderation
    stage_g1_term = find_term(
        fe_names,
        [
            covariate,
            "stage[T.Anticipation]",
            "G1_centered"
        ]
    )

    for label, term in [
        (
            f"{covariate} × G1",
            g1_term
        ),
        (
            f"{covariate} × Stage × G1",
            stage_g1_term
        )
    ]:

        row = fixed[
            fixed["term"] == term
        ].iloc[0]

        moderator_rows.append({
            "effect": label,
            "term": term,
            "beta": row["beta"],
            "SE": row["SE"],
            "CI95_low": row["CI95_low"],
            "CI95_high": row["CI95_high"],
            "p": row["p"]
        })


moderators = pd.DataFrame(
    moderator_rows
)

moderators.to_csv(
    OUT /
    "covariate_gradient_moderation_terms.tsv",
    sep="\t",
    index=False
)

print(
    moderators.to_string(
        index=False,
        float_format=lambda x: f"{x:.9f}"
    )
)

print()


# =============================================================================
# STEP 11. RANDOM EFFECTS
# =============================================================================

print("=" * 110)
print("STEP 11. RANDOM EFFECT COVARIANCE")
print("=" * 110)

print(
    full.cov_re
)

print()


# =============================================================================
# STEP 12. SAVE MODEL SUMMARIES
# =============================================================================

with open(
    OUT /
    "full_gradient_sensitive_model_summary.txt",
    "w"
) as f:

    f.write(
        full.summary().as_text()
    )

with open(
    OUT /
    "reduced_gradient_sensitive_model_summary.txt",
    "w"
) as f:

    f.write(
        reduced.summary().as_text()
    )

with open(
    OUT /
    "primary_gradient_sensitive_LRT.txt",
    "w"
) as f:

    f.write(
        "FINAL GRADIENT-SENSITIVE COVARIATE MODEL\n\n"
        "Primary test:\n"
        "Group × Stage × G1\n\n"
        f"Full logLik    = {full.llf:.12f}\n"
        f"Reduced logLik = {reduced.llf:.12f}\n"
        f"LRT chi2(1)    = {lrt:.12f}\n"
        f"LRT p          = {lrt_p:.12g}\n"
        f"Full converged = {full.converged}\n"
        f"Reduced converged = {reduced.converged}\n"
    )


# =============================================================================
# FINAL REPORT
# =============================================================================

primary = contrast_df[
    contrast_df["contrast"] ==
    "SZ minus HC difference in stage-gradient differentiation"
].iloc[0]

hc_delta = contrast_df[
    contrast_df["contrast"] ==
    "HC Anticipation minus Outcome slope difference"
].iloc[0]

sz_delta = contrast_df[
    contrast_df["contrast"] ==
    "SZ Anticipation minus Outcome slope difference"
].iloc[0]


print()
print("=" * 110)
print("FINAL GRADIENT-SENSITIVE COVARIATE RESULT")
print("=" * 110)

print()

print(
    "HC stage-gradient differentiation "
    "(at mean Age/Sex/FD):"
)

print(
    f"beta = {hc_delta['estimate']:.9f}"
)

print(
    f"95% CI "
    f"[{hc_delta['CI95_low']:.9f}, "
    f"{hc_delta['CI95_high']:.9f}]"
)

print(
    f"p = {hc_delta['p']:.12g}"
)

print()

print(
    "SZ stage-gradient differentiation "
    "(at mean Age/Sex/FD):"
)

print(
    f"beta = {sz_delta['estimate']:.9f}"
)

print(
    f"95% CI "
    f"[{sz_delta['CI95_low']:.9f}, "
    f"{sz_delta['CI95_high']:.9f}]"
)

print(
    f"p = {sz_delta['p']:.12g}"
)

print()

print(
    "PRIMARY FINAL:"
)

print(
    "SZ - HC difference in "
    "stage-gradient differentiation"
)

print(
    f"beta = {primary['estimate']:.9f}"
)

print(
    f"95% CI "
    f"[{primary['CI95_low']:.9f}, "
    f"{primary['CI95_high']:.9f}]"
)

print(
    f"Wald p = {primary['p']:.12g}"
)

print(
    f"LRT chi2(1) = {lrt:.9f}"
)

print(
    f"LRT p = {lrt_p:.12g}"
)

print()

print(
    "Covariate moderator results:"
)

print(
    moderators[
        [
            "effect",
            "beta",
            "p"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.9f}"
    )
)

print()

print(
    "Full converged =",
    full.converged
)

print(
    "Reduced converged =",
    reduced.converged
)

print()

print("OUTPUT:")
print(OUT)

print()

print(
    "*** GRADIENT-SENSITIVE COVARIATE FINAL_CHECK = PASS ***"
)
