#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    os.environ["STRIATAL_HISTORICAL_ROOT"]
)

OUT = (
    ROOT /
    "25_current64_individual_G1_statistics" /
    "01_anatomical_separation" /
    "05_covariate_adjusted"
)

TABLE_OUT = OUT / "01_tables"
FIG_OUT = OUT / "02_figures"
QC_OUT = OUT / "03_qc"

for d in [
    TABLE_OUT,
    FIG_OUT,
    QC_OUT,
]:
    d.mkdir(
        parents=True,
        exist_ok=True
    )


CLIN_FILE = (
    ROOT /
    "21_current64_individual_stats" /
    "01_inputs" /
    "01_clinical" /
    "final64_clinical_for_server_v3.tsv"
)

MOTION_FILE = (
    ROOT /
    "21_current64_individual_stats" /
    "03_derived_metrics" /
    "motion" /
    "current64_motion_posttrim.tsv"
)

G1_FILE = (
    ROOT /
    "25_current64_individual_G1_statistics" /
    "01_anatomical_separation" /
    "01_tables" /
    "individual_G1_anatomical_separation.tsv"
)


# =============================================================================
# STEP 1. LOAD
# =============================================================================

print("=" * 110)
print("STEP 1. LOAD INPUTS")
print("=" * 110)

for p in [
    CLIN_FILE,
    MOTION_FILE,
    G1_FILE,
]:
    print(
        p,
        "->",
        p.exists()
    )

    if not p.exists():
        raise FileNotFoundError(
            p
        )


clin = pd.read_csv(
    CLIN_FILE,
    sep="\t"
)

motion = pd.read_csv(
    MOTION_FILE,
    sep="\t"
)

g1 = pd.read_csv(
    G1_FILE,
    sep="\t"
)

print()
print(
    "clinical =",
    clin.shape
)

print(
    "motion   =",
    motion.shape
)

print(
    "G1       =",
    g1.shape
)


# =============================================================================
# STEP 2. STRICT SUBJECT MERGE
# =============================================================================

print()
print("=" * 110)
print("STEP 2. BUILD STRICT 64-ROW MASTER TABLE")
print("=" * 110)

clin_keep = clin[
    [
        "subject",
        "group",
        "age",
        "sex",
    ]
].copy()

motion_keep = motion[
    [
        "subject",
        "group",
        "mean_FD_posttrim",
    ]
].copy()

motion_keep = motion_keep.rename(
    columns={
        "group":
            "group_motion"
    }
)

g1_keep = g1[
    [
        "subject",
        "group",
        "G1_VST_mean",
        "G1_caudate_mean",
        "G1_putamen_mean",
        "G1_separation",
    ]
].copy()

g1_keep = g1_keep.rename(
    columns={
        "group":
            "group_G1"
    }
)


master = clin_keep.merge(
    motion_keep,
    on="subject",
    how="inner",
    validate="one_to_one"
)

master = master.merge(
    g1_keep,
    on="subject",
    how="inner",
    validate="one_to_one"
)


# exact group match
if not np.all(
    master["group"]
    ==
    master["group_motion"]
):
    raise RuntimeError(
        "Clinical vs motion group mismatch."
    )

if not np.all(
    master["group"]
    ==
    master["group_G1"]
):
    raise RuntimeError(
        "Clinical vs G1 group mismatch."
    )

master = master.drop(
    columns=[
        "group_motion",
        "group_G1",
    ]
)


# =============================================================================
# STEP 3. TYPE / COMPLETENESS QC
# =============================================================================

master["age"] = pd.to_numeric(
    master["age"],
    errors="coerce"
)

master[
    "mean_FD_posttrim"
] = pd.to_numeric(
    master[
        "mean_FD_posttrim"
    ],
    errors="coerce"
)

master[
    "G1_separation"
] = pd.to_numeric(
    master[
        "G1_separation"
    ],
    errors="coerce"
)


required_cols = [
    "subject",
    "group",
    "age",
    "sex",
    "mean_FD_posttrim",
    "G1_separation",
]

missing = (
    master[
        required_cols
    ]
    .isna()
    .sum()
)

print(
    "Rows =",
    len(master)
)

print(
    "Unique subjects =",
    master[
        "subject"
    ].nunique()
)

print()
print("Missing:")
print(
    missing.to_string()
)

print()
print("Groups:")
print(
    master[
        "group"
    ]
    .value_counts()
    .to_string()
)

print()
print("Sex:")
print(
    master[
        "sex"
    ]
    .value_counts()
    .to_string()
)


if len(master) != 64:
    raise RuntimeError(
        f"Master N = {len(master)}"
    )

if master[
    "subject"
].nunique() != 64:
    raise RuntimeError(
        "Duplicate subjects."
    )

if set(
    master["group"]
) != {
    "HC",
    "SZ",
}:
    raise RuntimeError(
        "Unexpected group coding."
    )

if set(
    master["sex"]
) != {
    "Male",
    "Female",
}:
    raise RuntimeError(
        "Unexpected sex coding."
    )

if missing.sum() != 0:
    raise RuntimeError(
        "Missing model covariates/outcome."
    )


# =============================================================================
# STEP 4. CENTER CONTINUOUS COVARIATES
# =============================================================================

master[
    "age_centered"
] = (
    master["age"]
    -
    master["age"].mean()
)

master[
    "mean_FD_posttrim_centered"
] = (
    master["mean_FD_posttrim"]
    -
    master[
        "mean_FD_posttrim"
    ].mean()
)


# coding columns purely for easy QC
master[
    "group_SZ"
] = (
    master["group"]
    ==
    "SZ"
).astype(int)

master[
    "sex_Male"
] = (
    master["sex"]
    ==
    "Male"
).astype(int)


MASTER_FILE = (
    TABLE_OUT /
    "current64_G1_separation_covariate_master.tsv"
)

master.to_csv(
    MASTER_FILE,
    sep="\t",
    index=False
)


# =============================================================================
# STEP 5. DESCRIPTIVE COVARIATE TABLE
# =============================================================================

print()
print("=" * 110)
print("STEP 5. GROUP COVARIATE SUMMARY")
print("=" * 110)

summary_rows = []

for group in [
    "HC",
    "SZ",
]:

    d = master[
        master["group"]
        ==
        group
    ]

    summary_rows.append({
        "group":
            group,

        "N":
            len(d),

        "age_mean":
            float(
                d["age"].mean()
            ),

        "age_SD":
            float(
                d["age"].std(
                    ddof=1
                )
            ),

        "Male_n":
            int(
                np.sum(
                    d["sex"]
                    ==
                    "Male"
                )
            ),

        "Female_n":
            int(
                np.sum(
                    d["sex"]
                    ==
                    "Female"
                )
            ),

        "mean_FD_mean":
            float(
                d[
                    "mean_FD_posttrim"
                ].mean()
            ),

        "mean_FD_SD":
            float(
                d[
                    "mean_FD_posttrim"
                ].std(
                    ddof=1
                )
            ),

        "G1_separation_mean":
            float(
                d[
                    "G1_separation"
                ].mean()
            ),

        "G1_separation_SD":
            float(
                d[
                    "G1_separation"
                ].std(
                    ddof=1
                )
            ),
    })


group_summary = pd.DataFrame(
    summary_rows
)

group_summary.to_csv(
    TABLE_OUT /
    "G1_separation_covariate_group_summary.tsv",
    sep="\t",
    index=False
)

print(
    group_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# STEP 6. MODEL 0 — UNADJUSTED OLS
# =============================================================================

print()
print("=" * 110)
print("STEP 6. MODEL 0 — UNADJUSTED OLS")
print("=" * 110)

formula0 = (
    'G1_separation ~ '
    'C(group, Treatment(reference="HC"))'
)

model0 = smf.ols(
    formula0,
    data=master
).fit()

model0_hc3 = model0.get_robustcov_results(
    cov_type="HC3"
)

print(
    model0.summary()
)


# =============================================================================
# STEP 7. MODEL 1 — ADJUSTED
# =============================================================================

print()
print("=" * 110)
print("STEP 7. MODEL 1 — ADJUSTED PRIMARY MODEL")
print("=" * 110)

formula1 = (
    'G1_separation ~ '
    'C(group, Treatment(reference="HC")) '
    '+ age_centered '
    '+ C(sex, Treatment(reference="Female")) '
    '+ mean_FD_posttrim_centered'
)

model1 = smf.ols(
    formula1,
    data=master
).fit()

model1_hc3 = model1.get_robustcov_results(
    cov_type="HC3"
)

print(
    model1.summary()
)


# =============================================================================
# STEP 8. EXTRACT MODEL TABLES
# =============================================================================

def model_table(
    model,
    robust_model,
    model_name
):

    names = model.model.exog_names

    ci = model.conf_int(
        alpha=0.05
    )

    robust_ci = robust_model.conf_int(
        alpha=0.05
    )

    rows = []

    for i, term in enumerate(
        names
    ):

        rows.append({
            "model":
                model_name,

            "term":
                term,

            "beta":
                float(
                    model.params.iloc[i]
                ),

            "SE_OLS":
                float(
                    model.bse.iloc[i]
                ),

            "t_OLS":
                float(
                    model.tvalues.iloc[i]
                ),

            "p_OLS":
                float(
                    model.pvalues.iloc[i]
                ),

            "CI95_OLS_lower":
                float(
                    ci.iloc[i, 0]
                ),

            "CI95_OLS_upper":
                float(
                    ci.iloc[i, 1]
                ),

            "SE_HC3":
                float(
                    robust_model.bse[i]
                ),

            "t_HC3":
                float(
                    robust_model.tvalues[i]
                ),

            "p_HC3":
                float(
                    robust_model.pvalues[i]
                ),

            "CI95_HC3_lower":
                float(
                    robust_ci[i, 0]
                ),

            "CI95_HC3_upper":
                float(
                    robust_ci[i, 1]
                ),
        })

    return pd.DataFrame(
        rows
    )


m0_table = model_table(
    model0,
    model0_hc3,
    "Model0_unadjusted"
)

m1_table = model_table(
    model1,
    model1_hc3,
    "Model1_adjusted"
)

coef_table = pd.concat(
    [
        m0_table,
        m1_table,
    ],
    ignore_index=True
)

coef_table.to_csv(
    TABLE_OUT /
    "G1_separation_regression_coefficients.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 9. IDENTIFY GROUP EFFECT
# =============================================================================

group_term = (
    'C(group, Treatment(reference="HC"))[T.SZ]'
)

if group_term not in model1.params.index:
    raise RuntimeError(
        "Cannot locate adjusted SZ group coefficient."
    )

group_idx = list(
    model1.params.index
).index(
    group_term
)

adjusted_beta = float(
    model1.params[
        group_term
    ]
)

adjusted_p_ols = float(
    model1.pvalues[
        group_term
    ]
)

adjusted_ci_ols = model1.conf_int().loc[
    group_term
].to_numpy(
    dtype=float
)

adjusted_se_hc3 = float(
    model1_hc3.bse[
        group_idx
    ]
)

adjusted_t_hc3 = float(
    model1_hc3.tvalues[
        group_idx
    ]
)

adjusted_p_hc3 = float(
    model1_hc3.pvalues[
        group_idx
    ]
)

adjusted_ci_hc3 = (
    model1_hc3.conf_int()[
        group_idx
    ]
)


# =============================================================================
# STEP 10. STANDARDIZED GROUP EFFECT
# =============================================================================

# Standardize outcome only.
# Binary group coefficient then reflects adjusted difference
# in outcome SD units.

master[
    "G1_separation_z"
] = (
    master[
        "G1_separation"
    ]
    -
    master[
        "G1_separation"
    ].mean()
) / master[
    "G1_separation"
].std(
    ddof=1
)

std_formula = (
    'G1_separation_z ~ '
    'C(group, Treatment(reference="HC")) '
    '+ age_centered '
    '+ C(sex, Treatment(reference="Female")) '
    '+ mean_FD_posttrim_centered'
)

std_model = smf.ols(
    std_formula,
    data=master
).fit()

std_hc3 = (
    std_model
    .get_robustcov_results(
        cov_type="HC3"
    )
)

std_idx = list(
    std_model.params.index
).index(
    group_term
)

standardized_group_beta = float(
    std_model.params[
        group_term
    ]
)

standardized_group_p_hc3 = float(
    std_hc3.pvalues[
        std_idx
    ]
)


# =============================================================================
# STEP 11. ADJUSTED MARGINAL MEANS
# =============================================================================

# G-computation:
# preserve each subject's Age, Sex and FD,
# predict everyone once as HC and once as SZ,
# then average predictions.

pred_hc_data = master.copy()
pred_hc_data[
    "group"
] = "HC"

pred_sz_data = master.copy()
pred_sz_data[
    "group"
] = "SZ"

pred_hc = model1.predict(
    pred_hc_data
)

pred_sz = model1.predict(
    pred_sz_data
)

adjusted_HC_mean = float(
    np.mean(
        pred_hc
    )
)

adjusted_SZ_mean = float(
    np.mean(
        pred_sz
    )
)

adjusted_HC_minus_SZ = float(
    adjusted_HC_mean
    -
    adjusted_SZ_mean
)


marginal = pd.DataFrame([
    {
        "group":
            "HC",

        "adjusted_mean_G1_separation":
            adjusted_HC_mean,
    },
    {
        "group":
            "SZ",

        "adjusted_mean_G1_separation":
            adjusted_SZ_mean,
    },
])

marginal.to_csv(
    TABLE_OUT /
    "G1_separation_adjusted_marginal_means.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 12. MODEL FIT
# =============================================================================

fit_table = pd.DataFrame([
    {
        "model":
            "Model0_unadjusted",

        "N":
            int(
                model0.nobs
            ),

        "R2":
            float(
                model0.rsquared
            ),

        "adjusted_R2":
            float(
                model0.rsquared_adj
            ),

        "AIC":
            float(
                model0.aic
            ),

        "BIC":
            float(
                model0.bic
            ),
    },
    {
        "model":
            "Model1_adjusted",

        "N":
            int(
                model1.nobs
            ),

        "R2":
            float(
                model1.rsquared
            ),

        "adjusted_R2":
            float(
                model1.rsquared_adj
            ),

        "AIC":
            float(
                model1.aic
            ),

        "BIC":
            float(
                model1.bic
            ),
    },
])

fit_table.to_csv(
    TABLE_OUT /
    "G1_separation_model_fit.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 13. VIF
# =============================================================================

# Build design matrix directly from fitted Model 1.
X = pd.DataFrame(
    model1.model.exog,
    columns=model1.model.exog_names
)

vif_rows = []

for i, col in enumerate(
    X.columns
):

    if col == "Intercept":
        continue

    vif_rows.append({
        "term":
            col,

        "VIF":
            float(
                variance_inflation_factor(
                    X.values,
                    i
                )
            ),
    })


vif = pd.DataFrame(
    vif_rows
)

vif.to_csv(
    TABLE_OUT /
    "G1_separation_model_VIF.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 14. RESIDUAL QC
# =============================================================================

resid = np.asarray(
    model1.resid,
    dtype=float
)

influence = model1.get_influence()

studentized = (
    influence.resid_studentized_external
)

cooks = (
    influence.cooks_distance[0]
)

diagnostic = master[
    [
        "subject",
        "group",
        "G1_separation",
    ]
].copy()

diagnostic[
    "fitted"
] = model1.fittedvalues

diagnostic[
    "residual"
] = resid

diagnostic[
    "studentized_residual"
] = studentized

diagnostic[
    "cooks_distance"
] = cooks

diagnostic.to_csv(
    TABLE_OUT /
    "G1_separation_regression_diagnostics.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 15. FIGURE 1 — RAW VS ADJUSTED GROUP EFFECT
# =============================================================================

raw_hc = master.loc[
    master["group"] == "HC",
    "G1_separation"
].to_numpy()

raw_sz = master.loc[
    master["group"] == "SZ",
    "G1_separation"
].to_numpy()


fig, ax = plt.subplots(
    figsize=(7.0, 6.0)
)

positions = [
    1,
    2,
]

data = [
    raw_hc,
    raw_sz,
]

ax.boxplot(
    data,
    positions=positions,
    widths=0.45,
    showfliers=False
)

rng = np.random.default_rng(
    20260825
)

for xpos, values in zip(
    positions,
    data
):

    jitter = rng.normal(
        0,
        0.05,
        size=len(values)
    )

    ax.scatter(
        xpos + jitter,
        values,
        s=28,
        alpha=0.60
    )


# adjusted marginal means
ax.scatter(
    [
        1,
        2,
    ],
    [
        adjusted_HC_mean,
        adjusted_SZ_mean,
    ],
    marker="D",
    s=85,
    zorder=5,
    label="Adjusted marginal mean"
)

ax.plot(
    [
        1,
        2,
    ],
    [
        adjusted_HC_mean,
        adjusted_SZ_mean,
    ],
    linewidth=1.5,
    linestyle="--"
)

ax.set_xticks(
    positions
)

ax.set_xticklabels(
    [
        "HC (n=32)",
        "SZ (n=32)",
    ]
)

ax.set_ylabel(
    "G1 anatomical separation"
)

ax.set_title(
    "G1 separation: raw distributions and covariate-adjusted means"
)

annotation = (
    f"Adjusted SZ − HC = {adjusted_beta:.4f}\n"
    f"HC3 p = {adjusted_p_hc3:.4g}\n"
    f"Std. β = {standardized_group_beta:.3f}"
)

ax.text(
    0.98,
    0.98,
    annotation,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round",
        alpha=0.08
    )
)

ax.legend(
    frameon=False,
    loc="lower left"
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

fig.tight_layout()

fig.savefig(
    FIG_OUT /
    "Figure_G1_separation_raw_and_adjusted.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG_OUT /
    "Figure_G1_separation_raw_and_adjusted.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# STEP 16. FIGURE 2 — ADJUSTED GROUP EFFECT WITH CI
# =============================================================================

# Plot coefficient directly:
# model coefficient is SZ - HC.

fig, ax = plt.subplots(
    figsize=(6.5, 4.5)
)

ax.errorbar(
    adjusted_beta,
    0,
    xerr=np.array([
        [
            adjusted_beta
            -
            adjusted_ci_hc3[0]
        ],
        [
            adjusted_ci_hc3[1]
            -
            adjusted_beta
        ],
    ]),
    marker="o",
    markersize=8,
    capsize=6,
    linewidth=1.8
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_yticks([
    0
])

ax.set_yticklabels([
    "SZ − HC"
])

ax.set_xlabel(
    "Adjusted difference in G1 anatomical separation"
)

ax.set_title(
    "Adjusted diagnostic-group effect"
)

ax.text(
    0.98,
    0.92,
    (
        f"β = {adjusted_beta:.4f}\n"
        f"95% CI [{adjusted_ci_hc3[0]:.4f}, "
        f"{adjusted_ci_hc3[1]:.4f}]\n"
        f"HC3 p = {adjusted_p_hc3:.4g}"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top"
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

ax.spines[
    "left"
].set_visible(
    False
)

fig.tight_layout()

fig.savefig(
    FIG_OUT /
    "Figure_adjusted_group_effect_G1_separation.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG_OUT /
    "Figure_adjusted_group_effect_G1_separation.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# STEP 17. FINAL SUMMARY
# =============================================================================

print()
print("=" * 110)
print("FINAL RESULT")
print("=" * 110)

print()
print("UNADJUSTED")
print("-" * 90)

m0_group_idx = list(
    model0.params.index
).index(
    group_term
)

print(
    "SZ - HC beta =",
    float(
        model0.params[
            group_term
        ]
    )
)

print(
    "OLS p =",
    float(
        model0.pvalues[
            group_term
        ]
    )
)

print(
    "HC3 p =",
    float(
        model0_hc3.pvalues[
            m0_group_idx
        ]
    )
)


print()
print("ADJUSTED PRIMARY MODEL")
print("-" * 90)

print(
    "Model:",
    formula1
)

print()
print(
    "Adjusted HC mean =",
    adjusted_HC_mean
)

print(
    "Adjusted SZ mean =",
    adjusted_SZ_mean
)

print(
    "Adjusted HC - SZ =",
    adjusted_HC_minus_SZ
)

print()
print(
    "Group coefficient SZ - HC =",
    adjusted_beta
)

print(
    "OLS 95% CI =",
    adjusted_ci_ols.tolist()
)

print(
    "OLS p =",
    adjusted_p_ols
)

print()
print(
    "HC3 SE =",
    adjusted_se_hc3
)

print(
    "HC3 t =",
    adjusted_t_hc3
)

print(
    "HC3 95% CI =",
    adjusted_ci_hc3.tolist()
)

print(
    "HC3 p =",
    adjusted_p_hc3
)

print()
print(
    "Standardized group beta =",
    standardized_group_beta
)

print(
    "Standardized group HC3 p =",
    standardized_group_p_hc3
)

print()
print("MODEL FIT")
print("-" * 90)

print(
    fit_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print()
print("VIF")
print("-" * 90)

print(
    vif.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("FULL ADJUSTED COEFFICIENT TABLE")
print("-" * 90)

print(
    m1_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# QC JSON
# =============================================================================

max_vif = float(
    vif[
        "VIF"
    ].max()
)

final_pass = bool(
    len(master) == 64
    and
    master[
        required_cols
    ].isna().sum().sum() == 0
    and
    np.isfinite(
        adjusted_beta
    )
    and
    np.isfinite(
        adjusted_p_hc3
    )
)

qc_json = {
    "N":
        64,

    "HC":
        int(
            np.sum(
                master["group"]
                ==
                "HC"
            )
        ),

    "SZ":
        int(
            np.sum(
                master["group"]
                ==
                "SZ"
            )
        ),

    "Age_complete":
        bool(
            master[
                "age"
            ].notna().all()
        ),

    "Sex_complete":
        bool(
            master[
                "sex"
            ].notna().all()
        ),

    "mean_FD_posttrim_complete":
        bool(
            master[
                "mean_FD_posttrim"
            ].notna().all()
        ),

    "G1_separation_complete":
        bool(
            master[
                "G1_separation"
            ].notna().all()
        ),

    "group_reference":
        "HC",

    "sex_reference":
        "Female",

    "primary_model":
        formula1,

    "primary_SE":
        "HC3 heteroskedasticity-robust",

    "adjusted_SZ_minus_HC_beta":
        adjusted_beta,

    "adjusted_group_HC3_p":
        adjusted_p_hc3,

    "adjusted_group_HC3_CI95":
        [
            float(
                adjusted_ci_hc3[0]
            ),
            float(
                adjusted_ci_hc3[1]
            ),
        ],

    "standardized_group_beta":
        standardized_group_beta,

    "model_R2":
        float(
            model1.rsquared
        ),

    "model_adjusted_R2":
        float(
            model1.rsquared_adj
        ),

    "max_VIF":
        max_vif,

    "final_check":
        (
            "PASS"
            if final_pass
            else
            "FAIL"
        ),
}

with open(
    QC_OUT /
    "FINAL_G1_separation_adjusted_QC.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        qc_json,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "FINAL_CHECK =",
    qc_json[
        "final_check"
    ]
)

print()
print("OUTPUT TABLES")
print("-" * 90)

for p in [
    MASTER_FILE,

    TABLE_OUT /
    "G1_separation_covariate_group_summary.tsv",

    TABLE_OUT /
    "G1_separation_regression_coefficients.tsv",

    TABLE_OUT /
    "G1_separation_adjusted_marginal_means.tsv",

    TABLE_OUT /
    "G1_separation_model_fit.tsv",

    TABLE_OUT /
    "G1_separation_model_VIF.tsv",

    TABLE_OUT /
    "G1_separation_regression_diagnostics.tsv",
]:

    print(p)

print()
print("OUTPUT FIGURES")
print("-" * 90)

for p in [
    FIG_OUT /
    "Figure_G1_separation_raw_and_adjusted.png",

    FIG_OUT /
    "Figure_adjusted_group_effect_G1_separation.png",
]:

    print(p)
