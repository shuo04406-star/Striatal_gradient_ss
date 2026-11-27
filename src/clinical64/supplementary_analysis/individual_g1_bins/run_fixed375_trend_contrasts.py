#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.formula.api as smf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(
    os.environ["STRIATAL_HISTORICAL_ROOT"]
)

OUT = (
    ROOT /
    "25_current64_individual_G1_statistics" /
    "10_fixed375_trend_contrasts"
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


IN_FILE = (
    ROOT /
    "25_current64_individual_G1_statistics" /
    "08_fixed375_normative_bins" /
    "01_tables" /
    "current64_fixed375_bin_covariate_long.tsv"
)


# =============================================================================
# HELPERS
# =============================================================================

def fit_mixed(
    formula,
    data,
    re_formula="1"
):

    model = smf.mixedlm(
        formula,
        data=data,
        groups=data[
            "subject"
        ],
        re_formula=re_formula
    )

    last = None

    for method in [
        "lbfgs",
        "bfgs",
        "powell",
    ]:

        try:

            result = model.fit(
                reml=False,
                method=method,
                maxiter=4000,
                disp=False
            )

            print(
                f"{method}: converged="
                f"{result.converged}"
            )

            if result.converged:
                return result

        except Exception as e:

            last = e

            print(
                method,
                "failed:",
                repr(e)
            )

    raise RuntimeError(
        f"MixedLM failed: {last}"
    )


def lrt(full, reduced):

    chi2 = float(
        2
        *
        (
            full.llf
            -
            reduced.llf
        )
    )

    df = int(
        len(
            full.fe_params
        )
        -
        len(
            reduced.fe_params
        )
    )

    p = float(
        stats.chi2.sf(
            chi2,
            df
        )
    )

    return (
        chi2,
        df,
        p
    )


# =============================================================================
# LOAD
# =============================================================================

print("=" * 110)
print("STEP 1. LOAD")
print("=" * 110)

print(
    IN_FILE,
    "->",
    IN_FILE.exists()
)

if not IN_FILE.exists():
    raise FileNotFoundError(
        IN_FILE
    )

df = pd.read_csv(
    IN_FILE,
    sep="\t"
)

if (
    len(df) != 640
    or
    df[
        "subject"
    ].nunique() != 64
):
    raise RuntimeError(
        "Expected 64 x 10 data."
    )


# =============================================================================
# ORDERED BIN CONTRASTS
# =============================================================================

print()
print("=" * 110)
print("STEP 2. ORDERED BIN VARIABLES")
print("=" * 110)

# Bin1...Bin10 -> centered -4.5 ... +4.5
df[
    "bin_linear"
] = (
    df[
        "bin"
    ].astype(float)
    -
    5.5
)

# Center quadratic term so it is orthogonal to the intercept;
# because bins are symmetric around zero, it is also orthogonal
# to the linear term.
raw_quad = (
    df[
        "bin_linear"
    ] ** 2
)

quad_mean = float(
    np.mean(
        np.arange(
            1,
            11,
            dtype=float
        )
        -
        5.5
    ) ** 2
)

# Correct the calculation explicitly from unique bin values
bin_values = (
    np.arange(
        1,
        11,
        dtype=float
    )
    -
    5.5
)

quad_center = float(
    np.mean(
        bin_values ** 2
    )
)

df[
    "bin_quadratic"
] = (
    raw_quad
    -
    quad_center
)


df[
    "group_SZ"
] = (
    df[
        "group"
    ]
    ==
    "SZ"
).astype(float)

df[
    "SZ_x_linear"
] = (
    df[
        "group_SZ"
    ]
    *
    df[
        "bin_linear"
    ]
)

df[
    "SZ_x_quadratic"
] = (
    df[
        "group_SZ"
    ]
    *
    df[
        "bin_quadratic"
    ]
)


print(
    df[
        [
            "bin",
            "bin_linear",
            "bin_quadratic",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "bin"
    )
    .to_string(
        index=False
    )
)


# =============================================================================
# PRIMARY MODELS
# =============================================================================

print()
print("=" * 110)
print("STEP 3. PRIMARY 1-DF LINEAR TREND")
print("=" * 110)

# Preserve arbitrary baseline bin shape with C(bin).
#
# Base:
# group difference constant across bins.
#
# Linear:
# SZ-HC difference allowed to change linearly with normative position.

base_formula = (
    'G1_bin_mean ~ '
    'C(bin, Treatment(reference=1)) '
    '+ group_SZ '
    '+ age_centered '
    '+ C(sex, Treatment(reference="Female")) '
    '+ mean_FD_posttrim_centered'
)

linear_formula = (
    'G1_bin_mean ~ '
    'C(bin, Treatment(reference=1)) '
    '+ group_SZ '
    '+ SZ_x_linear '
    '+ age_centered '
    '+ C(sex, Treatment(reference="Female")) '
    '+ mean_FD_posttrim_centered'
)

quadratic_formula = (
    'G1_bin_mean ~ '
    'C(bin, Treatment(reference=1)) '
    '+ group_SZ '
    '+ SZ_x_linear '
    '+ SZ_x_quadratic '
    '+ age_centered '
    '+ C(sex, Treatment(reference="Female")) '
    '+ mean_FD_posttrim_centered'
)


print("BASE")
base = fit_mixed(
    base_formula,
    df
)

print()
print("LINEAR")
linear = fit_mixed(
    linear_formula,
    df
)

print()
print("LINEAR + QUADRATIC")
quadratic = fit_mixed(
    quadratic_formula,
    df
)


# =============================================================================
# LRTs
# =============================================================================

linear_chi2, linear_df, linear_p = lrt(
    linear,
    base
)

quad_chi2, quad_df, quad_p = lrt(
    quadratic,
    linear
)


print()
print("=" * 110)
print("STEP 4. TREND TESTS")
print("=" * 110)

print(
    f"Linear Group×Bin trend: "
    f"chi2({linear_df})="
    f"{linear_chi2:.6f}, "
    f"p={linear_p:.10f}"
)

print(
    f"Additional quadratic trend: "
    f"chi2({quad_df})="
    f"{quad_chi2:.6f}, "
    f"p={quad_p:.10f}"
)


# =============================================================================
# DIRECT COEFFICIENTS
# =============================================================================

linear_beta = float(
    linear.fe_params[
        "SZ_x_linear"
    ]
)

linear_se = float(
    linear.bse_fe[
        "SZ_x_linear"
    ]
)

linear_z = (
    linear_beta
    /
    linear_se
)

linear_wald_p = float(
    2
    *
    stats.norm.sf(
        abs(
            linear_z
        )
    )
)

linear_ci = (
    linear_beta
    +
    np.array(
        [
            -1.96,
            1.96,
        ]
    )
    *
    linear_se
)


quad_beta = float(
    quadratic.fe_params[
        "SZ_x_quadratic"
    ]
)

quad_se = float(
    quadratic.bse_fe[
        "SZ_x_quadratic"
    ]
)

quad_z = (
    quad_beta
    /
    quad_se
)

quad_wald_p = float(
    2
    *
    stats.norm.sf(
        abs(
            quad_z
        )
    )
)

quad_ci = (
    quad_beta
    +
    np.array(
        [
            -1.96,
            1.96,
        ]
    )
    *
    quad_se
)


trend_table = pd.DataFrame([
    {
        "contrast":
            "Group x linear normative-bin trend",

        "interpretation":
            (
                "Change in SZ-HC difference "
                "per one-bin increase"
            ),

        "beta":
            linear_beta,

        "SE":
            linear_se,

        "z":
            linear_z,

        "Wald_p":
            linear_wald_p,

        "CI95_lower":
            linear_ci[0],

        "CI95_upper":
            linear_ci[1],

        "LRT_chi2":
            linear_chi2,

        "LRT_df":
            linear_df,

        "LRT_p":
            linear_p,
    },
    {
        "contrast":
            "Additional Group x quadratic trend",

        "interpretation":
            (
                "Additional curvature of SZ-HC "
                "difference after linear trend"
            ),

        "beta":
            quad_beta,

        "SE":
            quad_se,

        "z":
            quad_z,

        "Wald_p":
            quad_wald_p,

        "CI95_lower":
            quad_ci[0],

        "CI95_upper":
            quad_ci[1],

        "LRT_chi2":
            quad_chi2,

        "LRT_df":
            quad_df,

        "LRT_p":
            quad_p,
    },
])

trend_table.to_csv(
    TABLE_OUT /
    "fixed375_GroupXBin_linear_quadratic_trends.tsv",
    sep="\t",
    index=False
)

print()
print(
    trend_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# =============================================================================
# CLUSTER-ROBUST OLS SENSITIVITY
# =============================================================================

print()
print("=" * 110)
print("STEP 5. SUBJECT-CLUSTER ROBUST SENSITIVITY")
print("=" * 110)

ols = smf.ols(
    linear_formula,
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups":
            df[
                "subject"
            ]
    }
)

cluster_beta = float(
    ols.params[
        "SZ_x_linear"
    ]
)

cluster_se = float(
    ols.bse[
        "SZ_x_linear"
    ]
)

cluster_p = float(
    ols.pvalues[
        "SZ_x_linear"
    ]
)

cluster_ci = ols.conf_int().loc[
    "SZ_x_linear"
].to_numpy(
    dtype=float
)


cluster_table = pd.DataFrame([
    {
        "analysis":
            "OLS with SE clustered by subject",

        "beta_SZ_x_linear":
            cluster_beta,

        "SE_cluster":
            cluster_se,

        "CI95_lower":
            cluster_ci[0],

        "CI95_upper":
            cluster_ci[1],

        "p":
            cluster_p,
    }
])

cluster_table.to_csv(
    TABLE_OUT /
    "fixed375_linear_trend_cluster_robust_sensitivity.tsv",
    sep="\t",
    index=False
)

print(
    cluster_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# =============================================================================
# RANDOM-SLOPE SENSITIVITY
# =============================================================================

print()
print("=" * 110)
print("STEP 6. RANDOM-SLOPE SENSITIVITY")
print("=" * 110)

random_slope_status = "NOT_RUN"
random_slope_beta = np.nan
random_slope_p = np.nan

try:

    rs = fit_mixed(
        linear_formula,
        df,
        re_formula="1+bin_linear"
    )

    random_slope_status = (
        "PASS"
        if rs.converged
        else
        "REVIEW"
    )

    random_slope_beta = float(
        rs.fe_params[
            "SZ_x_linear"
        ]
    )

    rs_se = float(
        rs.bse_fe[
            "SZ_x_linear"
        ]
    )

    random_slope_p = float(
        2
        *
        stats.norm.sf(
            abs(
                random_slope_beta
                /
                rs_se
            )
        )
    )

except Exception as e:

    print(
        "Random-slope sensitivity failed:",
        repr(e)
    )

    random_slope_status = "FAILED"


random_slope_table = pd.DataFrame([
    {
        "status":
            random_slope_status,

        "beta_SZ_x_linear":
            random_slope_beta,

        "p":
            random_slope_p,
    }
])

random_slope_table.to_csv(
    TABLE_OUT /
    "fixed375_linear_trend_random_slope_sensitivity.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# VISUALIZE ADJUSTED DIFFERENCE TREND
# =============================================================================

# Read actual adjusted binwise coefficients from previous analysis.
BINWISE_FILE = (
    ROOT /
    "25_current64_individual_G1_statistics" /
    "08_fixed375_normative_bins" /
    "01_tables" /
    "fixed375_bins_binwise_HC_vs_SZ_adjusted_FDR.tsv"
)

binwise = pd.read_csv(
    BINWISE_FILE,
    sep="\t"
)


# fitted linear difference centered at empirical group difference
# at the middle of the axis.
#
# Estimate intercept of SZ-HC linear trend from MixedLM:
group_intercept = float(
    linear.fe_params[
        "group_SZ"
    ]
)

x = np.arange(
    1,
    11,
    dtype=float
)

xc = (
    x - 5.5
)

fitted_difference = (
    group_intercept
    +
    linear_beta
    *
    xc
)


fig, ax = plt.subplots(
    figsize=(
        8.0,
        5.8
    )
)

ax.errorbar(
    binwise[
        "bin"
    ],
    binwise[
        "beta_SZ_minus_HC"
    ],
    yerr=np.vstack([
        binwise[
            "beta_SZ_minus_HC"
        ]
        -
        binwise[
            "CI95_HC3_lower"
        ],

        binwise[
            "CI95_HC3_upper"
        ]
        -
        binwise[
            "beta_SZ_minus_HC"
        ],
    ]),
    marker="o",
    capsize=4,
    linewidth=1.2,
    label="Adjusted bin-wise estimates"
)

ax.plot(
    x,
    fitted_difference,
    linestyle="--",
    linewidth=2.2,
    label="Linear Group × Bin trend"
)

ax.axhline(
    0,
    linestyle=":",
    linewidth=1
)

ax.set_xticks(
    np.arange(
        1,
        11
    )
)

ax.set_xlabel(
    "Fixed 375 normative G1 bin"
)

ax.set_ylabel(
    "Adjusted SZ − HC difference in G1"
)

ax.set_title(
    "Linear compression trend along the normative G1 axis"
)

ax.text(
    0.02,
    0.98,
    (
        f"Linear interaction β = "
        f"{linear_beta:.5f} per bin\n"
        f"LRT p = {linear_p:.4g}"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round",
        alpha=0.08
    )
)

ax.legend(
    frameon=False
)

ax.spines[
    "top"
].set_visible(False)

ax.spines[
    "right"
].set_visible(False)

fig.tight_layout()

fig.savefig(
    FIG_OUT /
    "Figure_fixed375_linear_compression_trend.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# FINAL
# =============================================================================

final_pass = bool(
    base.converged
    and
    linear.converged
    and
    quadratic.converged
    and
    np.isfinite(
        linear_beta
    )
    and
    np.isfinite(
        linear_p
    )
)


qc = {
    "N_subjects":
        64,

    "rows":
        640,

    "primary_trend":
        "Group x linear normative-bin position",

    "bin_coding":
        "Bin 1..10 centered to -4.5..+4.5",

    "baseline_bin_shape":
        (
            "Categorical C(bin), therefore the "
            "normative profile itself is not forced "
            "to be linear."
        ),

    "linear_interaction_beta":
        linear_beta,

    "linear_interaction_LRT_p":
        linear_p,

    "quadratic_additional_LRT_p":
        quad_p,

    "cluster_robust_linear_p":
        cluster_p,

    "random_slope_status":
        random_slope_status,

    "random_slope_linear_p":
        (
            None
            if not np.isfinite(
                random_slope_p
            )
            else
            random_slope_p
        ),

    "interpretation_if_beta_negative":
        (
            "SZ-HC difference becomes progressively "
            "more negative from low to high normative "
            "bins, consistent with reduced gradient "
            "differentiation/compression."
        ),

    "analysis_role":
        (
            "ordered-trend follow-up decomposition "
            "of the previously significant 9-df "
            "Group x Bin omnibus interaction"
        ),

    "final_check":
        (
            "PASS"
            if final_pass
            else
            "REVIEW"
        ),
}

with open(
    QC_OUT /
    "FINAL_FIXED375_TREND_CONTRAST_QC.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        qc,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("=" * 110)
print("FINAL RESULT")
print("=" * 110)

print()
print("PRIMARY LINEAR COMPRESSION TREND")
print("-" * 100)

print(
    f"beta SZ×linear-bin = "
    f"{linear_beta:.8f}"
)

print(
    f"95% CI = "
    f"[{linear_ci[0]:.8f}, "
    f"{linear_ci[1]:.8f}]"
)

print(
    f"Wald p = "
    f"{linear_wald_p:.10f}"
)

print(
    f"LRT chi2({linear_df}) = "
    f"{linear_chi2:.8f}"
)

print(
    f"LRT p = "
    f"{linear_p:.10f}"
)


print()
print("ADDITIONAL QUADRATIC TREND")
print("-" * 100)

print(
    f"beta SZ×quadratic = "
    f"{quad_beta:.8f}"
)

print(
    f"LRT p = "
    f"{quad_p:.10f}"
)


print()
print("SENSITIVITY")
print("-" * 100)

print(
    f"Cluster-robust p = "
    f"{cluster_p:.10f}"
)

print(
    f"Random-slope status = "
    f"{random_slope_status}"
)

print(
    f"Random-slope p = "
    f"{random_slope_p}"
)


print()
print(
    "FINAL_CHECK =",
    qc[
        "final_check"
    ]
)
