#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    linregress,
    ttest_rel,
    wilcoxon,
    t as t_dist,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

MIXED_RESULT = (
    BASE /
    "random_slope_model/tables/RANDOM_SLOPE_KEY_RESULT.tsv"
)

OUT = BASE / "individual_slope_tests"
TAB = OUT / "tables"
FIG = OUT / "figures"

OUT.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

EXPECTED_N = 375


# =============================================================================
# 1. LOAD DATA
# =============================================================================

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

if df["Subject"].nunique() != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} subjects, "
        f"got {df['Subject'].nunique()}."
    )

if len(df) != EXPECTED_N * 20:
    raise RuntimeError(
        f"Expected {EXPECTED_N * 20} rows, "
        f"got {len(df)}."
    )

if df["FC"].isna().any():
    raise RuntimeError(
        "NaN FC detected."
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


# =============================================================================
# 2. INDIVIDUAL SLOPES
#
# For every subject:
#
# Outcome:
# FC = intercept + slope * G1c
#
# Anticipation:
# FC = intercept + slope * G1c
#
# =============================================================================

slope_rows = []

for (sub, stage), g in df.groupby(
    ["Subject", "Stage"]
):

    g = g.sort_values(
        "G1_bin"
    )

    if len(g) != 10:
        raise RuntimeError(
            f"{sub}, {stage}: "
            f"expected 10 bins, got {len(g)}"
        )

    lr = linregress(
        g["G1c"].to_numpy(dtype=float),
        g["FC"].to_numpy(dtype=float)
    )

    slope_rows.append({
        "Subject": sub,
        "Stage": stage,
        "slope": lr.slope,
        "intercept": lr.intercept,
        "r": lr.rvalue,
        "p_individual_linear": lr.pvalue,
        "slope_SE": lr.stderr,
    })


slopes_long = pd.DataFrame(
    slope_rows
)

slopes_long.to_csv(
    TAB / "individual_stage_slopes_long.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 3. WIDE TABLE: ONE ROW PER SUBJECT
# =============================================================================

slope_wide = (
    slopes_long
    .pivot(
        index="Subject",
        columns="Stage",
        values="slope"
    )
    .reset_index()
)

required_stage_cols = {
    "Anticipation",
    "Outcome"
}

if not required_stage_cols.issubset(
    slope_wide.columns
):
    raise RuntimeError(
        "Could not find both stage columns."
    )

slope_wide[
    "Anti_minus_Outcome"
] = (
    slope_wide["Anticipation"]
    -
    slope_wide["Outcome"]
)

slope_wide.to_csv(
    TAB / "individual_Anti_Outcome_slopes.tsv",
    sep="\t",
    index=False
)

if len(slope_wide) != EXPECTED_N:
    raise RuntimeError(
        "Slope table does not contain "
        "375 subjects."
    )


# =============================================================================
# 4. PAIRED SLOPE TEST
# =============================================================================

anti = slope_wide[
    "Anticipation"
].to_numpy(dtype=float)

outcome = slope_wide[
    "Outcome"
].to_numpy(dtype=float)

diff = (
    anti -
    outcome
)

n = len(diff)

mean_anti = np.mean(anti)
sd_anti = np.std(
    anti,
    ddof=1
)

mean_outcome = np.mean(outcome)
sd_outcome = np.std(
    outcome,
    ddof=1
)

mean_diff = np.mean(diff)
sd_diff = np.std(
    diff,
    ddof=1
)

se_diff = (
    sd_diff /
    np.sqrt(n)
)

dfree = n - 1

tcrit = t_dist.ppf(
    0.975,
    dfree
)

ci_low = (
    mean_diff -
    tcrit * se_diff
)

ci_high = (
    mean_diff +
    tcrit * se_diff
)

paired_t = ttest_rel(
    anti,
    outcome
)

# Cohen's dz for paired samples
cohen_dz = (
    mean_diff /
    sd_diff
)

# Wilcoxon signed-rank test
w = wilcoxon(
    anti,
    outcome,
    alternative="two-sided"
)

n_positive = int(
    np.sum(diff > 0)
)

n_negative = int(
    np.sum(diff < 0)
)

n_zero = int(
    np.sum(diff == 0)
)

prop_positive = (
    n_positive /
    n
)

slope_test = pd.DataFrame([{
    "N_subjects": n,

    "Anticipation_mean_slope":
        mean_anti,

    "Anticipation_SD_slope":
        sd_anti,

    "Outcome_mean_slope":
        mean_outcome,

    "Outcome_SD_slope":
        sd_outcome,

    "Anti_minus_Outcome_mean_difference":
        mean_diff,

    "difference_SD":
        sd_diff,

    "difference_SE":
        se_diff,

    "difference_CI95_low":
        ci_low,

    "difference_CI95_high":
        ci_high,

    "paired_t":
        paired_t.statistic,

    "paired_t_df":
        dfree,

    "paired_t_p":
        paired_t.pvalue,

    "Cohen_dz":
        cohen_dz,

    "Wilcoxon_W":
        w.statistic,

    "Wilcoxon_p":
        w.pvalue,

    "N_Anti_slope_greater_than_Outcome":
        n_positive,

    "N_Anti_slope_less_than_Outcome":
        n_negative,

    "N_equal":
        n_zero,

    "proportion_Anti_slope_greater":
        prop_positive,
}])

slope_test.to_csv(
    TAB / "PAIRED_SLOPE_TEST.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 5. CROSS-CHECK AGAINST MIXED MODEL INTERACTION
#
# Mean of individual paired slope differences
# should match fixed Stage x G1 interaction very closely.
# =============================================================================

crosscheck_rows = []

if MIXED_RESULT.exists():

    mm = pd.read_csv(
        MIXED_RESULT,
        sep="\t"
    )

    mixed_beta = float(
        mm.loc[
            0,
            "Anti_minus_Outcome_slope_difference"
        ]
    )

    abs_difference = abs(
        mixed_beta -
        mean_diff
    )

    crosscheck_rows.append({
        "paired_mean_slope_difference":
            mean_diff,

        "mixed_model_interaction_beta":
            mixed_beta,

        "absolute_difference":
            abs_difference,

        "match_within_1e_10":
            abs_difference < 1e-10,
    })

    print("\n" + "=" * 100)
    print("MIXED MODEL CROSS-CHECK")
    print("=" * 100)

    print(
        "Paired mean Δslope :",
        f"{mean_diff:.12f}"
    )

    print(
        "Mixed interaction  :",
        f"{mixed_beta:.12f}"
    )

    print(
        "Absolute difference:",
        f"{abs_difference:.12g}"
    )


pd.DataFrame(
    crosscheck_rows
).to_csv(
    TAB / "slope_vs_mixed_model_crosscheck.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 6. BH-FDR FUNCTION
# =============================================================================

def bh_fdr(pvalues):

    p = np.asarray(
        pvalues,
        dtype=float
    )

    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    q = (
        ranked *
        n /
        np.arange(
            1,
            n + 1
        )
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.clip(
        q,
        0,
        1
    )

    result = np.empty_like(q)
    result[order] = q

    return result


# =============================================================================
# 7. SUBJECT-LEVEL BIN-WISE PAIRED TESTS
#
# For each G1 bin:
#
# same 375 participants
# Anticipation vs Outcome
#
# =============================================================================

bin_rows = []

for b in range(1, 11):

    temp = (
        df.loc[
            df["G1_bin"].eq(b),
            [
                "Subject",
                "Stage",
                "FC"
            ]
        ]
        .pivot(
            index="Subject",
            columns="Stage",
            values="FC"
        )
        .dropna()
    )

    if len(temp) != EXPECTED_N:
        raise RuntimeError(
            f"Bin {b}: expected 375 paired "
            f"subjects, got {len(temp)}"
        )

    a = temp[
        "Anticipation"
    ].to_numpy(dtype=float)

    o = temp[
        "Outcome"
    ].to_numpy(dtype=float)

    d = a - o

    nd = len(d)

    md = np.mean(d)
    sdd = np.std(
        d,
        ddof=1
    )

    sed = (
        sdd /
        np.sqrt(nd)
    )

    tcrit_b = t_dist.ppf(
        0.975,
        nd - 1
    )

    cil = (
        md -
        tcrit_b * sed
    )

    cih = (
        md +
        tcrit_b * sed
    )

    tt = ttest_rel(
        a,
        o
    )

    wz = wilcoxon(
        a,
        o,
        alternative="two-sided"
    )

    dz = (
        md /
        sdd
    )

    bin_rows.append({
        "G1_bin": b,

        "N_subjects":
            nd,

        "Anticipation_mean":
            np.mean(a),

        "Anticipation_SD":
            np.std(
                a,
                ddof=1
            ),

        "Outcome_mean":
            np.mean(o),

        "Outcome_SD":
            np.std(
                o,
                ddof=1
            ),

        "Anti_minus_Outcome":
            md,

        "difference_SD":
            sdd,

        "difference_SE":
            sed,

        "difference_CI95_low":
            cil,

        "difference_CI95_high":
            cih,

        "paired_t":
            tt.statistic,

        "paired_t_df":
            nd - 1,

        "paired_t_p_uncorrected":
            tt.pvalue,

        "Cohen_dz":
            dz,

        "Wilcoxon_W":
            wz.statistic,

        "Wilcoxon_p_uncorrected":
            wz.pvalue,
    })


bin_tests = pd.DataFrame(
    bin_rows
)

bin_tests[
    "paired_t_p_FDR_BH"
] = bh_fdr(
    bin_tests[
        "paired_t_p_uncorrected"
    ]
)

bin_tests[
    "Wilcoxon_p_FDR_BH"
] = bh_fdr(
    bin_tests[
        "Wilcoxon_p_uncorrected"
    ]
)

bin_tests[
    "significant_paired_t_FDR_0.05"
] = (
    bin_tests[
        "paired_t_p_FDR_BH"
    ] < 0.05
)

bin_tests[
    "significant_Wilcoxon_FDR_0.05"
] = (
    bin_tests[
        "Wilcoxon_p_FDR_BH"
    ] < 0.05
)

bin_tests.to_csv(
    TAB / "SUBJECT_LEVEL_BINWISE_PAIRED_TESTS.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 8. GROUP DESCRIPTIVES FOR FIGURES
# =============================================================================

profile = (
    df
    .groupby(
        [
            "Stage",
            "G1_bin"
        ],
        as_index=False
    )
    .agg(
        mean_FC=(
            "FC",
            "mean"
        ),
        SD=(
            "FC",
            "std"
        ),
        N=(
            "FC",
            "size"
        )
    )
)

profile[
    "SEM"
] = (
    profile["SD"] /
    np.sqrt(
        profile["N"]
    )
)

profile[
    "CI95_halfwidth"
] = (
    1.96 *
    profile["SEM"]
)

profile.to_csv(
    TAB / "stage_profile_mean_CI.tsv",
    sep="\t",
    index=False
)


difference_profile = (
    bin_tests[
        [
            "G1_bin",
            "N_subjects",
            "Anti_minus_Outcome",
            "difference_SD",
            "difference_SE",
            "difference_CI95_low",
            "difference_CI95_high",
            "paired_t_p_FDR_BH",
            "significant_paired_t_FDR_0.05",
        ]
    ]
    .copy()
)

difference_profile.to_csv(
    TAB / "Anti_minus_Outcome_difference_profile.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 9. FIGURE: STAGE PROFILES WITH 95% CI
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8.5, 6)
)

for stage in [
    "Outcome",
    "Anticipation"
]:

    g = (
        profile.loc[
            profile[
                "Stage"
            ].eq(stage)
        ]
        .sort_values(
            "G1_bin"
        )
    )

    x = g[
        "G1_bin"
    ].to_numpy(dtype=float)

    y = g[
        "mean_FC"
    ].to_numpy(dtype=float)

    ci = g[
        "CI95_halfwidth"
    ].to_numpy(dtype=float)

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        label=stage
    )

    ax.fill_between(
        x,
        y - ci,
        y + ci,
        alpha=0.18
    )

ax.set_xticks(
    np.arange(
        1,
        11
    )
)

ax.set_xlabel(
    "G1 position"
)

ax.set_ylabel(
    "Mean Fisher-z connectivity"
)

ax.set_title(
    "Subject-level Anticipation and Outcome profiles along G1"
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    FIG /
    "Figure3C_subject_level_stage_profiles_95CI.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# 10. FIGURE: DIFFERENCE PROFILE
#
# < 0 : Outcome > Anticipation
# > 0 : Anticipation > Outcome
# =============================================================================

x = bin_tests[
    "G1_bin"
].to_numpy(dtype=float)

y = bin_tests[
    "Anti_minus_Outcome"
].to_numpy(dtype=float)

lo = bin_tests[
    "difference_CI95_low"
].to_numpy(dtype=float)

hi = bin_tests[
    "difference_CI95_high"
].to_numpy(dtype=float)

sig = bin_tests[
    "significant_paired_t_FDR_0.05"
].to_numpy(dtype=bool)


fig, ax = plt.subplots(
    figsize=(8.5, 6)
)

ax.axhline(
    0,
    linewidth=1,
    linestyle="--"
)

ax.plot(
    x,
    y,
    marker="o",
    linewidth=2
)

ax.fill_between(
    x,
    lo,
    hi,
    alpha=0.18
)

# Mark FDR-significant bins
offset = (
    0.04 *
    (
        np.nanmax(hi) -
        np.nanmin(lo)
    )
)

for xi, yi, is_sig in zip(
    x,
    y,
    sig
):
    if is_sig:

        if yi >= 0:
            ypos = yi + offset
            va = "bottom"
        else:
            ypos = yi - offset
            va = "top"

        ax.text(
            xi,
            ypos,
            "*",
            ha="center",
            va=va,
            fontsize=14
        )

ax.set_xticks(
    np.arange(
        1,
        11
    )
)

ax.set_xlabel(
    "G1 position"
)

ax.set_ylabel(
    "Anticipation − Outcome connectivity"
)

ax.set_title(
    "Stage-preference shift along striatal G1"
)

fig.tight_layout()

fig.savefig(
    FIG /
    "Figure3D_Anti_minus_Outcome_difference_95CI_FDR.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# 11. FINAL REPORT
# =============================================================================

print("\n" + "=" * 100)
print("INDIVIDUAL SLOPES")
print("=" * 100)

print(
    f"Anticipation mean slope = "
    f"{mean_anti:.10f} "
    f"(SD = {sd_anti:.10f})"
)

print(
    f"Outcome mean slope      = "
    f"{mean_outcome:.10f} "
    f"(SD = {sd_outcome:.10f})"
)

print(
    f"Anti - Outcome          = "
    f"{mean_diff:.10f}"
)

print(
    f"95% CI                  = "
    f"[{ci_low:.10f}, "
    f"{ci_high:.10f}]"
)

print(
    f"paired t({dfree})         = "
    f"{paired_t.statistic:.8f}"
)

print(
    f"paired p                = "
    f"{paired_t.pvalue:.12g}"
)

print(
    f"Cohen's dz              = "
    f"{cohen_dz:.6f}"
)

print(
    f"Wilcoxon p              = "
    f"{w.pvalue:.12g}"
)

print(
    f"Anti slope > Outcome    = "
    f"{n_positive}/{n} "
    f"({100 * prop_positive:.1f}%)"
)


print("\n" + "=" * 100)
print("SUBJECT-LEVEL BIN-WISE PAIRED TESTS")
print("=" * 100)

display_cols = [
    "G1_bin",
    "Anticipation_mean",
    "Outcome_mean",
    "Anti_minus_Outcome",
    "paired_t",
    "paired_t_p_uncorrected",
    "paired_t_p_FDR_BH",
    "significant_paired_t_FDR_0.05",
    "Cohen_dz",
]

print(
    bin_tests[
        display_cols
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("FDR-SIGNIFICANT G1 BINS")
print("=" * 100)

sig_table = (
    bin_tests.loc[
        bin_tests[
            "significant_paired_t_FDR_0.05"
        ],
        [
            "G1_bin",
            "Anti_minus_Outcome",
            "difference_CI95_low",
            "difference_CI95_high",
            "paired_t_p_FDR_BH",
            "Cohen_dz",
        ]
    ]
)

if len(sig_table) == 0:
    print(
        "No bins survive BH-FDR."
    )
else:
    print(
        sig_table.to_string(
            index=False
        )
    )


print("\n" + "=" * 100)
print("OUTPUTS")
print("=" * 100)

print(
    TAB /
    "individual_Anti_Outcome_slopes.tsv"
)

print(
    TAB /
    "PAIRED_SLOPE_TEST.tsv"
)

print(
    TAB /
    "SUBJECT_LEVEL_BINWISE_PAIRED_TESTS.tsv"
)

print(
    TAB /
    "Anti_minus_Outcome_difference_profile.tsv"
)

print(
    FIG /
    "Figure3C_subject_level_stage_profiles_95CI.png"
)

print(
    FIG /
    "Figure3D_Anti_minus_Outcome_difference_95CI_FDR.png"
)

print("\nDONE.")
