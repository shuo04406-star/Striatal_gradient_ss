#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd

from scipy.stats import t as t_dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =============================================================================
# 0. PATHS
# =============================================================================

BASE = Path(
    os.environ["STRIATAL_HISTORICAL_ROOT"]
)

P24 = BASE / "24_stage_family_reanalysis_v2"

P25 = (
    BASE /
    "25_subject_level_stage_g1_model1"
)

PRS = (
    P25 /
    "random_slope_model"
)

PIND = (
    P25 /
    "individual_slope_tests"
)

OUT = Path(os.environ["STRIATAL_OUTPUT_ROOT"])

FIG = OUT / "figures"
TAB = OUT / "tables"
REPORT = OUT / "report"

for d in [OUT, FIG, TAB, REPORT]:
    d.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================================
# 1. INPUT FILES
# =============================================================================

FILES = {
    "meta":
        P24 / "tables/roi_metadata_stage_subfamily_v2.tsv",

    "subfamily_raw":
        P24 / "tables/Figure3_subfamily_binwise_fisherz.tsv",

    "subfamily_z":
        P24 / "tables/Figure3_subfamily_binwise_column_z.tsv",

    "subfamily_trend":
        P24 / "tables/subfamily_linear_trend_along_G1.tsv",

    "stage_profile":
        PIND / "tables/stage_profile_mean_CI.tsv",

    "difference_profile":
        PIND / "tables/Anti_minus_Outcome_difference_profile.tsv",

    "mixed_fixed":
        PRS / "tables/RANDOM_SLOPE_fixed_effects.tsv",

    "simple_slopes":
        PRS / "tables/RANDOM_SLOPE_stage_simple_slopes.tsv",

    "random_effects":
        PRS / "tables/RANDOM_SLOPE_random_effects.tsv",

    "model_comparison":
        PRS / "tables/RI_vs_RS_model_comparison.tsv",

    "interaction_compare":
        PRS / "tables/Stage_x_G1_RI_vs_RS.tsv",

    "mixed_key":
        PRS / "tables/RANDOM_SLOPE_KEY_RESULT.tsv",

    "paired_slope":
        PIND / "tables/PAIRED_SLOPE_TEST.tsv",

    "individual_slopes":
        PIND / "tables/individual_Anti_Outcome_slopes.tsv",

    "binwise":
        PIND / "tables/SUBJECT_LEVEL_BINWISE_PAIRED_TESTS.tsv",

    "crosscheck":
        PIND / "tables/slope_vs_mixed_model_crosscheck.tsv",
}


for key, p in FILES.items():
    if not p.exists():
        raise FileNotFoundError(
            f"{key} not found:\n{p}"
        )


# =============================================================================
# 2. LOAD DATA
# =============================================================================

meta = pd.read_csv(
    FILES["meta"],
    sep="\t"
)

sub_raw = pd.read_csv(
    FILES["subfamily_raw"],
    sep="\t",
    index_col=0
)

sub_z = pd.read_csv(
    FILES["subfamily_z"],
    sep="\t",
    index_col=0
)

sub_trend = pd.read_csv(
    FILES["subfamily_trend"],
    sep="\t"
)

stage_profile = pd.read_csv(
    FILES["stage_profile"],
    sep="\t"
)

difference_profile = pd.read_csv(
    FILES["difference_profile"],
    sep="\t"
)

fixed = pd.read_csv(
    FILES["mixed_fixed"],
    sep="\t"
)

simple = pd.read_csv(
    FILES["simple_slopes"],
    sep="\t"
)

random_effects = pd.read_csv(
    FILES["random_effects"],
    sep="\t"
)

model_comp = pd.read_csv(
    FILES["model_comparison"],
    sep="\t"
)

interaction_comp = pd.read_csv(
    FILES["interaction_compare"],
    sep="\t"
)

mixed_key = pd.read_csv(
    FILES["mixed_key"],
    sep="\t"
)

paired = pd.read_csv(
    FILES["paired_slope"],
    sep="\t"
)

individual_slopes = pd.read_csv(
    FILES["individual_slopes"],
    sep="\t"
)

binwise = pd.read_csv(
    FILES["binwise"],
    sep="\t"
)

crosscheck = pd.read_csv(
    FILES["crosscheck"],
    sep="\t"
)


# =============================================================================
# 3. FIXED ORDERS / DISPLAY LABELS
# =============================================================================

SUBFAMILY_ORDER = [
    "IFG",
    "Insula",
    "IPL",
    "Occipital_pole",
    "MFG",
    "OFC_vmPFC",
    "Thalamus",
    "Amygdala",
    "ACC",
    "Cingulate",
    "Frontal_pole",
    "SMA",
    "medial_SFC",
    "Midbrain_pallidal",
]

DISPLAY = {
    "IFG": "IFG",
    "Insula": "Insula",
    "IPL": "IPL",
    "Occipital_pole": "Occipital pole",
    "MFG": "MFG",
    "OFC_vmPFC": "OFC/vmPFC",
    "Thalamus": "Thalamus",
    "Amygdala": "Amygdala",
    "ACC": "ACC",
    "Cingulate": "Cingulate",
    "Frontal_pole": "Frontal pole",
    "SMA": "SMA",
    "medial_SFC": "medial SFC",
    "Midbrain_pallidal": "Midbrain/pallidal",
}

sub_raw = sub_raw[
    SUBFAMILY_ORDER
]

sub_z = sub_z[
    SUBFAMILY_ORDER
]


# =============================================================================
# 4. TABLE 1: ROI COMPOSITION
# =============================================================================

roi_comp = (
    meta
    .pivot_table(
        index="subfamily",
        columns="stage",
        values="roi_id",
        aggfunc="count",
        fill_value=0
    )
)

for c in [
    "Anticipation",
    "Outcome",
    "Not_assigned"
]:
    if c not in roi_comp.columns:
        roi_comp[c] = 0

roi_comp = (
    roi_comp
    .reindex(SUBFAMILY_ORDER)
    .fillna(0)
    .astype(int)
)

roi_comp["Total"] = (
    roi_comp[
        [
            "Anticipation",
            "Outcome",
            "Not_assigned"
        ]
    ].sum(axis=1)
)

roi_comp = (
    roi_comp
    .reset_index()
)

roi_comp["subfamily"] = (
    roi_comp["subfamily"]
    .map(DISPLAY)
)

roi_comp.to_csv(
    TAB / "Table1_ROI_stage_composition.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 5. TABLE 2: SUBFAMILY GROUP-LEVEL TRENDS
# =============================================================================

subtrend_final = (
    sub_trend.copy()
)

subtrend_final["subfamily"] = (
    subtrend_final["subfamily"]
    .map(DISPLAY)
)

subtrend_final["direction"] = np.where(
    subtrend_final["slope_per_bin"] > 0,
    "Increasing",
    "Decreasing"
)

subtrend_final["p_lt_0.05"] = (
    subtrend_final["p_linear"] < 0.05
)

subtrend_final.to_csv(
    TAB / "Table2_subfamily_group_level_trends.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 6. TABLE 3: SUBJECT-LEVEL STAGE PROFILE DESCRIPTIVES
# =============================================================================

profile_final = stage_profile.copy()

# use exact t-based 95% CI
profile_final["CI95_low"] = np.nan
profile_final["CI95_high"] = np.nan

for i, r in profile_final.iterrows():

    crit = t_dist.ppf(
        0.975,
        int(r["N"]) - 1
    )

    hw = (
        crit *
        r["SEM"]
    )

    profile_final.loc[
        i,
        "CI95_low"
    ] = (
        r["mean_FC"] - hw
    )

    profile_final.loc[
        i,
        "CI95_high"
    ] = (
        r["mean_FC"] + hw
    )

profile_final = (
    profile_final[
        [
            "Stage",
            "G1_bin",
            "N",
            "mean_FC",
            "SD",
            "SEM",
            "CI95_low",
            "CI95_high",
        ]
    ]
    .sort_values(
        [
            "Stage",
            "G1_bin"
        ]
    )
)

profile_final.to_csv(
    TAB / "Table3_subject_level_stage_profiles.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 7. TABLE 4: PRIMARY MIXED MODEL
# =============================================================================

fixed_final = fixed.copy()

def clean_term(term):

    if term == "Intercept":
        return "Intercept"

    if term == "G1c":
        return "G1"

    if (
        "C(Stage" in term
        and ":G1c" not in term
    ):
        return "Stage: Anticipation − Outcome"

    if ":G1c" in term:
        return "Stage × G1"

    return term


fixed_final["term_display"] = (
    fixed_final["term"]
    .apply(clean_term)
)

fixed_final = fixed_final[
    [
        "term_display",
        "beta",
        "SE",
        "CI95_low",
        "CI95_high",
        "p",
        "interpretation",
    ]
]

fixed_final.to_csv(
    TAB / "Table4_primary_random_slope_mixed_model.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 8. TABLE 5: SIMPLE SLOPES
# =============================================================================

simple_final = simple[
    [
        "Stage",
        "slope",
        "SE",
        "CI95_low",
        "CI95_high",
        "z",
        "p",
    ]
].copy()

simple_final.to_csv(
    TAB / "Table5_stage_simple_slopes.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 9. TABLE 6A / 6B: RANDOM EFFECTS + MODEL FIT
# =============================================================================

random_effects.to_csv(
    TAB / "Table6A_random_effects.tsv",
    sep="\t",
    index=False
)

model_comp.to_csv(
    TAB / "Table6B_random_structure_model_comparison.tsv",
    sep="\t",
    index=False
)

interaction_comp.to_csv(
    TAB / "Table6C_StagexG1_RI_vs_RS.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 10. TABLE 7: INDIVIDUAL PAIRED SLOPE TEST
# =============================================================================

paired.to_csv(
    TAB / "Table7_individual_paired_slope_test.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 11. TABLE 8: SUBJECT-LEVEL BIN-WISE TESTS
# =============================================================================

bin_final = binwise.copy()

bin_final["direction"] = np.where(
    bin_final["Anti_minus_Outcome"] > 0,
    "Anticipation > Outcome",
    "Outcome > Anticipation"
)

bin_final["robust_both_tests_FDR"] = (
    bin_final[
        "significant_paired_t_FDR_0.05"
    ]
    &
    bin_final[
        "significant_Wilcoxon_FDR_0.05"
    ]
)

bin_final.to_csv(
    TAB / "Table8_subject_level_binwise_paired_tests.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 12. TABLE 9: DIFFERENCE PROFILE
# =============================================================================

diff_final = difference_profile.copy()

diff_final["preference"] = np.where(
    diff_final["Anti_minus_Outcome"] > 0,
    "Anticipation preference",
    "Outcome preference"
)

diff_final.to_csv(
    TAB / "Table9_stage_difference_profile.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 13. COPY CORE SOURCE TABLES FOR AUDIT
# =============================================================================

AUDIT = TAB / "source_tables"
AUDIT.mkdir(exist_ok=True)

for key, src in FILES.items():
    if src.suffix.lower() == ".tsv":
        shutil.copy2(
            src,
            AUDIT / src.name
        )


# =============================================================================
# 14. FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 120,
})

ANTI_COLOR = "#0072B2"
OUTCOME_COLOR = "#D55E00"
NEUTRAL = "#555555"

sub_colors = plt.cm.tab20(
    np.linspace(
        0,
        1,
        len(SUBFAMILY_ORDER)
    )
)


# =============================================================================
# 15. DRAW COMBINED FIGURE 3 A-D
# =============================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(19, 14)
)

axA, axB = axes[0]
axC, axD = axes[1]


# -----------------------------------------------------------------------------
# PANEL A: SUBFAMILY HEATMAP
# -----------------------------------------------------------------------------

arr = sub_z.to_numpy(
    dtype=float
)

vmax = np.nanmax(
    np.abs(arr)
)

im = axA.imshow(
    arr,
    aspect="auto",
    interpolation="nearest",
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax
)

axA.set_xticks(
    np.arange(
        len(SUBFAMILY_ORDER)
    )
)

axA.set_xticklabels(
    [
        DISPLAY[x]
        for x in SUBFAMILY_ORDER
    ],
    rotation=48,
    ha="right"
)

axA.set_yticks(
    np.arange(10)
)

axA.set_yticklabels(
    np.arange(1, 11)
)

axA.set_xlabel(
    "ROI subfamily"
)

axA.set_ylabel(
    "G1 position"
)

axA.set_title(
    "Subfamily-level connectivity organization along G1",
    loc="left",
    fontweight="bold"
)

cb = fig.colorbar(
    im,
    ax=axA,
    shrink=0.82,
    pad=0.02
)

cb.set_label(
    "Within-subfamily z-score"
)

axA.text(
    -0.10,
    1.06,
    "A",
    transform=axA.transAxes,
    fontsize=17,
    fontweight="bold"
)


# -----------------------------------------------------------------------------
# PANEL B: SUBFAMILY RAW PROFILES
# -----------------------------------------------------------------------------

x = np.arange(
    1,
    11
)

for i, name in enumerate(
    SUBFAMILY_ORDER
):

    label = DISPLAY[name]

    line_style = (
        "--"
        if name == "Midbrain_pallidal"
        else "-"
    )

    line_width = (
        2.3
        if name == "Midbrain_pallidal"
        else 1.7
    )

    axB.plot(
        x,
        sub_raw[name].to_numpy(
            dtype=float
        ),
        marker="o",
        markersize=3.5,
        linewidth=line_width,
        linestyle=line_style,
        color=sub_colors[i],
        label=label
    )

axB.set_xticks(
    np.arange(
        1,
        11
    )
)

axB.set_xlabel(
    "G1 position"
)

axB.set_ylabel(
    "Mean Fisher-z connectivity"
)

axB.set_title(
    "Subfamily connectivity profiles",
    loc="left",
    fontweight="bold"
)

axB.legend(
    frameon=False,
    ncol=2,
    loc="upper right",
    fontsize=7.7
)

axB.text(
    -0.10,
    1.06,
    "B",
    transform=axB.transAxes,
    fontsize=17,
    fontweight="bold"
)


# -----------------------------------------------------------------------------
# PANEL C: SUBJECT-LEVEL STAGE PROFILES
# -----------------------------------------------------------------------------

for stage, color in [
    ("Outcome", OUTCOME_COLOR),
    ("Anticipation", ANTI_COLOR),
]:

    g = (
        profile_final.loc[
            profile_final[
                "Stage"
            ].eq(stage)
        ]
        .sort_values(
            "G1_bin"
        )
    )

    xx = g[
        "G1_bin"
    ].to_numpy(dtype=float)

    yy = g[
        "mean_FC"
    ].to_numpy(dtype=float)

    lo = g[
        "CI95_low"
    ].to_numpy(dtype=float)

    hi = g[
        "CI95_high"
    ].to_numpy(dtype=float)

    axC.plot(
        xx,
        yy,
        marker="o",
        markersize=5,
        linewidth=2.4,
        color=color,
        label=stage
    )

    axC.fill_between(
        xx,
        lo,
        hi,
        color=color,
        alpha=0.16
    )

axC.axvline(
    3.5,
    linestyle=":",
    linewidth=1.2,
    color="0.45"
)

axC.set_xticks(
    np.arange(
        1,
        11
    )
)

axC.set_xlabel(
    "G1 position"
)

axC.set_ylabel(
    "Mean Fisher-z connectivity"
)

axC.set_title(
    "Reward-stage connectivity profiles across participants",
    loc="left",
    fontweight="bold"
)

axC.legend(
    frameon=False
)

axC.text(
    3.62,
    axC.get_ylim()[1] * 0.97,
    "Stage profiles cross\nbetween bins 3–4",
    fontsize=8,
    va="top",
    color="0.35"
)

axC.text(
    -0.10,
    1.06,
    "C",
    transform=axC.transAxes,
    fontsize=17,
    fontweight="bold"
)


# -----------------------------------------------------------------------------
# PANEL D: ANTI - OUTCOME DIFFERENCE PROFILE
# -----------------------------------------------------------------------------

xx = bin_final[
    "G1_bin"
].to_numpy(dtype=float)

yy = bin_final[
    "Anti_minus_Outcome"
].to_numpy(dtype=float)

lo = bin_final[
    "difference_CI95_low"
].to_numpy(dtype=float)

hi = bin_final[
    "difference_CI95_high"
].to_numpy(dtype=float)

yerr = np.vstack([
    yy - lo,
    hi - yy
])

axD.axhline(
    0,
    color="0.35",
    linewidth=1.2,
    linestyle="--"
)

axD.plot(
    xx,
    yy,
    color=NEUTRAL,
    linewidth=1.8,
    zorder=2
)

axD.errorbar(
    xx,
    yy,
    yerr=yerr,
    fmt="none",
    ecolor="0.4",
    elinewidth=1.2,
    capsize=3,
    zorder=1
)

neg = yy < 0
pos = yy > 0

axD.scatter(
    xx[neg],
    yy[neg],
    s=55,
    color=OUTCOME_COLOR,
    zorder=3,
    label="Outcome preference"
)

axD.scatter(
    xx[pos],
    yy[pos],
    s=55,
    color=ANTI_COLOR,
    zorder=3,
    label="Anticipation preference"
)

axD.axvline(
    3.5,
    linestyle=":",
    linewidth=1.2,
    color="0.45"
)

axD.set_xticks(
    np.arange(
        1,
        11
    )
)

axD.set_xlabel(
    "G1 position"
)

axD.set_ylabel(
    "Anticipation − Outcome connectivity"
)

axD.set_title(
    "Relative reward-stage preference along G1",
    loc="left",
    fontweight="bold"
)

axD.legend(
    frameon=False,
    loc="upper left"
)

ylim = axD.get_ylim()

axD.text(
    9.9,
    ylim[1] * 0.82,
    "Anticipation\npreference",
    ha="right",
    va="top",
    color=ANTI_COLOR,
    fontweight="bold"
)

axD.text(
    1.1,
    ylim[0] * 0.82,
    "Outcome\npreference",
    ha="left",
    va="bottom",
    color=OUTCOME_COLOR,
    fontweight="bold"
)

axD.text(
    -0.10,
    1.06,
    "D",
    transform=axD.transAxes,
    fontsize=17,
    fontweight="bold"
)


fig.suptitle(
    "Reward-related connectivity organization along the principal striatal gradient",
    fontsize=15,
    fontweight="bold",
    y=0.995
)

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.98
    ],
    h_pad=3.2,
    w_pad=2.6
)

fig.savefig(
    FIG / "Figure3_ABCD_FINAL.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    FIG / "Figure3_ABCD_FINAL.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# 16. INDIVIDUAL PANEL C
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8.5, 6.2)
)

for stage, color in [
    ("Outcome", OUTCOME_COLOR),
    ("Anticipation", ANTI_COLOR),
]:

    g = (
        profile_final.loc[
            profile_final[
                "Stage"
            ].eq(stage)
        ]
        .sort_values(
            "G1_bin"
        )
    )

    xx = g["G1_bin"].to_numpy(float)
    yy = g["mean_FC"].to_numpy(float)
    lo = g["CI95_low"].to_numpy(float)
    hi = g["CI95_high"].to_numpy(float)

    ax.plot(
        xx,
        yy,
        marker="o",
        linewidth=2.5,
        markersize=5.5,
        color=color,
        label=stage
    )

    ax.fill_between(
        xx,
        lo,
        hi,
        color=color,
        alpha=0.16
    )

ax.axvline(
    3.5,
    linestyle=":",
    color="0.45"
)

ax.set_xticks(
    range(1, 11)
)

ax.set_xlabel(
    "G1 position"
)

ax.set_ylabel(
    "Mean Fisher-z connectivity"
)

ax.set_title(
    "Anticipation and Outcome connectivity along striatal G1",
    fontweight="bold"
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    FIG / "Figure3C_stage_profiles_FINAL.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# 17. INDIVIDUAL PANEL D
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8.5, 6.2)
)

ax.axhline(
    0,
    color="0.35",
    linewidth=1.2,
    linestyle="--"
)

ax.plot(
    xx,
    yy,
    color=NEUTRAL,
    linewidth=1.8
)

ax.errorbar(
    xx,
    yy,
    yerr=yerr,
    fmt="none",
    ecolor="0.4",
    capsize=3
)

ax.scatter(
    xx[neg],
    yy[neg],
    s=65,
    color=OUTCOME_COLOR,
    label="Outcome preference"
)

ax.scatter(
    xx[pos],
    yy[pos],
    s=65,
    color=ANTI_COLOR,
    label="Anticipation preference"
)

ax.axvline(
    3.5,
    linestyle=":",
    color="0.45"
)

ax.set_xticks(
    range(1, 11)
)

ax.set_xlabel(
    "G1 position"
)

ax.set_ylabel(
    "Anticipation − Outcome connectivity"
)

ax.set_title(
    "Reward-stage preference shifts along striatal G1",
    fontweight="bold"
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    FIG / "Figure3D_stage_difference_FINAL.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# 18. REPORT HELPERS
# =============================================================================

def fmt_num(x, digits=4):

    if pd.isna(x):
        return ""

    return f"{float(x):.{digits}f}"


def fmt_p(x):

    if pd.isna(x):
        return ""

    x = float(x)

    if x == 0:
        return "<1e-300"

    if x < 0.0001:
        return f"{x:.2e}"

    return f"{x:.4f}"


def markdown_table(df):

    temp = df.copy()

    cols = [
        str(c)
        for c in temp.columns
    ]

    lines = []

    lines.append(
        "| " +
        " | ".join(cols) +
        " |"
    )

    lines.append(
        "| " +
        " | ".join(
            ["---"] * len(cols)
        ) +
        " |"
    )

    for _, row in temp.iterrows():

        vals = []

        for c in temp.columns:

            v = row[c]

            s = str(v)

            s = s.replace(
                "|",
                "\\|"
            )

            vals.append(s)

        lines.append(
            "| " +
            " | ".join(vals) +
            " |"
        )

    return "\n".join(lines)


def add_section(lines, title):

    lines.append("")
    lines.append(
        f"## {title}"
    )
    lines.append("")


# =============================================================================
# 19. BUILD FORMATTED REPORT TABLES
# =============================================================================

# -----------------------------------------------------------------------------
# TABLE 1 formatting
# -----------------------------------------------------------------------------

r1 = roi_comp.copy()

# -----------------------------------------------------------------------------
# TABLE 2 formatting
# -----------------------------------------------------------------------------

r2 = subtrend_final[
    [
        "subfamily",
        "slope_per_bin",
        "r",
        "p_linear",
        "direction",
    ]
].copy()

r2["slope_per_bin"] = (
    r2["slope_per_bin"]
    .map(lambda x: fmt_num(x, 5))
)

r2["r"] = (
    r2["r"]
    .map(lambda x: fmt_num(x, 3))
)

r2["p_linear"] = (
    r2["p_linear"]
    .map(fmt_p)
)

# -----------------------------------------------------------------------------
# TABLE 3 formatting
# -----------------------------------------------------------------------------

r3 = profile_final.copy()

for c in [
    "mean_FC",
    "SD",
    "SEM",
    "CI95_low",
    "CI95_high",
]:
    r3[c] = (
        r3[c]
        .map(lambda x: fmt_num(x, 4))
    )

# -----------------------------------------------------------------------------
# TABLE 4 formatting
# -----------------------------------------------------------------------------

r4 = fixed_final.copy()

for c in [
    "beta",
    "SE",
    "CI95_low",
    "CI95_high",
]:
    r4[c] = (
        r4[c]
        .map(lambda x: fmt_num(x, 6))
    )

r4["p"] = (
    r4["p"]
    .map(fmt_p)
)

# -----------------------------------------------------------------------------
# TABLE 5 formatting
# -----------------------------------------------------------------------------

r5 = simple_final.copy()

for c in [
    "slope",
    "SE",
    "CI95_low",
    "CI95_high",
    "z",
]:
    r5[c] = (
        r5[c]
        .map(lambda x: fmt_num(x, 6))
    )

r5["p"] = (
    r5["p"]
    .map(fmt_p)
)

# -----------------------------------------------------------------------------
# TABLE 6 model comparison
# -----------------------------------------------------------------------------

r6 = model_comp.copy()

for c in [
    "logLik",
    "AIC",
    "BIC",
]:
    r6[c] = (
        r6[c]
        .map(lambda x: fmt_num(x, 2))
    )

# -----------------------------------------------------------------------------
# TABLE 7 paired slope
# -----------------------------------------------------------------------------

r7 = paired.copy()

keep7 = [
    "N_subjects",
    "Anticipation_mean_slope",
    "Anticipation_SD_slope",
    "Outcome_mean_slope",
    "Outcome_SD_slope",
    "Anti_minus_Outcome_mean_difference",
    "difference_CI95_low",
    "difference_CI95_high",
    "paired_t",
    "paired_t_df",
    "paired_t_p",
    "Cohen_dz",
    "Wilcoxon_p",
    "N_Anti_slope_greater_than_Outcome",
    "proportion_Anti_slope_greater",
]

r7 = r7[
    keep7
].copy()

for c in [
    "Anticipation_mean_slope",
    "Anticipation_SD_slope",
    "Outcome_mean_slope",
    "Outcome_SD_slope",
    "Anti_minus_Outcome_mean_difference",
    "difference_CI95_low",
    "difference_CI95_high",
]:
    r7[c] = (
        r7[c]
        .map(lambda x: fmt_num(x, 6))
    )

r7["paired_t"] = (
    r7["paired_t"]
    .map(lambda x: fmt_num(x, 3))
)

r7["paired_t_p"] = (
    r7["paired_t_p"]
    .map(fmt_p)
)

r7["Cohen_dz"] = (
    r7["Cohen_dz"]
    .map(lambda x: fmt_num(x, 3))
)

r7["Wilcoxon_p"] = (
    r7["Wilcoxon_p"]
    .map(fmt_p)
)

r7[
    "proportion_Anti_slope_greater"
] = (
    r7[
        "proportion_Anti_slope_greater"
    ]
    .map(
        lambda x:
        f"{100*x:.1f}%"
    )
)

# -----------------------------------------------------------------------------
# TABLE 8 binwise
# -----------------------------------------------------------------------------

r8 = bin_final[
    [
        "G1_bin",
        "Anticipation_mean",
        "Outcome_mean",
        "Anti_minus_Outcome",
        "difference_CI95_low",
        "difference_CI95_high",
        "paired_t",
        "paired_t_p_FDR_BH",
        "Cohen_dz",
        "Wilcoxon_p_FDR_BH",
        "direction",
        "robust_both_tests_FDR",
    ]
].copy()

for c in [
    "Anticipation_mean",
    "Outcome_mean",
    "Anti_minus_Outcome",
    "difference_CI95_low",
    "difference_CI95_high",
]:
    r8[c] = (
        r8[c]
        .map(lambda x: fmt_num(x, 4))
    )

r8["paired_t"] = (
    r8["paired_t"]
    .map(lambda x: fmt_num(x, 2))
)

r8["paired_t_p_FDR_BH"] = (
    r8["paired_t_p_FDR_BH"]
    .map(fmt_p)
)

r8["Cohen_dz"] = (
    r8["Cohen_dz"]
    .map(lambda x: fmt_num(x, 3))
)

r8["Wilcoxon_p_FDR_BH"] = (
    r8["Wilcoxon_p_FDR_BH"]
    .map(fmt_p)
)


# =============================================================================
# 20. GET KEY VALUES FOR NARRATIVE
# =============================================================================

k = mixed_key.iloc[0]
ps = paired.iloc[0]

interaction_beta = float(
    k[
        "Anti_minus_Outcome_slope_difference"
    ]
)

interaction_lo = float(
    k[
        "interaction_CI95_low"
    ]
)

interaction_hi = float(
    k[
        "interaction_CI95_high"
    ]
)

interaction_p = float(
    k[
        "Stage_x_G1_p"
    ]
)

anti_slope = float(
    k[
        "Anticipation_slope"
    ]
)

out_slope = float(
    k[
        "Outcome_slope"
    ]
)

anti_p = float(
    k[
        "Anticipation_slope_p"
    ]
)

out_p = float(
    k[
        "Outcome_slope_p"
    ]
)

paired_t = float(
    ps["paired_t"]
)

paired_df = int(
    ps["paired_t_df"]
)

paired_p = float(
    ps["paired_t_p"]
)

dz = float(
    ps["Cohen_dz"]
)

prop = float(
    ps[
        "proportion_Anti_slope_greater"
    ]
)

n_pos = int(
    ps[
        "N_Anti_slope_greater_than_Outcome"
    ]
)

sig_t_bins = (
    bin_final.loc[
        bin_final[
            "significant_paired_t_FDR_0.05"
        ],
        "G1_bin"
    ]
    .astype(int)
    .tolist()
)

sig_both_bins = (
    bin_final.loc[
        bin_final[
            "robust_both_tests_FDR"
        ],
        "G1_bin"
    ]
    .astype(int)
    .tolist()
)

negative_bins = (
    bin_final.loc[
        bin_final[
            "Anti_minus_Outcome"
        ] < 0,
        "G1_bin"
    ]
    .astype(int)
    .tolist()
)

positive_bins = (
    bin_final.loc[
        bin_final[
            "Anti_minus_Outcome"
        ] > 0,
        "G1_bin"
    ]
    .astype(int)
    .tolist()
)


# =============================================================================
# 21. GENERATE FULL RESULT REPORT
# =============================================================================

L = []

L.append(
    "# Figure 3 final analysis report"
)

L.append("")

L.append(
    "Primary dataset: 375 participants; "
    "2847 striatal voxels; 65 external reward-related ROIs; "
    "10 fixed group-G1 bins."
)

L.append("")

L.append(
    "Primary subject-level inferential model: "
    "`FC ~ Stage × G1 + (1 + G1 | Subject)`."
)


# -----------------------------------------------------------------------------
# TABLE 1
# -----------------------------------------------------------------------------

add_section(
    L,
    "Table 1. ROI and reward-stage composition"
)

L.append(
    markdown_table(r1)
)

L.append("")

L.append(
    "**结果说明：** 65 个 ROI 被整理为 14 个 subfamilies。"
    "其中 43 个 ROI 属于 Anticipation，19 个属于 Outcome。"
    "VTA、SNr 和 VP/VeP 共 3 个 Midbrain/pallidal ROI 保留在 "
    "65-ROI 和 subfamily 分析中，但不进入 Anticipation–Outcome 两阶段分析。"
)

L.append("")

L.append(
    "**结论：** Figure 3A–B 使用全部 65 个 ROI；"
    "Figure 3C–D 的 stage-level 分析严格使用 43 Anticipation vs 19 Outcome，"
    "不包含 3 个 midbrain/pallidal ROI。"
)


# -----------------------------------------------------------------------------
# TABLE 2
# -----------------------------------------------------------------------------

add_section(
    L,
    "Table 2. Group-level subfamily profiles along G1"
)

L.append(
    markdown_table(r2)
)

L.append("")

sig_sub = (
    sub_trend.loc[
        sub_trend["p_linear"] < 0.05,
        "subfamily"
    ]
    .map(DISPLAY)
    .dropna()
    .tolist()
)

L.append(
    "**结果说明：** 不同 subfamily 沿 G1 呈现不同方向和不同程度的变化，"
    "说明 reward-related connectivity 并不是由所有 ROI 共享一条完全相同的梯度。"
    + (
        " 在 10-bin group-average 线性描述中，p<.05 的 subfamilies 包括："
        + "、".join(sig_sub)
        + "。"
        if len(sig_sub) > 0
        else ""
    )
)

L.append("")

L.append(
    "**结论：** Figure 3A–B 主要用于展示 subfamily-level 的功能异质性。"
    "这里的 p 值来自 10 个 group-average bins，属于 group-level profile 描述，"
    "不作为 375 名被试总体推断的核心统计。"
)


# -----------------------------------------------------------------------------
# TABLE 3
# -----------------------------------------------------------------------------

add_section(
    L,
    "Table 3. Subject-level Anticipation and Outcome profiles"
)

L.append(
    markdown_table(r3)
)

L.append("")

L.append(
    "**结果说明：** Outcome 在低 G1 端具有更高的平均 connectivity，"
    "随后向高 G1 快速下降；Anticipation 也总体下降，但幅度明显较浅。"
    "两条平均 profile 的相对顺序在 bin 3 与 bin 4 之间发生反转。"
)

L.append("")

L.append(
    "**结论：** 数据不是“一条上升、一条下降”，"
    "而是 Outcome 下降更快，因此 reward-stage 的相对优势沿 G1 从 "
    "Outcome-dominant 转变为 Anticipation-dominant。"
)


# -----------------------------------------------------------------------------
# TABLE 4
# -----------------------------------------------------------------------------

add_section(
    L,
    "Table 4. Primary random-slope mixed-effects model"
)

L.append(
    markdown_table(r4)
)

L.append("")

L.append(
    "**结果说明：** 主模型允许每名被试具有自己的 baseline 和 G1 slope。"
    f"最关键的 Stage × G1 interaction 为 β={interaction_beta:.6f}, "
    f"95% CI [{interaction_lo:.6f}, {interaction_hi:.6f}], "
    f"p={fmt_p(interaction_p)}。"
)

L.append("")

L.append(
    "**结论：** Anticipation 与 Outcome 沿纹状体 G1 的 connectivity trajectories "
    "显著不同。正的 interaction 表示 Anticipation slope 比 Outcome slope 更高，"
    "即 Outcome 的负向下降显著更陡。"
)


# -----------------------------------------------------------------------------
# TABLE 5
# -----------------------------------------------------------------------------

add_section(
    L,
    "Table 5. Stage-specific simple slopes"
)

L.append(
    markdown_table(r5)
)

L.append("")

L.append(
    f"**结果说明：** Outcome slope = {out_slope:.6f}，"
    f"Anticipation slope = {anti_slope:.6f}。"
    "两者都为负值，但 Outcome 的下降幅度约为 Anticipation 的 4 倍以上。"
)

L.append("")

L.append(
    "**结论：** 两种 reward stages 的绝对 connectivity 都总体向高 G1 减弱，"
    "但 Outcome 表现出明显更陡的负向 gradient。"
    "因此真正沿 G1 增强的是 Anticipation 相对于 Outcome 的相对优势，"
    "而不是 Anticipation 的绝对 FC 本身。"
)


# -----------------------------------------------------------------------------
# TABLE 6
# -----------------------------------------------------------------------------

add_section(
    L,
    "Table 6. Random-effects structure and model comparison"
)

L.append(
    markdown_table(r6)
)

L.append("")

ri_aic = float(
    model_comp.loc[
        model_comp[
            "model"
        ].eq(
            "Random intercept"
        ),
        "AIC"
    ].iloc[0]
)

rs_aic = float(
    model_comp.loc[
        model_comp[
            "model"
        ].eq(
            "Random intercept + G1 slope"
        ),
        "AIC"
    ].iloc[0]
)

L.append(
    "**结果说明：** Random-slope model 收敛正常，"
    f"AIC 从 {ri_aic:.2f} 降至 {rs_aic:.2f}，"
    "说明允许不同被试具有不同 G1 slopes 明显改善模型拟合。"
)

L.append("")

L.append(
    "**结论：** 被试之间确实存在 G1 slope 的个体差异，"
    "因此 `(1 + G1 | Subject)` 比仅使用 random intercept 更适合作为当前主模型。"
)


# -----------------------------------------------------------------------------
# TABLE 7
# -----------------------------------------------------------------------------

add_section(
    L,
    "Table 7. Individual paired-slope analysis"
)

L.append(
    markdown_table(r7)
)

L.append("")

L.append(
    f"**结果说明：** 每名被试分别计算 Anticipation 和 Outcome 的 10-bin G1 slope，"
    f"再进行被试内配对比较。结果为 t({paired_df})={paired_t:.2f}, "
    f"p={fmt_p(paired_p)}, Cohen's dz={dz:.2f}。"
    f"{n_pos}/375（{100*prop:.1f}%）被试表现为 "
    "Anticipation slope > Outcome slope。"
)

L.append("")

L.append(
    "**结论：** Stage × G1 effect 不只是 group-average pattern。"
    "它在绝大多数被试中方向一致，并具有较大的被试内效应量。"
)


# -----------------------------------------------------------------------------
# TABLE 8
# -----------------------------------------------------------------------------

add_section(
    L,
    "Table 8. Subject-level bin-wise Anticipation vs Outcome comparisons"
)

L.append(
    markdown_table(r8)
)

L.append("")

L.append(
    "**结果说明：** paired t-test 经 BH-FDR 后，"
    f"显著 bins 为 {sig_t_bins}。"
    f"其中 bins {negative_bins} 的差值为负，表示 Outcome > Anticipation；"
    f"bins {positive_bins} 的差值为正，表示 Anticipation > Outcome。"
    f"参数检验与 Wilcoxon-FDR 同时支持的 bins 为 {sig_both_bins}。"
)

L.append("")

L.append(
    "**结论：** reward-stage preference 沿 G1 呈系统性反转。"
    "Bin 1–3 主要表现为 Outcome preference；"
    "Bin 3–4 之间发生 crossover；"
    "随后逐渐转为 Anticipation preference。"
    "Bin 4 位于 crossover 附近，效应量很小，"
    "虽然 paired t-test 经 FDR 后显著，但 Wilcoxon-FDR 不显著，"
    "因此不应过度强调该单一 bin。"
)


# -----------------------------------------------------------------------------
# OVERALL
# -----------------------------------------------------------------------------

add_section(
    L,
    "Overall conclusion"
)

L.append(
    "1. 不同 reward-related subfamilies 与纹状体 G1 具有明显不同的 connectivity profiles。"
)

L.append("")

L.append(
    "2. 在 reward-stage 层面，Anticipation 和 Outcome 都表现为总体负向 G1 slopes，"
    "但 Outcome 的下降显著更陡。"
)

L.append("")

L.append(
    "3. 因此沿 G1 发生的关键变化并不是 Anticipation 绝对 connectivity 持续增强，"
    "而是 Anticipation 相对于 Outcome 的 connectivity advantage 持续增加。"
)

L.append("")

L.append(
    "4. 这种相对 stage preference 从低-G1 的 Outcome-dominant pattern，"
    "经 bin 3–4 附近的 crossover，转变为高-G1 的 Anticipation-dominant pattern。"
)

L.append("")

L.append(
    "5. 该效应在 random-slope mixed model 和 individual paired-slope analysis 中"
    "得到一致的 375-subject-level 支持。"
)


report_text = "\n".join(L)

(
    REPORT /
    "Figure3_FINAL_results_report.md"
).write_text(
    report_text,
    encoding="utf-8"
)

(
    REPORT /
    "Figure3_FINAL_results_report.txt"
).write_text(
    report_text,
    encoding="utf-8"
)


# =============================================================================
# 22. FIGURE CAPTION
# =============================================================================

caption = """
Figure 3. Reward-related connectivity organization along the principal striatal gradient (G1).

(A) Subfamily-level connectivity profiles across ten equally populated G1 bins. Values are standardized within each subfamily across bins and therefore depict relative profile shape rather than absolute between-subfamily connectivity strength.

(B) Mean Fisher-z connectivity profiles for the 14 ROI subfamilies. The midbrain/pallidal subfamily (VTA, SNr, and VP/VeP) is retained in the subfamily-level analysis.

(C) Subject-level mean connectivity profiles for outcome- and anticipation-related ROIs across G1. Shaded regions indicate 95% confidence intervals across participants. Outcome-related connectivity decreases substantially more steeply along G1 than anticipation-related connectivity.

(D) Subject-level Anticipation minus Outcome connectivity difference across G1. Negative values indicate relative Outcome preference and positive values indicate relative Anticipation preference. The stage-preference profile changes sign between G1 bins 3 and 4, indicating a shift from Outcome-dominant connectivity at the low-G1 end toward Anticipation-dominant connectivity at the high-G1 end. VTA, SNr, and VP/VeP were excluded from all Anticipation–Outcome stage-level analyses.

Primary inference was based on a linear mixed-effects model with subject-specific random intercepts and G1 slopes: FC ~ Stage × G1 + (1 + G1 | Subject).
""".strip()

(
    REPORT /
    "Figure3_FINAL_caption.txt"
).write_text(
    caption + "\n",
    encoding="utf-8"
)


# =============================================================================
# 23. FINAL CONSOLE OUTPUT
# =============================================================================

print("\n" + "=" * 100)
print("FINAL FIGURE 3 COMPLETE")
print("=" * 100)

print("\nFigures:")
for p in sorted(
    FIG.glob("*")
):
    print(
        f"  {p}"
    )

print("\nFinal tables:")
for p in sorted(
    TAB.glob("Table*.tsv")
):
    print(
        f"  {p}"
    )

print("\nReport:")
print(
    REPORT /
    "Figure3_FINAL_results_report.txt"
)

print("\nCaption:")
print(
    REPORT /
    "Figure3_FINAL_caption.txt"
)

print("\n" + "=" * 100)
print("PRIMARY RESULT")
print("=" * 100)

print(
    f"Outcome slope       = "
    f"{out_slope:.8f}"
)

print(
    f"Anticipation slope  = "
    f"{anti_slope:.8f}"
)

print(
    f"Stage x G1 beta     = "
    f"{interaction_beta:.8f}"
)

print(
    f"95% CI              = "
    f"[{interaction_lo:.8f}, "
    f"{interaction_hi:.8f}]"
)

print(
    f"Stage x G1 p        = "
    f"{interaction_p:.12g}"
)

print(
    f"Paired slope t({paired_df}) = "
    f"{paired_t:.4f}"
)

print(
    f"Cohen dz            = "
    f"{dz:.4f}"
)

print(
    f"Anti slope > Outcome = "
    f"{n_pos}/375 "
    f"({100*prop:.1f}%)"
)

print("\nDONE.")
