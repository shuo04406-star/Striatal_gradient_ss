"""Create STEP 2 supervisor-meeting figures from frozen analysis outputs.

This script is presentation-only. It reads existing STEP 2 results and the
canonical stage-level table, performs no new statistical test, and writes only
to the configured supplementary output tree.
"""

from __future__ import annotations

import os
from pathlib import Path
import json
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np
import pandas as pd


RESULTS = Path(os.environ["STRIATAL_VARIANCE_OUTPUT"])
STAGE_TABLE = Path(os.environ["STRIATAL_STAGE_FC_TABLE"])
OUTPUT = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP2_variance_meeting'))

SUBJECT_FILE = RESULTS / "subject_level_dispersion.tsv"
BS_FILE = RESULTS / "between_subject_variance_summary.tsv"
BR_FILE = RESULTS / "between_region_variance_summary.tsv"
TEST_FILE = RESULTS / "variance_statistical_tests.tsv"
SENS_FILE = RESULTS / "variance_sensitivity_analysis.tsv"

HC = "#5B8DB8"
SZ = "#E07A5F"
INK = "#263238"
MUTED = "#62727B"
GRID = "#DDE3E8"
BS_COLOR = "#6F63A7"
BR_COLOR = "#3B8C83"

STAGES = ["Anticipation", "Outcome"]
GROUPS = ["HC", "SZ"]
GROUP_COLORS = {"HC": HC, "SZ": SZ}
RNG = np.random.default_rng(20261011)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.edgecolor": "#8B98A1",
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_and_validate() -> tuple[pd.DataFrame, ...]:
    required = [SUBJECT_FILE, BS_FILE, BR_FILE, TEST_FILE, SENS_FILE, STAGE_TABLE]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))

    subject = pd.read_csv(SUBJECT_FILE, sep="\t")
    bs = pd.read_csv(BS_FILE, sep="\t")
    br = pd.read_csv(BR_FILE, sep="\t")
    tests = pd.read_csv(TEST_FILE, sep="\t")
    sens = pd.read_csv(SENS_FILE, sep="\t")
    stage = pd.read_csv(STAGE_TABLE, sep="\t")

    if subject.shape != (128, 18):
        raise ValueError(f"Unexpected subject-level shape: {subject.shape}")
    if subject.isna().any().any():
        raise ValueError("Missing values in subject_level_dispersion.tsv")
    if subject.duplicated(["Subject", "Stage"]).any():
        raise ValueError("Duplicate Subject–Stage rows in subject_level_dispersion.tsv")
    counts = subject.groupby(["Stage", "Group"]).size().to_dict()
    expected_counts = {(s, g): 32 for s in STAGES for g in GROUPS}
    if counts != expected_counts:
        raise ValueError(f"Unexpected subject counts: {counts}")
    roi_counts = subject.groupby("Stage")["roi_count"].unique().to_dict()
    if not (
        np.array_equal(roi_counts["Anticipation"], np.array([43]))
        and np.array_equal(roi_counts["Outcome"], np.array([19]))
    ):
        raise ValueError(f"Unexpected fixed ROI counts: {roi_counts}")
    if (subject["regional_dispersion"] <= 0).any():
        raise ValueError("Non-positive regional dispersion found")

    if stage.shape[0] != 1280 or stage.duplicated(["Subject", "Stage", "Bin"]).any():
        raise ValueError(f"Unexpected stage-level table shape or duplicates: {stage.shape}")
    stage_counts = stage.groupby(["Stage", "Group"])["Subject"].nunique().to_dict()
    if stage_counts != expected_counts or stage["Bin"].nunique() != 10:
        raise ValueError("Stage-level subject/bin completeness check failed")

    primary = tests.loc[tests["role"].eq("primary")].copy()
    expected = json.loads(os.environ["STRIATAL_VARIANCE_REFERENCE_JSON"])
    for test_id, values in expected.items():
        row = primary.loc[primary["test_id"].eq(test_id)]
        if len(row) != 1:
            raise ValueError(f"Missing or repeated primary test: {test_id}")
        observed = row[["estimate", "ci_low", "ci_high", "holm_adjusted_p"]].iloc[0].to_numpy(float)
        if not np.allclose(observed, np.array(values), atol=5e-7, rtol=0):
            raise ValueError(f"Frozen result mismatch for {test_id}: {observed}")

    return subject, bs, br, tests, sens, stage


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.115,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
    )


def jittered_scatter(
    ax: plt.Axes,
    values: np.ndarray,
    x: float,
    color: str,
    *,
    width: float = 0.085,
    size: float = 29,
    alpha: float = 0.76,
    edgecolor: str = "white",
    linewidth: float = 0.45,
    zorder: int = 3,
) -> np.ndarray:
    jitter = RNG.uniform(-width, width, size=len(values))
    xs = x + jitter
    ax.scatter(
        xs,
        values,
        s=size,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    return xs


def grouped_box_scatter(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_col: str,
    *,
    flag_col: str | None = None,
) -> None:
    arrays = [data.loc[data["Group"].eq(g), value_col].to_numpy(float) for g in GROUPS]
    bp = ax.boxplot(
        arrays,
        positions=[0, 1],
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": INK, "linewidth": 1.5},
        whiskerprops={"color": MUTED, "linewidth": 1.0},
        capprops={"color": MUTED, "linewidth": 1.0},
        boxprops={"edgecolor": MUTED, "linewidth": 1.0},
        zorder=2,
    )
    for patch, g in zip(bp["boxes"], GROUPS):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.22)

    for x, g, vals in zip([0, 1], GROUPS, arrays):
        group_data = data.loc[data["Group"].eq(g)].copy()
        xs = jittered_scatter(ax, vals, x, GROUP_COLORS[g])
        if flag_col is not None:
            flags = group_data[flag_col].astype(bool).to_numpy()
            if flags.any():
                ax.scatter(
                    xs[flags],
                    vals[flags],
                    s=71,
                    facecolor="none",
                    edgecolor=INK,
                    linewidth=1.35,
                    zorder=5,
                )
    ax.set_xticks([0, 1], ["HC\n(n=32)", "SZ\n(n=32)"])
    ax.set_xlim(-0.55, 1.55)
    clean_axis(ax)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Preserve the fixed 16:9 canvas for direct placement in meeting slides.
    fig.savefig(OUTPUT / f"{stem}.png", dpi=320, facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def primary_row(tests: pd.DataFrame, analysis: str, stage_name: str) -> pd.Series:
    row = tests.loc[
        tests["role"].eq("primary")
        & tests["analysis"].eq(analysis)
        & tests["stage"].eq(stage_name)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one primary row for {analysis}, {stage_name}")
    return row.iloc[0]


def figure_1(subject: pd.DataFrame, tests: pd.DataFrame, stage: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.333, 7.5))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.15, 1.05],
        left=0.075,
        right=0.985,
        top=0.855,
        bottom=0.15,
        hspace=0.43,
        wspace=0.22,
    )
    profile_axes = []
    deviation_axes = []
    for col, stage_name in enumerate(STAGES):
        ax = fig.add_subplot(gs[0, col], sharey=profile_axes[0] if profile_axes else None)
        profile_axes.append(ax)
        for g in GROUPS:
            d = stage.loc[stage["Stage"].eq(stage_name) & stage["Group"].eq(g)]
            wide = d.pivot(index="Subject", columns="Bin", values="FC").sort_index(axis=1)
            x = wide.columns.to_numpy(float)
            mean = wide.mean(axis=0).to_numpy(float)
            sd = wide.std(axis=0, ddof=1).to_numpy(float)
            ax.fill_between(x, mean - sd, mean + sd, color=GROUP_COLORS[g], alpha=0.16, linewidth=0)
            ax.plot(x, mean, color=GROUP_COLORS[g], linewidth=2.3, marker="o", markersize=3.7, label=g)
        row = primary_row(tests, "between_subject", stage_name)
        ax.set_title(stage_name, loc="left", fontweight="bold", pad=13)
        ax.text(
            0.98,
            0.96,
            f"SZ/HC = {row['estimate']:.3f}\n95% CI {row['ci_low']:.3f}–{row['ci_high']:.3f}\nHolm p = {row['holm_adjusted_p']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.2,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID, "alpha": 0.95},
        )
        ax.set_xlabel("G1 bin")
        if col == 0:
            ax.set_ylabel("Mean FC (Fisher-z)\nshading = between-subject SD")
        else:
            ax.tick_params(labelleft=False)
        ax.set_xticks(range(1, 11))
        clean_axis(ax)
        add_panel_label(ax, chr(ord("A") + col))

        ax2 = fig.add_subplot(gs[1, col], sharey=deviation_axes[0] if deviation_axes else None)
        deviation_axes.append(ax2)
        dsub = subject.loc[subject["Stage"].eq(stage_name)]
        grouped_box_scatter(ax2, dsub, "profile_deviation_contribution")
        if col == 0:
            ax2.set_ylabel("Deviation contribution\n(Fisher-z²)")
        else:
            ax2.tick_params(labelleft=False)
        ax2.set_title("Subject deviation from own group mean profile", loc="left", fontsize=10.2, pad=4)

    fig.suptitle("Between-subject variability of 10-bin FC profiles", x=0.075, y=0.965, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.075,
        0.918,
        "Group mean profile ± SD (top) and all subject-level deviation contributions (bottom)",
        ha="left",
        va="center",
        color=MUTED,
        fontsize=10.5,
    )
    handles = [
        Line2D([0], [0], color=HC, lw=2.5, marker="o", markersize=4, label="HC (n=32)"),
        Line2D([0], [0], color=SZ, lw=2.5, marker="o", markersize=4, label="SZ (n=32)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.982, 0.968), frameon=False, ncol=2)
    fig.text(
        0.5,
        0.055,
        "Subject dots are n/(n−1)-scaled squared deviations from the subject's own group mean 10-bin profile; "
        "their group mean equals V. They are not individual between-subject variances.",
        ha="center",
        va="center",
        fontsize=9.2,
        color=MUTED,
    )
    save_figure(fig, "Figure_1_BetweenSubject")


def figure_2(subject: pd.DataFrame, tests: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.333, 7.5), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.19, wspace=0.20)
    ymax = subject["regional_dispersion"].max() * 1.15
    for i, (ax, stage_name) in enumerate(zip(axes, STAGES)):
        d = subject.loc[subject["Stage"].eq(stage_name)]
        grouped_box_scatter(ax, d, "regional_dispersion")
        row = primary_row(tests, "between_region", stage_name)
        roi_n = int(d["roi_count"].iloc[0])
        ax.set_ylim(0, ymax)
        ax.set_title(f"{stage_name}  ·  {roi_n} fixed ROIs", loc="left", fontweight="bold", pad=14)
        ax.text(
            0.03,
            0.97,
            "Adjusted HC3 model\n"
            f"SZ/HC = {row['estimate']:.3f}  (95% CI {row['ci_low']:.3f}–{row['ci_high']:.3f})\n"
            f"Holm p = {row['holm_adjusted_p']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.6,
            linespacing=1.38,
            bbox={"boxstyle": "round,pad=0.38", "facecolor": "white", "edgecolor": GRID, "alpha": 0.96},
        )
        if i == 0:
            ax.set_ylabel("Regional dispersion (Fisher-z²)\nmean ROI sample variance across 10 bins")
        else:
            ax.tick_params(labelleft=False)
        add_panel_label(ax, chr(ord("A") + i))

    fig.suptitle("Between-region variability: every subject retained", x=0.085, y=0.955, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.085,
        0.90,
        "Dots and boxplots show unadjusted subject-level regional dispersion; no extreme value is hidden.",
        ha="left",
        fontsize=10.7,
        color=MUTED,
    )
    fig.text(
        0.5,
        0.075,
        "Statistical annotations: covariate-adjusted geometric mean ratios from log(regional dispersion) ~ Group + Age + Sex + Mean FD,\n"
        "with HC3 confidence intervals. The plotted points themselves are unadjusted.",
        ha="center",
        va="center",
        fontsize=9.4,
        color=MUTED,
    )
    save_figure(fig, "Figure_2_BetweenRegion_SubjectScatter")


def figure_3(tests: pd.DataFrame) -> None:
    rows = [
        ("between_subject", "Anticipation", "Between-subject  ·  Anticipation", 4.45, BS_COLOR),
        ("between_subject", "Outcome", "Between-subject  ·  Outcome", 3.45, BS_COLOR),
        ("between_region", "Anticipation", "Between-region  ·  Anticipation", 1.55, BR_COLOR),
        ("between_region", "Outcome", "Between-region  ·  Outcome", 0.55, BR_COLOR),
    ]
    fig = plt.figure(figsize=(13.333, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.25], left=0.19, right=0.98, top=0.82, bottom=0.27, wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    txt = fig.add_subplot(gs[0, 1], sharey=ax)

    ax.axhspan(2.8, 5.0, color=BS_COLOR, alpha=0.055, zorder=0)
    ax.axhspan(0.0, 2.2, color=BR_COLOR, alpha=0.055, zorder=0)
    for analysis, stage_name, label, y, color in rows:
        row = primary_row(tests, analysis, stage_name)
        est, lo, hi = float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])
        ax.errorbar(
            est,
            y,
            xerr=np.array([[est - lo], [hi - est]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=4,
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4,
        )
        txt.text(0.02, y, f"{est:.3f}  [{lo:.3f}, {hi:.3f}]", va="center", fontsize=10.3, color=INK)
        txt.text(0.76, y, f"{float(row['holm_adjusted_p']):.3f}", va="center", ha="center", fontsize=10.3, color=INK)

    ax.axvline(1.0, color=INK, linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 2.7)
    ax.set_ylim(0.0, 5.0)
    ax.set_yticks([r[3] for r in rows], [r[2] for r in rows])
    ax.set_xticks([0.75, 1.0, 1.5, 2.0, 2.5])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("SZ / HC variance ratio  (log scale)", labelpad=9)
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(1.0, 4.88, "No difference", ha="center", va="bottom", fontsize=8.8, color=MUTED)

    txt.set_xlim(0, 1)
    txt.set_ylim(0, 5.0)
    txt.axis("off")
    txt.text(0.02, 4.90, "Ratio  [95% CI]", fontsize=9.7, fontweight="bold", va="bottom")
    txt.text(0.76, 4.90, "Holm p", fontsize=9.7, fontweight="bold", va="bottom", ha="center")
    txt.axhline(2.5, color=GRID, linewidth=1.0)

    fig.suptitle("Summary of the four primary variance comparisons", x=0.075, y=0.955, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.075,
        0.900,
        "All 95% confidence intervals cross 1; no primary comparison passes Holm correction.",
        ha="left",
        fontsize=10.7,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.155,
        "Between-subject",
        color=BS_COLOR,
        fontsize=10.2,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.120,
        "Trace of the 10-bin between-subject covariance matrix / 10; subject bootstrap CI.",
        color=MUTED,
        fontsize=9.1,
    )
    fig.text(
        0.54,
        0.155,
        "Between-region",
        color=BR_COLOR,
        fontsize=10.2,
        fontweight="bold",
    )
    fig.text(
        0.54,
        0.120,
        "Age/Sex/Mean FD-adjusted geometric mean ratio of subject regional dispersion; HC3 CI.",
        color=MUTED,
        fontsize=9.1,
    )
    fig.text(
        0.075,
        0.065,
        "The two analysis families use different variance definitions and confidence-interval procedures; ratios are comparable only as SZ/HC effect directions.",
        color=MUTED,
        fontsize=8.9,
    )
    save_figure(fig, "Figure_3_Variance_Summary")


def sens_value(sens: pd.DataFrame, analysis: str, stage_name: str, sensitivity: str) -> float:
    row = sens.loc[
        sens["analysis"].eq(analysis)
        & sens["stage"].eq(stage_name)
        & sens["sensitivity"].eq(sensitivity)
    ]
    if len(row) != 1:
        raise ValueError(f"Missing sensitivity result: {analysis}, {stage_name}, {sensitivity}")
    return float(row["estimate"].iloc[0])


def figure_4(subject: pd.DataFrame, tests: pd.DataFrame, sens: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.333, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.2], left=0.075, right=0.98, top=0.81, bottom=0.18, wspace=0.26)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    outcome = subject.loc[subject["Stage"].eq("Outcome")].copy()
    grouped_box_scatter(ax1, outcome, "profile_deviation_contribution", flag_col="extreme_profile_flag")
    ax1.set_ylabel("Profile deviation contribution (Fisher-z²)")
    ax1.set_title("Outcome: influential subjects", loc="left", fontweight="bold", pad=13)
    add_panel_label(ax1, "A")
    ax1.text(
        0.04,
        0.95,
        "Black rings = predefined extreme-profile flags\nHC: 1 flagged  ·  SZ: 5 flagged",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GRID, "alpha": 0.95},
    )
    ax1.text(
        0.5,
        -0.16,
        "All subjects: SZ/HC = 1.452\nExclude predefined extremes: SZ/HC = 0.995",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=9.7,
        fontweight="bold",
    )

    entries = [
        (
            "BS Outcome",
            float(primary_row(tests, "between_subject", "Outcome")["estimate"]),
            sens_value(sens, "between_subject", "Outcome", "exclude_predefined_extreme_subjects"),
            2.5,
            BS_COLOR,
        ),
        (
            "BR Anticipation",
            float(primary_row(tests, "between_region", "Anticipation")["estimate"]),
            sens_value(sens, "between_region", "Anticipation", "exclude_predefined_extreme_subjects"),
            1.5,
            BR_COLOR,
        ),
        (
            "BR Outcome",
            float(primary_row(tests, "between_region", "Outcome")["estimate"]),
            sens_value(sens, "between_region", "Outcome", "exclude_predefined_extreme_subjects"),
            0.5,
            BR_COLOR,
        ),
    ]
    for label, main, excluded, y, color in entries:
        ax2.plot([main, excluded], [y, y], color=color, alpha=0.6, linewidth=2.0, zorder=1)
        ax2.scatter(main, y, s=72, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax2.scatter(excluded, y, s=72, facecolor="white", edgecolor=color, linewidth=2.0, zorder=4)
        ax2.text(max(main, excluded) + 0.022, y, f"{main:.3f} → {excluded:.3f}", va="center", fontsize=9.6, color=INK)
    ax2.axvline(1.0, color=INK, linewidth=1.2, linestyle="--", zorder=0)
    ax2.set_xlim(0.88, 1.66)
    ax2.set_ylim(0.0, 3.0)
    ax2.set_yticks([e[3] for e in entries], [e[0] for e in entries])
    ax2.set_xlabel("SZ / HC variance ratio")
    ax2.set_title("Main estimate → exclusion sensitivity", loc="left", fontweight="bold", pad=13)
    clean_axis(ax2, grid_axis="x")
    add_panel_label(ax2, "B")
    ax2.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor="white", markersize=8, label="Main analysis"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=1.8, markersize=8, label="Exclude predefined extremes"),
        ],
        frameon=False,
        loc="upper right",
    )
    ax2.text(
        0.03,
        0.035,
        "Point estimates only; no new CI was calculated.\nNo observation was removed from the primary analysis.",
        transform=ax2.transAxes,
        fontsize=8.9,
        color=MUTED,
        va="bottom",
    )

    fig.suptitle("Sensitivity to predefined extreme subjects", x=0.075, y=0.955, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.075,
        0.898,
        "The Outcome between-subject point estimate changes most when the flagged high-deviation subjects are excluded.",
        ha="left",
        fontsize=10.7,
        color=MUTED,
    )
    save_figure(fig, "Figure_4_Sensitivity")


def write_summary(tests: pd.DataFrame, sens: pd.DataFrame) -> None:
    joint = sens.loc[
        sens["analysis"].eq("between_region")
        & sens["stage"].eq("Joint")
        & sens["sensitivity"].eq("Group_x_Stage_clustered_model")
    ].iloc[0]
    text = f"""# STEP 2 Variance Analysis — Supervisor Meeting Summary

> 本材料仅用于导师汇报；数值来自已完成的 STEP 2 固定分析，没有重新拟合统计模型或增加检验。
> Meeting material only. Values are taken from the frozen STEP 2 analysis; no model was refit and no new test was added.

## 1. 为什么进行 variance analysis？ / Why analyze variability?

**中文：** 均值差异不能回答组内异质性问题。方差分析用于检验 SZ 的 FC profile 或 ROI 空间分布是否比 HC 更分散，从而直接回应“患者组是否更加异质”。
**English:** Mean differences do not address within-group heterogeneity. Variance analysis tests whether SZ shows greater dispersion of FC profiles or ROI-wise spatial values than HC.

## 2. 两类 variability 测量什么？ / What do the two variability measures quantify?

**中文：** Between-subject variability 是每个 reward-stage ROI 类别下，组内被试完整 10-bin FC 向量的总体离散度，即 10 个 bin 被试间样本方差的平均值。Between-region variability 是每名被试在固定 ROI 集合内的 FC 空间离散度：先对每个 bin 计算 ROI 间样本方差，再对 10 个 bin 取平均。这里的 Stage 是 ROI 的奖励加工阶段分类，不是实际扫描任务阶段。
**English:** Between-subject variability is the group dispersion of complete 10-bin FC vectors, equal to the mean across bins of the across-subject sample variance. Between-region variability is each subject's spatial ROI dispersion: ROI sample variance is computed within each bin and then averaged across 10 bins. “Stage” denotes ROI reward-stage classification, not an actual scan phase.

## 3. SZ 的方差是否显著高于 HC？ / Is variability significantly higher in SZ?

**中文：** 没有充分证据。Between-subject 的 SZ/HC 方差比为 Anticipation 1.421（95% CI 0.830–2.477，Holm p=0.612）和 Outcome 1.452（0.878–2.397，Holm p=0.612）。Between-region 的协变量调整后比值为 1.047（0.793–1.383，Holm p=1.000）和 1.054（0.791–1.405，Holm p=1.000）。四项置信区间均跨 1。非显著不等于两组等效。
**English:** There is insufficient evidence. Between-subject SZ/HC ratios were 1.421 for Anticipation (95% CI 0.830–2.477; Holm p=0.612) and 1.452 for Outcome (0.878–2.397; Holm p=0.612). Covariate-adjusted between-region ratios were 1.047 (0.793–1.383; Holm p=1.000) and 1.054 (0.791–1.405; Holm p=1.000). All four confidence intervals include 1. Non-significance does not establish equivalence.

## 4. 两个 Stage 是否不同？ / Do the two ROI categories differ?

**中文：** 两类结果的方向相似。Between-region 的预先规定 Group × Stage 检验得到 ratio-of-ratios={joint['estimate']:.3f}（95% CI {joint['ci_low']:.3f}–{joint['ci_high']:.3f}，p={joint['raw_p']:.3f}），未提示组效应随 Stage 改变。Anticipation 与 Outcome 的 ROI 数分别为 43 和 19，原始 regional dispersion 不能直接按同一空间尺度比较。Between-subject 没有预先规定跨 Stage 交互检验，因此不事后增加检验。
**English:** Effect directions were similar across ROI categories. The prespecified between-region Group × Stage test gave a ratio-of-ratios of {joint['estimate']:.3f} (95% CI {joint['ci_low']:.3f}–{joint['ci_high']:.3f}; p={joint['raw_p']:.3f}), with no evidence that the group effect differed by stage. Anticipation and Outcome contain 43 and 19 ROIs, so their raw regional dispersions are not on an identical spatial scale. No cross-stage between-subject interaction was prespecified, and none was added post hoc.

## 5. 极端被试有什么影响？ / How influential are extreme subjects?

**中文：** Outcome 的 between-subject 比值从 1.452 降到 0.995（排除预定义 extreme-profile 被试后；HC 31、SZ 27），说明该点估计受到少数高偏离被试影响。Anticipation 的相同比值从 1.421 变为 1.498。Between-region 排除预定义 extreme-regional 被试后，Anticipation/Outcome 比值分别为 1.001/1.004，均接近 1。主要分析没有删除任何观测；排除分析只用于评估影响程度。
**English:** For Outcome, the between-subject ratio fell from 1.452 to 0.995 after excluding predefined extreme-profile subjects (HC 31; SZ 27), showing sensitivity to a small number of high-deviation subjects. The corresponding Anticipation ratio changed from 1.421 to 1.498. After excluding predefined extreme-regional subjects, the Anticipation/Outcome between-region ratios were 1.001/1.004. No observation was removed from the primary analyses; exclusions were sensitivity checks only.

## 6. 能告诉老师什么，不能告诉老师什么？ / What can and cannot be concluded?

**中文：** 本结果能量化固定 Definition A、64 名被试和固定 ROI 分类下的两种离散度，并显示目前没有充分统计学证据支持 SZ 与 HC 存在方差差异；Outcome 的 between-subject 点估计对少数被试较敏感。它不能证明两组等效，也不能证明疾病特异性、功能区分能力、神经机制重组或对其他样本和处理版本的可推广性。
**English:** These results quantify two prespecified forms of dispersion in the fixed Definition A sample and show that current evidence is insufficient to support an SZ–HC variance difference; the Outcome between-subject point estimate is sensitive to a few subjects. The analysis does not establish equivalence, disease specificity, functional differentiation, neural reorganization, or generalizability to other samples or processing versions.
"""
    (OUTPUT / "SUPERVISOR_MEETING_SUMMARY.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subject, bs, br, tests, sens, stage = read_and_validate()
    figure_1(subject, tests, stage)
    figure_2(subject, tests)
    figure_3(tests)
    figure_4(subject, tests, sens)
    write_summary(tests, sens)


if __name__ == "__main__":
    main()
