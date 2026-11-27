"""Create the four STEP 3A meeting figures from completed statistical outputs."""

from __future__ import annotations

import os
from pathlib import Path
import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image


OUTPUT = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP3_ROI_robustness'))
EFFECT_FILE = OUTPUT / "effect_size_stability.tsv"
PROFILE_FILE = OUTPUT / "profile_stability_metrics.tsv"
SUMMARY_FILE = OUTPUT / "subfamily_deletion_results.tsv"
LOG_FILE = OUTPUT / "STEP3A_EXECUTION_LOG.txt"

INK = "#263238"
MUTED = "#64737C"
GRID = "#DCE3E8"
FULL = "#3F4650"
ANT = "#4C78A8"
OUT = "#E07A5F"
STAGE_COLOR = {"Anticipation": ANT, "Outcome": OUT}
STAGES = ["Anticipation", "Outcome"]


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
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.8,
            "axes.edgecolor": "#8C99A2",
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [EFFECT_FILE, PROFILE_FILE, SUMMARY_FILE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    effect = pd.read_csv(EFFECT_FILE, sep="\t")
    profile = pd.read_csv(PROFILE_FILE, sep="\t")
    summary = pd.read_csv(SUMMARY_FILE, sep="\t")
    if effect.shape[0] != 42:
        raise ValueError(f"Unexpected effect table rows: {effect.shape}")
    if profile.shape[0] != 26:
        raise ValueError(f"Unexpected profile table rows: {profile.shape}")
    if summary.shape[0] != 13 or summary["subfamily"].nunique() != 13:
        raise ValueError(f"Unexpected deletion summary: {summary.shape}")
    if not summary["other_stage_numerically_unchanged"].astype(bool).all():
        raise ValueError("At least one deletion changed the other stage")
    if summary[["threeway_effect", "threeway_delta_boot_ci_low", "threeway_delta_boot_ci_high"]].isna().any().any():
        raise ValueError("Missing primary effect values")
    return effect, profile, summary


def clean_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=320, facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def figure_1(effect: pd.DataFrame, summary: pd.DataFrame) -> None:
    primary = effect.loc[effect["effect"].eq("Group_x_Stage_x_G1")].copy()
    meta = summary.set_index("subfamily")
    primary["subfamily"] = primary["model"].str.replace("Minus ", "", regex=False)
    primary["stage"] = primary["subfamily"].map(meta["stage"])
    primary["deleted_roi_count"] = primary["subfamily"].map(meta["deleted_roi_count"])
    full = primary.loc[primary["model"].eq("Full")]
    deleted = primary.loc[~primary["model"].eq("Full")].sort_values(["stage", "subfamily"])
    plot = pd.concat([full, deleted], ignore_index=True)
    labels = [
        "Full (62 ROIs)"
        if row.model == "Full"
        else f"Minus {row.subfamily}  (−{int(row.deleted_roi_count)} ROI{'s' if int(row.deleted_roi_count) != 1 else ''})"
        for row in plot.itertuples(index=False)
    ]
    colors = [FULL if row.model == "Full" else STAGE_COLOR[row.stage] for row in plot.itertuples(index=False)]
    y = np.arange(len(plot))[::-1]

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    fig.subplots_adjust(left=0.27, right=0.965, top=0.84, bottom=0.15)
    for index, row in enumerate(plot.itertuples(index=False)):
        yi = y[index]
        color = colors[index]
        ax.errorbar(
            row.estimate,
            yi,
            xerr=np.array([[row.estimate - row.CI95_low], [row.CI95_high - row.estimate]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=3.5,
            markersize=7.3 if row.model == "Full" else 6.4,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )
    ax.axvline(0, color=INK, linestyle="--", linewidth=1.05)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Group × Stage × G1 estimate (Fisher-z per centered bin unit)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}"))
    clean_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=FULL, lw=1.8, label="Full"),
            Line2D([0], [0], marker="o", color=ANT, lw=1.8, label="Removed from Anticipation"),
            Line2D([0], [0], marker="o", color=OUT, lw=1.8, label="Removed from Outcome"),
        ],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=3,
    )
    fig.suptitle("Subfamily deletion effect stability", x=0.075, y=0.955, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.075,
        0.900,
        "Points are unchanged-model MixedLM estimates; lines are model-based 95% Wald confidence intervals.",
        ha="left",
        fontsize=10.6,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.055,
        "A change in statistical significance is not used as the stability criterion. Each deletion starts from Full and removes only one subfamily.",
        ha="left",
        fontsize=9.2,
        color=MUTED,
    )
    save(fig, "Figure_1_SubfamilyDeletion_EffectStability")


def stage_palette(stage: str, n: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("Blues" if stage == "Anticipation" else "Oranges")
    return [cmap(value) for value in np.linspace(0.45, 0.88, n)]


def figure_2(profile: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.333, 7.5))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.985, top=0.82, bottom=0.18, hspace=0.34, wspace=0.20)
    bins = np.arange(1, 11)
    for col, stage in enumerate(STAGES):
        data = profile.loc[profile["stage"].eq(stage) & profile["stage_is_affected"].astype(bool)].sort_values("subfamily")
        palette = stage_palette(stage, len(data))
        top = fig.add_subplot(gs[0, col])
        bottom = fig.add_subplot(gs[1, col])
        first = data.iloc[0]
        full_values = np.array([first[f"full_D_bin{b}"] for b in bins], dtype=float)
        top.plot(bins, full_values, color=FULL, linewidth=2.8, marker="o", markersize=4.2, label="Full")
        for color, row in zip(palette, data.itertuples(index=False)):
            deleted = np.array([getattr(row, f"deleted_D_bin{b}") for b in bins], dtype=float)
            delta = np.array([getattr(row, f"delta_from_full_bin{b}") for b in bins], dtype=float)
            top.plot(bins, deleted, color=color, linewidth=1.45, alpha=0.92, marker="o", markersize=2.7, label=f"Minus {row.subfamily}")
            bottom.plot(bins, delta, color=color, linewidth=1.65, alpha=0.95, marker="o", markersize=3.0, label=row.subfamily)
        top.set_title(stage, loc="left", fontweight="bold", pad=9)
        top.set_xticks(bins)
        top.set_ylabel("SZ − HC mean FC (Fisher-z)" if col == 0 else "")
        clean_axis(top, "y")
        bottom.axhline(0, color=FULL, linewidth=1.0, linestyle="--")
        bottom.set_xticks(bins)
        bottom.set_xlabel("G1 bin")
        bottom.set_ylabel("Deleted − Full D(bin)\n(Fisher-z)" if col == 0 else "")
        clean_axis(bottom, "y")
        bottom.legend(frameon=False, ncol=2, loc="best", fontsize=7.6, handlelength=1.6, columnspacing=0.9)
        top.text(
            0.99,
            0.96,
            "Deletions from the other stage are identical to Full and omitted",
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=8.1,
            color=MUTED,
        )

    fig.suptitle("HC/SZ group-difference profile stability", x=0.08, y=0.955, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.08,
        0.900,
        "Top: Full and leave-one-subfamily-out D(bin) profiles. Bottom: the deletion-induced change relative to Full.",
        ha="left",
        fontsize=10.6,
        color=MUTED,
    )
    fig.text(
        0.08,
        0.06,
        "D(bin) = mean FC in SZ − mean FC in HC. Correlation alone is not used to judge stability; RMSE and maximum absolute bin change are reported separately.",
        ha="left",
        fontsize=9.1,
        color=MUTED,
    )
    save(fig, "Figure_2_GroupDifference_ProfileStability")


def figure_3(summary: pd.DataFrame) -> None:
    data = summary.sort_values("absolute_threeway_change", ascending=True).reset_index(drop=True)
    y = np.arange(len(data))
    colors = [STAGE_COLOR[stage] for stage in data["stage"]]
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    fig.subplots_adjust(left=0.22, right=0.95, top=0.84, bottom=0.16)
    ax.axvline(0, color=INK, linestyle="--", linewidth=1.0, zorder=0)
    for yi, row, color in zip(y, data.itertuples(index=False), colors):
        ax.hlines(yi, 0, row.threeway_change_from_full, color=color, linewidth=3.0, alpha=0.45, zorder=1)
        # Plot percentile intervals directly. A bias-shifted bootstrap interval
        # does not have to contain the observed change, while matplotlib xerr
        # requires non-negative distances from the point estimate.
        lo = float(row.threeway_delta_boot_ci_low)
        hi = float(row.threeway_delta_boot_ci_high)
        ax.hlines(yi, lo, hi, color=color, linewidth=1.5, zorder=2)
        ax.vlines([lo, hi], yi - 0.10, yi + 0.10, color=color, linewidth=1.2, zorder=2)
        ax.scatter(
            row.threeway_change_from_full,
            yi,
            s=49,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        xoffset = 8 if row.threeway_change_from_full >= 0 else -8
        align = "left" if row.threeway_change_from_full >= 0 else "right"
        ax.annotate(
            f"{int(row.deleted_roi_count)} ROI{'s' if int(row.deleted_roi_count) != 1 else ''}",
            (row.threeway_change_from_full, yi),
            xytext=(xoffset, 0),
            textcoords="offset points",
            va="center",
            ha=align,
            fontsize=8.0,
            color=MUTED,
        )
    ax.set_yticks(y, data["subfamily"])
    ax.set_xlabel("Change from Full in Group × Stage × G1 estimate")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4f}"))
    clean_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=ANT, lw=1.8, label="Anticipation subfamily"),
            Line2D([0], [0], marker="o", color=OUT, lw=1.8, label="Outcome subfamily"),
        ],
        frameon=False,
        loc="lower right",
        ncol=2,
    )
    fig.suptitle("Subfamily influence ranking", x=0.075, y=0.955, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.075,
        0.900,
        "Subfamilies are ordered by absolute effect change. Error bars are paired subject-bootstrap 95% intervals for Deleted − Full.",
        ha="left",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.055,
        "ROI counts are shown beside each estimate. Count and anatomical specificity are not separated without count-matched random deletion (STEP 3B, not run).",
        ha="left",
        fontsize=9.1,
        color=MUTED,
    )
    save(fig, "Figure_3_SubfamilyInfluence_Ranking")


def figure_4(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.333, 7.5), sharex=False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.16, wspace=0.28)
    all_max = summary["affected_stage_profile_RMSE"].max() * 1.20
    for index, (ax, stage) in enumerate(zip(axes, STAGES)):
        data = summary.loc[summary["stage"].eq(stage)].sort_values("affected_stage_profile_RMSE", ascending=True)
        y = np.arange(len(data))
        color = STAGE_COLOR[stage]
        ax.barh(y, data["affected_stage_profile_RMSE"], color=color, alpha=0.28, height=0.58)
        ax.scatter(data["affected_stage_profile_RMSE"], y, s=54, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        for yi, row in zip(y, data.itertuples(index=False)):
            ax.text(
                row.affected_stage_profile_RMSE + all_max * 0.018,
                yi,
                f"{int(row.deleted_roi_count)} ROI{'s' if int(row.deleted_roi_count) != 1 else ''}",
                va="center",
                fontsize=8.2,
                color=MUTED,
            )
        ax.set_yticks(y, data["subfamily"])
        ax.set_xlim(0, all_max)
        ax.set_xlabel("Profile RMSE (Fisher-z)")
        ax.set_title(stage, loc="left", fontweight="bold", pad=10)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4f}"))
        clean_axis(ax)
        ax.text(-0.14, 1.06, chr(ord("A") + index), transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    fig.suptitle("Profile RMSE ranking after subfamily deletion", x=0.075, y=0.955, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.075,
        0.900,
        "RMSE compares the 10-bin SZ−HC D profile after deletion with the corresponding Full profile in the affected stage.",
        ha="left",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.055,
        "The other stage is unchanged by construction and numerical verification. ROI counts are shown to prevent interpreting size-related influence as region specificity.",
        ha="left",
        fontsize=9.1,
        color=MUTED,
    )
    save(fig, "Figure_4_ProfileRMSE_Ranking")


def append_log() -> None:
    lines = [
        "",
        f"{dt.datetime.now().isoformat()} | INFO | Python plotting QA completed",
        f"{dt.datetime.now().isoformat()} | INFO | matplotlib={mpl.__version__}; pandas={pd.__version__}; numpy={np.__version__}",
    ]
    for path in sorted(OUTPUT.glob("Figure_*.png")):
        with Image.open(path) as image:
            lines.append(f"{dt.datetime.now().isoformat()} | INFO | {path.name}: pixels={image.size}; dpi={image.info.get('dpi')}")
    for path in sorted(OUTPUT.glob("Figure_*.pdf")):
        lines.append(f"{dt.datetime.now().isoformat()} | INFO | {path.name}: bytes={path.stat().st_size}; signature={path.read_bytes()[:5]!r}")
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    configure_style()
    effect, profile, summary = load_and_validate()
    figure_1(effect, summary)
    figure_2(profile)
    figure_3(summary)
    figure_4(summary)
    append_log()


if __name__ == "__main__":
    main()
