"""Create the five STEP 3B meeting figures from finalized TSV outputs only."""

from __future__ import annotations

import os
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(os.path.join(os.environ["STRIATAL_SUPPLEMENT_ROOT"], 'STEP3_ROI_robustness/STEP3B_count_matched'))
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
LOG = ROOT / "logs" / "STEP3B_EXECUTION_LOG.txt"

ANT = "#3B73A8"
OUT = "#D56A4C"
TARGET = "#8B2E3C"
RANDOM = "#56636F"
LIGHT = "#D9DEE3"
INK = "#20262D"
GRID = "#E8EAED"
STAGE_COLOR = {"Anticipation": ANT, "Outcome": OUT}
KEY = ["Insula", "MFG", "Amygdala", "IFG", "OFC_vmPFC"]


def configure() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#B8BEC5",
        "axes.linewidth": 0.8,
        "axes.titleweight": "bold",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def save(fig: plt.Figure, stem: str) -> None:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def load() -> dict[str, pd.DataFrame]:
    names = {
        "effect": "roi_count_matched_effect_benchmark.tsv",
        "profile": "roi_count_matched_profile_benchmark.tsv",
        "decomp": "profile_offset_shape_decomposition.tsv",
        "mixed": "mixedlm_random_deletion_results.tsv",
        "manifest": "random_deletion_combinations.tsv",
        "overall": "step3b_overall_summary.tsv",
    }
    frames = {key: pd.read_csv(TABLES / filename, sep="\t") for key, filename in names.items()}
    required = {
        "effect": 13, "profile": 13, "decomp": 26, "overall": 13,
    }
    for key, rows in required.items():
        if len(frames[key]) != rows:
            raise RuntimeError(f"Unexpected {key} row count: {len(frames[key])} != {rows}")
    if not frames["mixed"]["success"].all():
        logging.warning("Figure input contains failed MixedLM fits; valid fits only are plotted")
    return frames


def figure1(frames: dict[str, pd.DataFrame]) -> None:
    effect = frames["effect"].set_index("subfamily")
    merged = frames["manifest"].merge(
        frames["mixed"], on=["combo_id", "stage", "deleted_roi_count"], validate="one_to_one"
    )
    full_beta = float(effect.iloc[0]["beta_full"])
    fig, axes = plt.subplots(2, 3, figsize=(13.333, 7.5), constrained_layout=True)
    for i, name in enumerate(KEY):
        ax = axes.flat[i]
        row = effect.loc[name]
        pool = merged.loc[
            merged["stage"].eq(row["stage"])
            & merged["deleted_roi_count"].eq(int(row["deleted_roi_count"]))
            & merged["success"]
        ].copy()
        # Large sampled strata already omit all target combinations; the exact
        # Amygdala target is explicitly removed from its enumerated k=2 pool.
        if name == "Amygdala":
            pool = pool.loc[~pool["is_any_target_combination"]]
        values = pool["beta"].to_numpy(float) - full_beta
        color = STAGE_COLOR[str(row["stage"])]
        ax.hist(values, bins=min(28, max(8, int(np.sqrt(len(values))))), color=color, alpha=0.60,
                edgecolor="white", linewidth=0.4)
        ax.axvspan(row["random_delta_beta_q2_5"], row["random_delta_beta_q97_5"],
                   color=RANDOM, alpha=0.10, zorder=0)
        ax.axvline(0, color="#7B838B", lw=1, ls="--")
        ax.axvline(row["target_delta_beta"], color=TARGET, lw=2.2)
        ax.set_title(f"{name}  |  {row['stage']}  |  remove {int(row['deleted_roi_count'])}")
        ax.set_xlabel(r"Change in Group × Stage × G1 coefficient, $\Delta\beta$")
        ax.set_ylabel("Random combinations")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.text(0.02, 0.95,
                f"target Δβ = {row['target_delta_beta']:+.4f}\n"
                f"|Δβ| percentile = {row['absolute_delta_beta_percentile']:.1f}\n"
                f"valid n = {int(row['valid_random_fits'])}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#D5DADF", alpha=0.94))
        panel(ax, chr(ord("A") + i))
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.02, 0.93, "Reading the benchmark", fontsize=12, fontweight="bold", va="top")
    axes.flat[-1].text(
        0.02, 0.78,
        "Bars: same-stage, same-count random ROI deletions\n"
        "Red line: observed subfamily deletion\n"
        "Gray band: random 2.5th–97.5th percentiles\n\n"
        "Empirical percentile is a composition-sensitivity\n"
        "benchmark, not a disease-effect p value.",
        fontsize=10, va="top", linespacing=1.45,
    )
    fig.suptitle("Effect-size sensitivity under ROI-count-matched deletion", fontsize=15, fontweight="bold")
    save(fig, "Figure_1_CountMatched_Effect")


def figure2(frames: dict[str, pd.DataFrame]) -> None:
    data = frames["profile"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.333, 7.5), constrained_layout=True, sharex=True)
    global_max = float(max(data["total_rmse_random_q97_5"].max(), data["target_total_rmse"].max())) * 1.12
    for i, stage in enumerate(("Anticipation", "Outcome")):
        ax = axes[i]
        part = data.loc[data["stage"].eq(stage)].sort_values("target_total_rmse")
        y = np.arange(len(part))
        med = part["total_rmse_random_median"].to_numpy(float)
        lo = part["total_rmse_random_q2_5"].to_numpy(float)
        hi = part["total_rmse_random_q97_5"].to_numpy(float)
        ax.hlines(y, lo, hi, color=LIGHT, lw=5, zorder=1)
        ax.scatter(med, y, s=30, color=RANDOM, label="random median", zorder=2)
        ax.scatter(part["target_total_rmse"], y, s=56, marker="D", color=STAGE_COLOR[stage],
                   edgecolor="white", linewidth=0.7, label="target deletion", zorder=3)
        for yy, row in zip(y, part.itertuples(index=False)):
            ax.text(global_max * 0.99, yy, f"nROI={int(row.deleted_roi_count)}  pct={row.total_rmse_target_percentile:.1f}",
                    ha="right", va="center", fontsize=8)
        ax.set_yticks(y, part["subfamily"])
        ax.set_xlim(0, global_max)
        ax.set_xlabel("Total profile RMSE (Fisher-z FC)")
        ax.set_title(stage)
        ax.grid(axis="x", color=GRID, lw=0.7)
        panel(ax, chr(ord("A") + i))
    fig.suptitle("Observed profile change versus same-stage, same-count ROI deletions\n"
                 "Circles: random median; diamonds: target deletion; bars: random 2.5th–97.5th percentiles",
                 fontsize=15, fontweight="bold")
    save(fig, "Figure_2_CountMatched_RMSE")


def figure3(frames: dict[str, pd.DataFrame]) -> None:
    data = frames["decomp"].loc[frames["decomp"]["stage_is_affected"]].copy()
    data["order"] = data["stage"].map({"Anticipation": 0, "Outcome": 1})
    data = data.sort_values(["order", "total_rmse"], ascending=[True, False]).reset_index(drop=True)
    labels = [f"{row.subfamily}  ({'Ant' if row.stage == 'Anticipation' else 'Out'}, n={int(row.deleted_roi_count)})"
              for row in data.itertuples(index=False)]
    y = np.arange(len(data))
    colors = [STAGE_COLOR[value] for value in data["stage"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.333, 7.5), constrained_layout=True, sharey=True)

    ax = axes[0]
    ax.axvline(0, color="#7B838B", lw=1)
    ax.hlines(y, data["offset_bootstrap_ci_low"], data["offset_bootstrap_ci_high"], color=colors, lw=2, alpha=0.55)
    ax.scatter(data["offset_change"], y, c=colors, s=45, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Signed profile offset change (Fisher-z FC)")
    ax.set_title("Mean shift across the 10 bins")
    ax.grid(axis="x", color=GRID, lw=0.7)
    panel(ax, "A")

    ax = axes[1]
    ax.hlines(y, data["shape_bootstrap_ci_low"], data["shape_bootstrap_ci_high"], color=colors, lw=2, alpha=0.55)
    ax.scatter(data["shape_rmse"], y, c=colors, s=45, zorder=3)
    xlimit = float(max(data["shape_bootstrap_ci_high"].max(), data["shape_rmse"].max())) * 1.25
    ax.set_xlim(0, xlimit)
    for yy, row in zip(y, data.itertuples(index=False)):
        ax.text(xlimit * 0.98, yy,
                f"Total={row.total_rmse:.4f} | shape share={100*row.shape_share_squared:.0f}%",
                ha="right", va="center", fontsize=7.6)
    ax.set_xlabel("Centered shape RMSE (Fisher-z FC)")
    ax.set_title("Shape change after removing offset")
    ax.grid(axis="x", color=GRID, lw=0.7)
    panel(ax, "B")
    fig.suptitle("Profile-change decomposition: Total RMSE² = offset² + shape RMSE²\n"
                 "Points: observed change; lines: 95% paired subject-bootstrap intervals (5,000 replicates)",
                 fontsize=15, fontweight="bold")
    save(fig, "Figure_3_Offset_Shape")


def figure4(frames: dict[str, pd.DataFrame]) -> None:
    data = frames["decomp"].loc[
        frames["decomp"]["stage_is_affected"] & frames["decomp"]["subfamily"].isin(["Insula", "IFG"])
    ].set_index("subfamily")
    bins = np.arange(1, 11)
    fig, axes = plt.subplots(2, 2, figsize=(13.333, 7.5), constrained_layout=True)
    for row_index, name in enumerate(("Insula", "IFG")):
        row = data.loc[name]
        color = STAGE_COLOR[str(row["stage"])]
        full = np.array([row[f"full_D_bin{i}"] for i in bins], dtype=float)
        deleted = np.array([row[f"deleted_D_bin{i}"] for i in bins], dtype=float)
        change = np.array([row[f"delta_D_bin{i}"] for i in bins], dtype=float)
        centered = np.array([row[f"centered_change_bin{i}"] for i in bins], dtype=float)

        ax = axes[row_index, 0]
        ax.plot(bins, full, color=INK, marker="o", lw=2, label="Full SZ−HC")
        ax.plot(bins, deleted, color=color, marker="o", lw=2, label=f"Minus {name}")
        ax.axhline(0, color="#8A929A", lw=0.8)
        ax.set_xticks(bins)
        ax.set_ylabel("Group-difference FC (SZ−HC)")
        ax.set_title(f"{name}: full and deleted profiles")
        ax.grid(color=GRID, lw=0.6)
        ax.legend(frameon=False, ncol=2)
        panel(ax, "A" if row_index == 0 else "C")

        ax = axes[row_index, 1]
        ax.plot(bins, change, color=color, marker="o", lw=2, label="Total bin-wise change")
        ax.plot(bins, centered, color=TARGET, marker="o", lw=1.6, label="Centered shape component")
        ax.axhline(float(row["offset_change"]), color=RANDOM, ls="--", lw=1.8, label="Offset component")
        ax.axhline(0, color="#8A929A", lw=0.8)
        ax.set_xticks(bins)
        ax.set_ylabel("Deletion-induced change in SZ−HC FC")
        ax.set_title(f"offset={row['offset_change']:+.4f}; shape RMSE={row['shape_rmse']:.4f}")
        ax.grid(color=GRID, lw=0.6)
        ax.legend(frameon=False, ncol=2)
        panel(ax, "B" if row_index == 0 else "D")
    for ax in axes[-1, :]:
        ax.set_xlabel("G1 bin")
    fig.suptitle("Why MixedLM Δβ and whole-profile RMSE need not agree", fontsize=15, fontweight="bold")
    save(fig, "Figure_4_Insula_IFG_Profiles")


def figure5(frames: dict[str, pd.DataFrame]) -> None:
    data = frames["overall"].copy()
    data["stage_order"] = data["stage"].map({"Anticipation": 0, "Outcome": 1})
    data = data.sort_values(["stage_order", "total_rmse_target_percentile"], ascending=[True, False])
    metrics = [
        ("absolute_delta_beta_percentile", "|Δβ|"),
        ("total_rmse_target_percentile", "Total RMSE"),
        ("absolute_offset_change_target_percentile", "|Offset|"),
        ("shape_rmse_target_percentile", "Shape RMSE"),
        ("maximum_absolute_bin_change_target_percentile", "Max bin"),
    ]
    matrix = data[[column for column, _ in metrics]].to_numpy(float)
    fig = plt.figure(figsize=(13.333, 7.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.58, 1.0])
    table_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(metrics)), [label for _, label in metrics], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(data)), [""] * len(data))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] >= 67 else INK
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=color, fontsize=9)
    boundary = int((data["stage"].eq("Anticipation")).sum()) - 0.5
    ax.axhline(boundary, color="white", lw=5)
    ax.set_title("Count-matched empirical percentile", pad=16)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Target percentile (0–100)")

    table_ax.set_xlim(0, 1)
    table_ax.set_ylim(len(data) - 0.5, -1.25)
    table_ax.axis("off")
    columns = [
        (0.00, "Stage"), (0.13, "Subfamily"), (0.40, "nROI"),
        (0.51, "Δβ"), (0.64, "Total"), (0.76, "Offset"), (0.89, "Shape"),
    ]
    for xpos, label in columns:
        table_ax.text(xpos, -0.82, label, fontweight="bold", fontsize=9, ha="left", va="center")
    for yy, row in enumerate(data.itertuples(index=False)):
        stage_short = "Ant" if row.stage == "Anticipation" else "Out"
        color = STAGE_COLOR[row.stage]
        table_ax.text(0.00, yy, stage_short, color=color, fontweight="bold", va="center", fontsize=8.5)
        table_ax.text(0.13, yy, row.subfamily, va="center", fontsize=8.5)
        table_ax.text(0.42, yy, f"{int(row.deleted_roi_count)}", va="center", fontsize=8.5)
        table_ax.text(0.51, yy, f"{row.target_delta_beta:+.4f}", va="center", fontsize=8.2)
        table_ax.text(0.64, yy, f"{row.total_rmse:.4f}", va="center", fontsize=8.2)
        table_ax.text(0.76, yy, f"{row.offset_change:+.4f}", va="center", fontsize=8.2)
        table_ax.text(0.89, yy, f"{row.shape_rmse:.4f}", va="center", fontsize=8.2)
    table_ax.axhline(boundary, color="#D5DADF", lw=1.3, xmin=0, xmax=1)
    table_ax.set_title("Observed deletion metrics (Fisher-z FC scale)", pad=16)
    fig.suptitle("Overall STEP 3B stability map", fontsize=15, fontweight="bold")
    save(fig, "Figure_5_Overall_Stability")


def qa() -> None:
    rows = []
    for png in sorted(FIGURES.glob("Figure_*.png")):
        with Image.open(png) as image:
            width, height = image.size
            rows.append((png.name, width, height, image.mode, png.stat().st_size))
            if width < 3000 or height < 1600:
                raise RuntimeError(f"Low-resolution export: {png} {width}x{height}")
    pdfs = sorted(FIGURES.glob("Figure_*.pdf"))
    if len(rows) != 5 or len(pdfs) != 5:
        raise RuntimeError(f"Expected five PNG and five PDF figures; found {len(rows)} and {len(pdfs)}")
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write("\nPython figure export QA\n")
        for row in rows:
            handle.write(f"{row[0]}\t{row[1]}x{row[2]}\t{row[3]}\t{row[4]} bytes\n")
        handle.write("Five vector PDF counterparts present.\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    configure()
    frames = load()
    figure1(frames)
    figure2(frames)
    figure3(frames)
    figure4(frames)
    figure5(frames)
    qa()
    logging.info("STEP 3B figures created and export dimensions verified")


if __name__ == "__main__":
    main()
