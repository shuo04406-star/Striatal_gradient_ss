from __future__ import annotations

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _stage_coupling_common import (
    BAYES_RESULTS,
    ENDPOINTS,
    LOO_RESULTS,
    OUT_DIR,
    RESULTS,
    format_p,
    load_subject_data,
    require,
)


COLORS = {
    "overall_fc_slope": "#2F6F9F",
    "bin10_similarity": "#8A5A8F",
    "neutral": "#5D6770",
    "light": "#D9E0E5",
    "ink": "#20262C",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["neutral"],
            "ytick.color": COLORS["neutral"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["ink"],
    )


def scatter_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    results: pd.Series,
    y_col: str,
    title: str,
    color: str,
    y_label: str,
) -> None:
    x = data["TEPS_stage_contrast"].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    ax.axhline(0, color=COLORS["light"], lw=0.7, zorder=0)
    ax.axvline(0, color=COLORS["light"], lw=0.7, zorder=0)
    ax.scatter(
        x,
        y,
        s=25,
        facecolor=color,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.88,
        zorder=2,
    )
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=5)
    ax.set_xlabel("TEPS stage contrast\n(Anticipatory z - Consummatory z)")
    ax.set_ylabel(y_label)
    annotation = (
        f"n = {int(results['n'])}\n"
        f"Spearman rho = {results['rho']:.2f}\n"
        f"95% bootstrap CI [{results['ci95_low']:.2f}, {results['ci95_high']:.2f}]\n"
        f"raw p = {format_p(float(results['p_raw']))}"
    )
    ax.text(
        0.08,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        linespacing=1.28,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=COLORS["light"], linewidth=0.6, alpha=0.94),
    )
    ax.tick_params(length=2.5, width=0.6)


def loo_panel(ax: plt.Axes, loo: pd.DataFrame) -> None:
    rng = np.random.default_rng(20261029)
    labels = ["Overall FC\nslope", "Bin 10\nsimilarity"]
    for x_pos, endpoint in enumerate(ENDPOINTS, start=1):
        d = loo.loc[loo["endpoint"].eq(endpoint)]
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(
            x_pos + jitter,
            d["rho"],
            s=15,
            color=COLORS[endpoint],
            alpha=0.58,
            edgecolor="none",
        )
        full = float(d["full_sample_rho"].iloc[0])
        ax.scatter(
            x_pos,
            full,
            s=48,
            marker="D",
            facecolor="white",
            edgecolor=COLORS[endpoint],
            linewidth=1.2,
            zorder=4,
        )
        ax.plot([x_pos - 0.18, x_pos + 0.18], [d["rho"].median()] * 2, color=COLORS[endpoint], lw=1.2)
    ax.axhline(0, color=COLORS["neutral"], lw=0.7, ls="--", alpha=0.7)
    ax.set_xlim(0.55, 2.45)
    ax.set_xticks([1, 2], labels)
    ax.set_ylabel("Leave-one-out Spearman rho")
    ax.set_title("Deletion stability", loc="left", fontsize=8, fontweight="bold", pad=5)
    ax.text(
        0.03,
        0.08,
        "Small points: one participant deleted\nOpen diamonds: full-sample estimates",
        transform=ax.transAxes,
        fontsize=6.2,
        va="bottom",
        color=COLORS["neutral"],
    )
    ax.tick_params(length=2.5, width=0.6)


def bayes_panel(ax: plt.Axes, bayes: pd.DataFrame) -> None:
    primary = bayes.loc[bayes["prior_role"].eq("primary_prior")].copy()
    order = ["overall_fc_slope", "bin10_similarity"]
    y_positions = {"overall_fc_slope": 2, "bin10_similarity": 1}
    ax.axvspan(-0.10, 0.10, color="#EEF1F3", zorder=0)
    ax.axvline(0, color=COLORS["neutral"], lw=0.7, ls="--", alpha=0.75)
    for endpoint in order:
        row = primary.loc[primary["endpoint"].eq(endpoint)].iloc[0]
        y = y_positions[endpoint]
        ax.plot(
            [row["credible_interval_95_low"], row["credible_interval_95_high"]],
            [y, y],
            color=COLORS[endpoint],
            lw=2.0,
            solid_capstyle="round",
        )
        ax.scatter(
            row["posterior_median_r"],
            y,
            s=38,
            color=COLORS[endpoint],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            0.98,
            y,
            f"P(|r| > .20) = {row['p_abs_r_gt_0_20']:.2f}",
            ha="right",
            va="center",
            fontsize=6.2,
            color=COLORS["neutral"],
        )
    ax.set_yticks([1, 2], ["Bin 10 similarity", "Overall FC slope"])
    ax.set_ylim(0.5, 2.5)
    ax.set_xlim(-0.65, 1.02)
    ax.set_xlabel("Posterior Spearman-type correlation (median and 95% CrI)")
    ax.set_title("Bayesian posterior summary", loc="left", fontsize=8, fontweight="bold", pad=5)
    ax.text(
        0.03,
        0.08,
        "Shaded band: |r| <= .10\nPrimary prior: LKJ(1) on latent copula correlation",
        transform=ax.transAxes,
        fontsize=6.1,
        va="bottom",
        color=COLORS["neutral"],
    )
    ax.tick_params(length=2.5, width=0.6)


def main() -> None:
    configure_matplotlib()
    data = load_subject_data(eligible_only=True)
    require(RESULTS.is_file(), "Missing correlation results")
    require(LOO_RESULTS.is_file(), "Missing leave-one-out results")
    require(BAYES_RESULTS.is_file(), "Missing Bayesian results")
    results = pd.read_csv(RESULTS)
    results = results.loc[results["contrast_scale"].eq("stagewise_z_difference")]
    loo = pd.read_csv(LOO_RESULTS)
    bayes = pd.read_csv(BAYES_RESULTS)

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 145 / 25.4))
    ax_a, ax_b, ax_c, ax_d = axes.flat
    row_a = results.loc[results["endpoint"].eq("overall_fc_slope")].iloc[0]
    row_b = results.loc[results["endpoint"].eq("bin10_similarity")].iloc[0]
    scatter_panel(
        ax_a,
        data,
        row_a,
        "NeuralSlope_stage_contrast",
        "Behavior-neural coupling: overall FC slope",
        COLORS["overall_fc_slope"],
        "Neural slope stage contrast\n(Anticipation z - Outcome z)",
    )
    scatter_panel(
        ax_b,
        data,
        row_b,
        "Bin10Similarity_stage_contrast",
        "Behavior-neural coupling: Bin 10 similarity",
        COLORS["bin10_similarity"],
        "Bin 10 similarity stage contrast\n(Anticipation z - Outcome z)",
    )
    loo_panel(ax_c, loo)
    bayes_panel(ax_d, bayes)
    for ax, label in zip(axes.flat, ["a", "b", "c", "d"]):
        panel_label(ax, label)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.945, bottom=0.105, wspace=0.36, hspace=0.43)

    figure_dir = OUT_DIR / "figures"
    base = figure_dir / "Figure1_stage_coupling"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".tiff"), dpi=600)
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)

    source = data[
        [
            "subject",
            "TEPS_stage_contrast",
            "NeuralSlope_stage_contrast",
            "Bin10Similarity_stage_contrast",
        ]
    ].copy()
    source.to_csv(figure_dir / "Figure1_stage_coupling_source_data.csv", index=False)
    qa = """# Figure QA\n\n- Core conclusion: the figure assesses whether behavioral reward-stage differentiation covaries with matched neural-stage differentiation.\n- Archetype: quantitative 2 x 2 grid; panels a-b are the primary/secondary evidence, c-d are stability/uncertainty.\n- Backend: Python/matplotlib only.\n- Final size: 183 x 145 mm.\n- Exports: editable SVG, PDF with TrueType text, 600-dpi TIFF, 300-dpi PNG preview.\n- Statistics: n denotes SZ participants with both TEPS subscales and both neural-stage values; correlations are Spearman; intervals in a-b are 10,000-resample subject-level percentile bootstrap CIs.\n- No significance stars are used.\n- Image integrity: no raster source images, crops, contrast changes, or pseudo-color transformations.\n- Source data: Figure1_stage_coupling_source_data.csv.\n"""
    (figure_dir / "FIGURE_QA.md").write_text(qa, encoding="utf-8")
    print(f"Figure written to {base.with_suffix('.svg')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
