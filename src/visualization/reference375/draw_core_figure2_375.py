#!/usr/bin/env python3
"""Redraw manuscript Figure 2 from the existing 375-person FC table.

No gradient estimation, FC estimation, or mixed-model refitting is performed.
Output: three separate PNG panels, plotting-source TSVs, LaTeX figure snippet,
standalone LaTeX preview, and a provenance/audit JSON file.
"""
from __future__ import annotations

import os
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import t

BASE = os.environ.get("STRIATAL_HISTORICAL_ROOT")
DEFAULT_INPUT = Path(BASE) / "25_subject_level_stage_g1_model1/tables/subject_stage_G1_long.tsv" if BASE else None
DEFAULT_OUTPUT = Path(BASE) / "30_manuscript_core_figures_v1" if BASE else None
STAGES = ("Anticipation", "Outcome")
BINS = np.arange(1, 11)
N_EXPECTED = 375


def fail(message: str) -> None:
    raise RuntimeError(message)


def numeric(s: pd.Series, name: str) -> pd.Series:
    a = pd.to_numeric(s, errors="raise")
    if not np.isfinite(a.to_numpy(dtype=float)).all():
        fail(f"Missing or nonfinite values in {name}; no silent row deletion is allowed.")
    return a


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    d = pd.read_csv(path, sep="\t", dtype={"Subject": "string"})
    canonical = {"subject": "Subject", "g1_bin": "G1_bin", "stage": "Stage", "fc": "FC"}
    d = d.rename(columns={c: canonical.get(str(c).strip().lower(), c) for c in d.columns})
    required = ["Subject", "G1_bin", "Stage", "FC"]
    if d.columns.duplicated().any() or not set(required).issubset(d.columns):
        fail(f"Unexpected columns: {list(d.columns)}; expected {required}.")
    if d[required].isna().any().any():
        fail("Missing data in the neural table. No participants will be silently excluded.")
    d["Subject"] = d["Subject"].astype("string").str.strip()
    if d["Subject"].eq("").any():
        fail("Empty subject ID.")
    stage_map = {"anticipation": "Anticipation", "outcome": "Outcome"}
    d["Stage"] = d["Stage"].astype(str).str.strip().str.lower().map(stage_map)
    if d["Stage"].isna().any():
        fail("Unknown ROI-set labels; only Anticipation and Outcome are allowed.")
    bins = numeric(d["G1_bin"], "G1_bin")
    if not np.equal(bins, np.floor(bins)).all():
        fail("G1_bin must contain integer bin indices; values will not be rounded.")
    d["G1_bin"] = bins.astype(int)
    d["FC"] = numeric(d["FC"], "FC").astype(float)
    if d["Subject"].nunique() != N_EXPECTED or len(d) != N_EXPECTED * 20:
        fail(f"Expected 375 participants and 7500 rows, found {d['Subject'].nunique()} and {len(d)}.")
    if set(d["G1_bin"]) != set(BINS):
        fail("Bins must be exactly 1 through 10.")
    if d.duplicated(["Subject", "Stage", "G1_bin"]).any():
        fail("Duplicate Subject x ROI-set x bin rows.")
    counts = d.groupby(["Subject", "Stage"]).size()
    if len(counts) != 750 or not counts.eq(10).all():
        fail("Every participant must have both ROI sets and all 10 bins.")
    print(f"Validated: {N_EXPECTED} participants x 2 ROI sets x 10 bins = {len(d)} rows.")
    return d


def mean_ci(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided pointwise t intervals across participants, along axis 0."""
    n = a.shape[0]
    m = a.mean(axis=0)
    half = t.ppf(0.975, n - 1) * a.std(axis=0, ddof=1) / np.sqrt(n)
    return m, m - half, m + half


def make_axes(figsize: tuple[float, float]):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    return fig, ax


def save_panel(fig, path: Path) -> None:
    fig.tight_layout(pad=1.1)
    fig.savefig(path, dpi=400)
    plt.close(fig)
    print("Saved:", path)


def write_latex(out: Path, n_positive: int) -> None:
    pct = 100 * n_positive / N_EXPECTED
    text = r"""% Requires: graphicx, subcaption. Compile from this package directory.
% When integrating into the main paper, copy figures/ and this snippet together.
\begin{figure}[p]
\centering
\captionsetup[subfigure]{font=small,labelformat=simple,justification=raggedright,singlelinecheck=false}
\renewcommand{\thesubfigure}{\Alph{subfigure}}
\begin{subfigure}[t]{0.49\textwidth}
  \centering
  \includegraphics[width=\linewidth]{figures/Figure2A_ROIset_profiles.png}
  \caption{ROI-set connectivity profiles.}
  \label{fig:core_profiles}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\textwidth}
  \centering
  \includegraphics[width=\linewidth]{figures/Figure2B_paired_difference.png}
  \caption{Within-participant connectivity difference.}
  \label{fig:core_difference}
\end{subfigure}
\par\medskip
\begin{subfigure}[t]{0.98\textwidth}
  \centering
  \includegraphics[width=\linewidth]{figures/Figure2C_individual_slope_difference.png}
  \caption{Distribution of individual slope differences.}
  \label{fig:core_individual_slopes}
\end{subfigure}
\caption{\textbf{Resting-state connectivity to anticipation- and outcome-associated ROI sets differs along striatal G1.}
(A) Mean Fisher $z$ connectivity across the 10 fixed group-G1 bins for 43 anticipation-associated and 19 outcome-associated ROIs; shading denotes pointwise 95\% confidence intervals across 375 participants.
(B) Mean within-participant anticipation-associated minus outcome-associated connectivity, with pointwise 95\% confidence intervals calculated from the paired differences. Negative and positive values indicate stronger connectivity to the outcome-associated and anticipation-associated ROI sets, respectively. Lines in A and B connect bin means and are not fitted linear trends.
(C) Distribution of participant-specific linear slope differences, calculated as the anticipation-associated slope minus the outcome-associated slope using bin position $b-5.5$. @@N@@ of 375 participants (@@PCT@@\%) had a positive difference. These are slopes per bin index, not per unit of continuous G1 value. All panels summarize the same participants and are complementary presentations, not independent replications. ROI-set labels reflect meta-analysis-guided associations with reward processing stages; the functional connectivity was measured at rest, not during a reward task.}
\label{fig:core_reward_connectivity}
\end{figure}
"""
    text = text.replace("@@N@@", str(n_positive)).replace("@@PCT@@", f"{pct:.1f}")
    (out / "Figure2_insert.tex").write_text(text, encoding="utf-8")
    preview = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{amsmath}
\pagestyle{empty}
\begin{document}
\setcounter{figure}{1}
\input{Figure2_insert.tex}
\end{document}
"""
    (out / "Figure2_preview.tex").write_text(preview, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, required=DEFAULT_INPUT is None)
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, required=DEFAULT_OUTPUT is None)
    args = p.parse_args()
    # Each run receives its own directory; no existing analysis files are overwritten.
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    out = args.out / stamp
    out.mkdir(parents=True, exist_ok=False)
    figdir, tabdir = out / "figures", out / "tables"
    figdir.mkdir()
    tabdir.mkdir()
    print("INPUT :", args.input)
    print("OUTPUT:", out)

    d = load_and_validate(args.input)
    ids = sorted(d["Subject"].unique().tolist())
    arrays = {}
    for stage in STAGES:
        w = d[d["Stage"] == stage].pivot(index="Subject", columns="G1_bin", values="FC")
        arrays[stage] = w.reindex(index=ids, columns=BINS).to_numpy(dtype=float)
    anti, outcome = arrays["Anticipation"], arrays["Outcome"]
    delta = anti - outcome
    bc = BINS - 5.5
    slope_anti = (anti @ bc) / (bc @ bc)
    slope_out = (outcome @ bc) / (bc @ bc)
    slope_delta = slope_anti - slope_out
    n_positive = int((slope_delta > 0).sum())

    # Source check against the uploaded manuscript's rounded Table 5 values.
    # A mismatch is reported, NEVER corrected by flipping signs or changing bins.
    checks = [
        ("Anticipation_mean_slope", float(slope_anti.mean()), float(os.environ["STRIATAL_EXPECTED_ANTICIPATION_SLOPE"]), 5e-6),
        ("Outcome_mean_slope", float(slope_out.mean()), float(os.environ["STRIATAL_EXPECTED_OUTCOME_SLOPE"]), 5e-6),
        ("Mean_slope_difference", float(slope_delta.mean()), float(os.environ["STRIATAL_EXPECTED_SLOPE_DIFFERENCE"]), 5e-6),
        ("Participants_positive_difference", n_positive, 334, 0),
    ]
    audit_rows = []
    for name, actual, expected, tolerance in checks:
        ok = bool(abs(actual - expected) <= tolerance)
        audit_rows.append(dict(check=name, actual=actual, expected=expected, tolerance=tolerance, passed=ok))
        print(f"{name:35s}: {actual:.9g} | expected {expected:.9g} | {'PASS' if ok else 'MISMATCH'}")
    pd.DataFrame(audit_rows).to_csv(tabdir / "manuscript_value_check.tsv", sep="\t", index=False)
    if not all(row["passed"] for row in audit_rows):
        fail(f"Source values do not match the manuscript. Stop before plotting. Audit: {tabdir}")

    profiles = []
    for stage in STAGES:
        m, lo, hi = mean_ci(arrays[stage])
        for j, b in enumerate(BINS):
            profiles.append(dict(ROI_set=stage, bin=b, N=375, mean=m[j], CI_low=lo[j], CI_high=hi[j]))
    profiles = pd.DataFrame(profiles)
    profiles.to_csv(tabdir / "Figure2A_plot_source.tsv", sep="\t", index=False)
    dm, dl, dh = mean_ci(delta)
    diff_table = pd.DataFrame(dict(bin=BINS, N=375, mean_difference=dm, CI_low=dl, CI_high=dh))
    diff_table.to_csv(tabdir / "Figure2B_plot_source.tsv", sep="\t", index=False)
    slope_table = pd.DataFrame(dict(Subject=ids, anticipation_slope=slope_anti,
                                    outcome_slope=slope_out, slope_difference=slope_delta))
    slope_table.to_csv(tabdir / "Figure2C_plot_source_375.tsv", sep="\t", index=False)
    mean_diff, lo_diff, hi_diff = (float(v) for v in mean_ci(slope_delta))

    # A: actual bin means; no imposed linear shape and no participant exclusion.
    fig, ax = make_axes((4.35, 3.55))
    for stage, label in zip(STAGES, ("Anticipation-associated", "Outcome-associated")):
        q = profiles[profiles["ROI_set"] == stage]
        x = q["bin"].to_numpy()
        ax.plot(x, q["mean"].to_numpy(), marker="o", markersize=3.5, linewidth=1.5, label=label)
        ax.fill_between(x, q["CI_low"].to_numpy(), q["CI_high"].to_numpy(), alpha=0.17)
    ax.set(xlabel="Group-G1 bin", ylabel="Mean connectivity (Fisher z)", xticks=BINS, xlim=(0.7, 10.3))
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    save_panel(fig, figdir / "Figure2A_ROIset_profiles.png")

    # B: CI of the within-person difference, not subtraction of separate CIs.
    fig, ax = make_axes((4.35, 3.55))
    ax.errorbar(BINS, dm, yerr=np.vstack([dm-dl, dh-dm]), marker="o", markersize=3.5,
                linewidth=1.3, capsize=3)
    ax.axhline(0, linestyle="--", linewidth=1, alpha=0.65)
    ax.set(xlabel="Group-G1 bin", ylabel="Anticipation - outcome\nconnectivity (Fisher z)",
           xticks=BINS, xlim=(0.7, 10.3))
    save_panel(fig, figdir / "Figure2B_paired_difference.png")

    # C: distribution of the previously defined participant-specific slope contrast.
    fig, ax = make_axes((8.7, 3.1))
    heights, _, _ = ax.hist(slope_delta, bins=24, alpha=0.8)
    ax.set_ylim(0, max(1.0, float(heights.max())) * 1.42)
    ax.axvline(0, linestyle="--", linewidth=1.1, label="Zero difference")
    ax.axvline(mean_diff, linestyle="-", linewidth=1.4, label="Sample mean")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Participant slope difference (Fisher z per bin)\nAnticipation-associated - outcome-associated")
    ax.set_ylabel("Participants")
    note = (f"{n_positive}/375 ({100*n_positive/375:.1f}%) > 0\n"
            f"Mean = {mean_diff:.6f}\n95% CI [{lo_diff:.6f}, {hi_diff:.6f}]")
    ax.text(0.985, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save_panel(fig, figdir / "Figure2C_individual_slope_difference.png")

    write_latex(out, n_positive)
    provenance = dict(
        input=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        output=str(out.resolve()), participants=375, rows=len(d), bin_position="b - 5.5",
        FC_units="Fisher z, as specified in the source manuscript",
        ROI_sets={"Anticipation": 43, "Outcome": 19},
        processing="Plotting from existing FC; no gradient/FC estimation or mixed-model refit",
        mean_slope_difference=mean_diff, CI_low=lo_diff, CI_high=hi_diff,
        positive_slope_differences=n_positive, checks=audit_rows,
        packages={"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    # This pointer is only for locating the new output; it is not a source-data file.
    (args.out / "LATEST_RUN.txt").write_text(str(out.resolve()) + "\n", encoding="utf-8")
    print("\nDONE: no new behavioral tests or mixed models were run.")
    print("PNG panels:", figdir)
    print("Plot sources:", tabdir)
    print("LaTeX snippet:", out / "Figure2_insert.tex")
    print("Preview source:", out / "Figure2_preview.tex")


if __name__ == "__main__":
    main()
