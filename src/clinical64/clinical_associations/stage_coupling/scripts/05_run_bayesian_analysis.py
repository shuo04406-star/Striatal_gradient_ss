from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _stage_coupling_common import (
    BAYES_GRID_POINTS,
    BAYES_RESULTS,
    ENDPOINTS,
    bayesian_rank_correlation_grid,
    load_subject_data,
    require,
)


def precision_assessment(row: dict[str, object]) -> str:
    low = float(row["credible_interval_95_low"])
    high = float(row["credible_interval_95_high"])
    p_moderate = float(row["p_abs_r_gt_0_20"])
    if p_moderate < 0.10:
        return "posterior mostly concentrated on small effects"
    if low < -0.20 and high > 0.20:
        return "insufficient precision: moderate effects in either direction remain compatible"
    if low < 0 < high and max(abs(low), abs(high)) > 0.20:
        return "insufficient precision: both near-zero and at least one moderate direction remain compatible"
    return "posterior shows directional concentration but retains material uncertainty"


def main() -> None:
    data = load_subject_data(eligible_only=True)
    x = data["TEPS_stage_contrast"].to_numpy(dtype=float)
    summaries: list[dict[str, object]] = []
    grids: list[pd.DataFrame] = []

    for endpoint, spec in ENDPOINTS.items():
        y = data[spec["y"]].to_numpy(dtype=float)
        require(np.isfinite(x).all() and np.isfinite(y).all(), f"Missing Bayesian input for {endpoint}")
        for prior_role, eta in [("primary_prior", 1.0), ("prior_sensitivity", 2.0)]:
            summary, grid = bayesian_rank_correlation_grid(
                x, y, eta=eta, grid_points=BAYES_GRID_POINTS
            )
            summary.update(
                {
                    "endpoint": endpoint,
                    "endpoint_label": spec["label"],
                    "analysis_role": spec["hierarchy"],
                    "prior_role": prior_role,
                    "likelihood": "Gaussian-copula rank likelihood approximation using average-rank normal scores",
                    "target": "population Spearman-type correlation",
                    "software": f"Python / NumPy {np.__version__} / SciPy deterministic quadrature",
                }
            )
            summary["precision_assessment"] = precision_assessment(summary)
            summaries.append(summary)
            grid.insert(0, "prior_role", prior_role)
            grid.insert(0, "endpoint", endpoint)
            grids.append(grid)

    summary_df = pd.DataFrame(summaries)
    require(len(summary_df) == 4, "Expected primary and sensitivity posterior for both endpoints")
    summary_df.to_csv(BAYES_RESULTS, index=False)
    pd.concat(grids, ignore_index=True).to_csv(
        BAYES_RESULTS.with_name("bayesian_posterior_grid.csv"), index=False, float_format="%.10g"
    )
    for row in summary_df.loc[summary_df["prior_role"].eq("primary_prior")].itertuples():
        print(
            f"{row.endpoint}: posterior median={row.posterior_median_r:.4f}, "
            f"95% CrI [{row.credible_interval_95_low:.4f}, {row.credible_interval_95_high:.4f}]"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
