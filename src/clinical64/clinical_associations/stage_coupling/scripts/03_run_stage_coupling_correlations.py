from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _stage_coupling_common import ENDPOINTS, RESULTS, load_subject_data, require


def main() -> None:
    data = load_subject_data(eligible_only=True)
    x = data["TEPS_stage_contrast"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []

    for endpoint, spec in ENDPOINTS.items():
        for contrast_scale, y_col, analysis_role in [
            ("stagewise_z_difference", spec["y"], spec["hierarchy"]),
            ("raw_stage_difference", spec["raw_y"], "sensitivity"),
        ]:
            y = data[y_col].to_numpy(dtype=float)
            require(np.isfinite(x).all() and np.isfinite(y).all(), f"Missing values in {endpoint}")
            result = spearmanr(x, y)
            ties_x = len(np.unique(x)) < len(x)
            ties_y = len(np.unique(y)) < len(y)
            p_method = (
                "two-sided asymptotic Spearman p; exact p unavailable because ties are present"
                if ties_x or ties_y
                else "two-sided asymptotic Spearman p"
            )
            rows.append(
                {
                    "endpoint": endpoint,
                    "endpoint_label": spec["label"],
                    "analysis_role": analysis_role,
                    "behavior_contrast": "TEPS_Ant_z - TEPS_Cons_z",
                    "neural_contrast": y_col,
                    "contrast_scale": contrast_scale,
                    "n": len(data),
                    "method": "Spearman rank correlation",
                    "rho": float(result.statistic),
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "p_raw": float(result.pvalue),
                    "p_method": p_method,
                    "bootstrap_resamples": np.nan,
                    "bootstrap_sign_consistency": np.nan,
                    "ties_in_behavior_contrast": bool(ties_x),
                    "ties_in_neural_contrast": bool(ties_y),
                    "multiplicity": "none; primary and prespecified secondary interpreted separately",
                    "interpretation_scope": (
                        "prespecified stage-coupling test"
                        if contrast_scale == "stagewise_z_difference"
                        else "raw-difference sensitivity only"
                    ),
                }
            )

    results = pd.DataFrame(rows)
    require(len(results) == 4, "Expected two standardized tests and two raw-difference sensitivities")
    results.to_csv(RESULTS, index=False)
    for row in results.loc[results["contrast_scale"].eq("stagewise_z_difference")].itertuples():
        print(f"{row.endpoint}: N={row.n}, rho={row.rho:.4f}, p={row.p_raw:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
