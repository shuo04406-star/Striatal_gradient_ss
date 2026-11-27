from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _stage_coupling_common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    BOOT_RESULTS,
    ENDPOINTS,
    LOO_RESULTS,
    RESULTS,
    bootstrap_spearman,
    load_subject_data,
    require,
)


def robust_outlier_flags(values: pd.Series) -> tuple[np.ndarray, float]:
    x = values.to_numpy(dtype=float)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    if mad == 0:
        z = np.zeros(len(x))
    else:
        z = 0.67448975 * (x - median) / mad
    return np.abs(z) > 3.5, float(np.max(np.abs(z)))


def main() -> None:
    data = load_subject_data(eligible_only=True).reset_index(drop=True)
    require(RESULTS.is_file(), "Run the correlation script first")
    results = pd.read_csv(RESULTS)
    x = data["TEPS_stage_contrast"].to_numpy(dtype=float)
    bootstrap_rows: list[dict[str, object]] = []
    loo_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    inspection = data[["subject", "TEPS_stage_contrast"]].copy()
    x_flags, x_max_rz = robust_outlier_flags(data["TEPS_stage_contrast"])
    inspection["TEPS_contrast_robust_outlier"] = x_flags

    for endpoint_index, (endpoint, spec) in enumerate(ENDPOINTS.items()):
        y = data[spec["y"]].to_numpy(dtype=float)
        full_rho = float(spearmanr(x, y).statistic)
        boot = bootstrap_spearman(
            x,
            y,
            n_resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED + endpoint_index * 1000,
        )
        boot.update(
            {
                "endpoint": endpoint,
                "endpoint_label": spec["label"],
                "analysis_role": spec["hierarchy"],
                "method": "subject-level percentile bootstrap of Spearman rho",
            }
        )
        bootstrap_rows.append(boot)

        endpoint_loo: list[dict[str, object]] = []
        for i, subject in enumerate(data["subject"]):
            keep = np.arange(len(data)) != i
            rho = float(spearmanr(x[keep], y[keep]).statistic)
            endpoint_loo.append(
                {
                    "endpoint": endpoint,
                    "endpoint_label": spec["label"],
                    "deleted_subject": subject,
                    "n_after_deletion": int(keep.sum()),
                    "rho": rho,
                    "full_sample_rho": full_rho,
                    "delta_rho": rho - full_rho,
                    "absolute_change": abs(rho - full_rho),
                    "same_direction_as_full": bool(np.sign(rho) == np.sign(full_rho)),
                }
            )
        loo_rows.extend(endpoint_loo)
        loo_df = pd.DataFrame(endpoint_loo)
        max_row = loo_df.loc[loo_df["absolute_change"].idxmax()]
        y_flags, y_max_rz = robust_outlier_flags(data[spec["y"]])
        inspection[spec["y"]] = data[spec["y"]]
        inspection[f"{endpoint}_robust_outlier"] = y_flags
        summary_rows.append(
            {
                "endpoint": endpoint,
                "full_sample_rho": full_rho,
                "loo_min_rho": float(loo_df["rho"].min()),
                "loo_max_rho": float(loo_df["rho"].max()),
                "loo_median_rho": float(loo_df["rho"].median()),
                "loo_direction_stable": bool(loo_df["same_direction_as_full"].all()),
                "max_change_deleted_subject": str(max_row["deleted_subject"]),
                "max_absolute_change": float(max_row["absolute_change"]),
                "max_abs_robust_z_behavior": x_max_rz,
                "max_abs_robust_z_neural": y_max_rz,
                "n_robust_outliers_behavior": int(x_flags.sum()),
                "n_robust_outliers_neural": int(y_flags.sum()),
                "ties_behavior_n_values": int(data["TEPS_stage_contrast"].duplicated(keep=False).sum()),
                "ties_neural_n_values": int(data[spec["y"]].duplicated(keep=False).sum()),
            }
        )

        mask = results["endpoint"].eq(endpoint) & results["contrast_scale"].eq(
            "stagewise_z_difference"
        )
        results.loc[mask, "ci95_low"] = boot["ci95_low"]
        results.loc[mask, "ci95_high"] = boot["ci95_high"]
        results.loc[mask, "bootstrap_resamples"] = boot["valid_resamples"]
        results.loc[mask, "bootstrap_sign_consistency"] = boot["sign_consistency"]

    pd.DataFrame(bootstrap_rows).to_csv(BOOT_RESULTS, index=False)
    pd.DataFrame(loo_rows).to_csv(LOO_RESULTS, index=False)
    pd.DataFrame(summary_rows).to_csv(
        LOO_RESULTS.with_name("stability_summary.csv"), index=False
    )
    inspection.to_csv(LOO_RESULTS.with_name("scatter_inspection.csv"), index=False)
    results.to_csv(RESULTS, index=False)
    for row in summary_rows:
        print(
            f"{row['endpoint']}: LOO rho range [{row['loo_min_rho']:.4f}, "
            f"{row['loo_max_rho']:.4f}], max change after deleting "
            f"{row['max_change_deleted_subject']}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
