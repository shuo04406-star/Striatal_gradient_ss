from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from _stage_coupling_common import ENDPOINTS, JOINT_RESULTS, RESULTS, load_subject_data, require


def build_long(data: pd.DataFrame, spec: dict[str, str]) -> pd.DataFrame:
    ant = data[["subject", "TEPS_Ant_z", spec["ant"]]].copy()
    ant.columns = ["subject", "TEPS_z", "Neural_z"]
    ant["Stage"] = "Anticipation"
    ant["stage_ant"] = 1.0

    outcome = data[["subject", "TEPS_Cons_z", spec["out"]]].copy()
    outcome.columns = ["subject", "TEPS_z", "Neural_z"]
    outcome["Stage"] = "Outcome"
    outcome["stage_ant"] = 0.0
    return pd.concat([outcome, ant], ignore_index=True)


def clustered_model(long: pd.DataFrame) -> dict[str, float | int | str | bool]:
    design = pd.DataFrame(
        {
            "Intercept": 1.0,
            "Stage_Anticipation": long["stage_ant"],
            "TEPS_z": long["TEPS_z"],
            "Stage_Anticipation_x_TEPS_z": long["stage_ant"] * long["TEPS_z"],
        }
    )
    fit = sm.OLS(long["Neural_z"].to_numpy(dtype=float), design).fit()
    robust = fit.get_robustcov_results(
        cov_type="cluster",
        groups=long["subject"].to_numpy(),
        use_correction=True,
        df_correction=True,
        use_t=True,
    )
    term_index = list(design.columns).index("Stage_Anticipation_x_TEPS_z")
    ci = robust.conf_int(alpha=0.05)[term_index]
    df_inference = getattr(robust, "df_resid_inference", len(long["subject"].unique()) - 1)
    return {
        "interaction_beta": float(robust.params[term_index]),
        "interaction_se": float(robust.bse[term_index]),
        "interaction_ci95_low": float(ci[0]),
        "interaction_ci95_high": float(ci[1]),
        "interaction_p": float(robust.pvalues[term_index]),
        "df_inference": float(df_inference),
        "model_rank": int(np.linalg.matrix_rank(design.to_numpy())),
        "condition_number": float(np.linalg.cond(design.to_numpy())),
    }


def main() -> None:
    data = load_subject_data(eligible_only=True)
    require(RESULTS.is_file(), "Run 03_run_stage_coupling_correlations.py first")
    correlations = pd.read_csv(RESULTS)
    rows: list[dict[str, object]] = []
    long_tables: list[pd.DataFrame] = []

    for endpoint, spec in ENDPOINTS.items():
        long = build_long(data, spec)
        require(len(long) == 2 * len(data), f"Long-format row count failed for {endpoint}")
        require(not long.duplicated(["subject", "Stage"]).any(), f"Duplicate subject-stage rows for {endpoint}")
        estimates = clustered_model(long)
        contrast_rho = float(
            correlations.loc[
                correlations["endpoint"].eq(endpoint)
                & correlations["contrast_scale"].eq("stagewise_z_difference"),
                "rho",
            ].iloc[0]
        )
        beta = float(estimates["interaction_beta"])
        rows.append(
            {
                "endpoint": endpoint,
                "endpoint_label": spec["label"],
                "n_subjects": int(data["subject"].nunique()),
                "n_observations": int(len(long)),
                "model": "Neural_z ~ Stage * TEPS_z; OLS with subject-clustered CR1 covariance",
                "stage_reference": "Outcome",
                "interaction_term": "Stage_Anticipation_x_TEPS_z",
                **estimates,
                "clusters": int(data["subject"].nunique()),
                "convergence": "not applicable; closed-form OLS with cluster-robust covariance",
                "mixed_model_used": False,
                "mixed_model_reason": "Only two observations per participant; prespecified clustered repeated-measures approach avoids unstable random-effect variance estimation.",
                "contrast_spearman_rho": contrast_rho,
                "direction_consistent_with_difference_analysis": bool(
                    np.sign(beta) == np.sign(contrast_rho) or beta == 0 or contrast_rho == 0
                ),
            }
        )
        long.insert(0, "endpoint", endpoint)
        long_tables.append(long)

    pd.DataFrame(rows).to_csv(JOINT_RESULTS, index=False)
    pd.concat(long_tables, ignore_index=True).to_csv(
        JOINT_RESULTS.with_name("joint_stage_model_long_data.csv"), index=False
    )
    for row in rows:
        print(
            f"{row['endpoint']}: interaction beta={row['interaction_beta']:.4f}, "
            f"SE={row['interaction_se']:.4f}, p={row['interaction_p']:.4f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
