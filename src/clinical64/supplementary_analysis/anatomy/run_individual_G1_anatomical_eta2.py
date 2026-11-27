#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib

from scipy import stats

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    os.environ["STRIATAL_HISTORICAL_ROOT"]
)

OUT = (
    ROOT /
    "25_current64_individual_G1_statistics" /
    "03_anatomical_eta2"
)

TABLE_OUT = OUT / "01_tables"
FIG_OUT = OUT / "02_figures"
QC_OUT = OUT / "03_qc"

for d in [
    TABLE_OUT,
    FIG_OUT,
    QC_OUT,
]:
    d.mkdir(
        parents=True,
        exist_ok=True
    )


IND_ROOT = (
    ROOT /
    "19_current64_375reference_gradients" /
    "04_subject_gradients"
)

ROW_MAP_FILE = (
    ROOT /
    "19_current64_375reference_gradients" /
    "01_reference" /
    "row_mapping_2847_to_2608.npy"
)

COORD_FILE = (
    ROOT /
    "06_fingerprints" /
    "seed_voxel_coordinates.tsv"
)

ATLAS_ROOT = (
    ROOT /
    "01_inputs" /
    "03_oxford_atlas" /
    "prebuilt_masks_from_old"
)

VST_FILE = (
    ATLAS_ROOT /
    "oxford_vst_mask_ref.nii.gz"
)

CAUDATE_FILE = (
    ATLAS_ROOT /
    "oxford_caudate_mask_ref.nii.gz"
)

PUTAMEN_FILE = (
    ATLAS_ROOT /
    "oxford_putamen_mask_ref.nii.gz"
)

CLIN_FILE = (
    ROOT /
    "21_current64_individual_stats" /
    "01_inputs" /
    "01_clinical" /
    "final64_clinical_for_server_v3.tsv"
)

MOTION_FILE = (
    ROOT /
    "21_current64_individual_stats" /
    "03_derived_metrics" /
    "motion" /
    "current64_motion_posttrim.tsv"
)

GROUP_ROOT = (
    ROOT /
    "08_group_gradient_pass64" /
    "group_lists"
)

HC_FILE = (
    GROUP_ROOT /
    "subjects_HC_32.txt"
)

SZ_FILE = (
    GROUP_ROOT /
    "subjects_SZ_32.txt"
)


# =============================================================================
# SETTINGS
# =============================================================================

N_BOOT = 20000
N_PERM = 50000

RNG_BOOT = np.random.default_rng(20260826)
RNG_PERM = np.random.default_rng(20260827)
RNG_JITTER = np.random.default_rng(12345)


# =============================================================================
# HELPERS
# =============================================================================

def read_list(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return [
            x.strip()
            for x in f
            if x.strip()
            and not x.startswith("#")
        ]


def eta_squared(
    values,
    labels
):
    """
    One-way anatomical eta-squared:

        eta2 = SS_between / SS_total

    labels:
        VST
        Caudate
        Putamen
    """

    y = np.asarray(
        values,
        dtype=float
    )

    labels = np.asarray(
        labels
    )

    grand_mean = np.mean(
        y
    )

    ss_total = np.sum(
        (
            y
            -
            grand_mean
        ) ** 2
    )

    ss_between = 0.0

    for region in [
        "VST",
        "Caudate",
        "Putamen",
    ]:

        x = y[
            labels
            ==
            region
        ]

        ss_between += (
            len(x)
            *
            (
                np.mean(x)
                -
                grand_mean
            ) ** 2
        )

    if ss_total <= 0:
        return np.nan

    return float(
        ss_between
        /
        ss_total
    )


def hedges_g(
    hc,
    sz
):

    hc = np.asarray(
        hc,
        dtype=float
    )

    sz = np.asarray(
        sz,
        dtype=float
    )

    nx = len(hc)
    ny = len(sz)

    vx = np.var(
        hc,
        ddof=1
    )

    vy = np.var(
        sz,
        ddof=1
    )

    pooled_sd = np.sqrt(
        (
            (nx - 1) * vx
            +
            (ny - 1) * vy
        )
        /
        (
            nx + ny - 2
        )
    )

    d = (
        np.mean(hc)
        -
        np.mean(sz)
    ) / pooled_sd

    df = (
        nx + ny - 2
    )

    J = (
        1.0
        -
        3.0
        /
        (
            4.0 * df
            -
            1.0
        )
    )

    return float(
        J * d
    )


def welch_df(
    x,
    y
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    nx = len(x)
    ny = len(y)

    vx = np.var(
        x,
        ddof=1
    )

    vy = np.var(
        y,
        ddof=1
    )

    a = vx / nx
    b = vy / ny

    return float(
        (
            a + b
        ) ** 2
        /
        (
            a ** 2 / (nx - 1)
            +
            b ** 2 / (ny - 1)
        )
    )


def bootstrap_diff(
    hc,
    sz
):

    hc = np.asarray(
        hc,
        dtype=float
    )

    sz = np.asarray(
        sz,
        dtype=float
    )

    vals = np.empty(
        N_BOOT,
        dtype=float
    )

    for i in range(
        N_BOOT
    ):

        x = RNG_BOOT.choice(
            hc,
            size=len(hc),
            replace=True
        )

        y = RNG_BOOT.choice(
            sz,
            size=len(sz),
            replace=True
        )

        vals[i] = (
            np.mean(x)
            -
            np.mean(y)
        )

    lo, hi = np.percentile(
        vals,
        [
            2.5,
            97.5,
        ]
    )

    return (
        float(lo),
        float(hi)
    )


def permutation_p(
    hc,
    sz
):

    hc = np.asarray(
        hc,
        dtype=float
    )

    sz = np.asarray(
        sz,
        dtype=float
    )

    observed = (
        np.mean(hc)
        -
        np.mean(sz)
    )

    pooled = np.concatenate(
        [
            hc,
            sz,
        ]
    )

    n_hc = len(hc)

    extreme = 0

    for _ in range(
        N_PERM
    ):

        p = RNG_PERM.permutation(
            pooled
        )

        diff = (
            np.mean(
                p[:n_hc]
            )
            -
            np.mean(
                p[n_hc:]
            )
        )

        if abs(diff) >= abs(
            observed
        ):
            extreme += 1

    return float(
        (
            extreme + 1
        )
        /
        (
            N_PERM + 1
        )
    )


def summarize(
    x
):

    x = np.asarray(
        x,
        dtype=float
    )

    q1, q3 = np.percentile(
        x,
        [
            25,
            75,
        ]
    )

    return {
        "N":
            len(x),

        "mean":
            float(
                np.mean(x)
            ),

        "SD":
            float(
                np.std(
                    x,
                    ddof=1
                )
            ),

        "median":
            float(
                np.median(x)
            ),

        "Q1":
            float(q1),

        "Q3":
            float(q3),

        "min":
            float(
                np.min(x)
            ),

        "max":
            float(
                np.max(x)
            ),
    }


# =============================================================================
# STEP 1. INPUT CHECK
# =============================================================================

print("=" * 110)
print("STEP 1. INPUT CHECK")
print("=" * 110)

for p in [
    IND_ROOT,
    ROW_MAP_FILE,
    COORD_FILE,
    VST_FILE,
    CAUDATE_FILE,
    PUTAMEN_FILE,
    CLIN_FILE,
    MOTION_FILE,
    HC_FILE,
    SZ_FILE,
]:

    print(
        p,
        "->",
        p.exists()
    )

    if not p.exists():
        raise FileNotFoundError(
            p
        )


# =============================================================================
# STEP 2. FINAL64
# =============================================================================

print()
print("=" * 110)
print("STEP 2. FINAL64")
print("=" * 110)

hc = read_list(
    HC_FILE
)

sz = read_list(
    SZ_FILE
)

if (
    len(hc) != 32
    or
    len(sz) != 32
):
    raise RuntimeError(
        f"HC={len(hc)}, SZ={len(sz)}"
    )

if set(hc) & set(sz):
    raise RuntimeError(
        "HC/SZ overlap."
    )

subjects = (
    hc + sz
)

group_map = {
    s: "HC"
    for s in hc
}

group_map.update({
    s: "SZ"
    for s in sz
})

print(
    "HC =",
    len(hc)
)

print(
    "SZ =",
    len(sz)
)


# =============================================================================
# STEP 3. COMMON2608 ANATOMICAL LABELS
# =============================================================================

print()
print("=" * 110)
print("STEP 3. COMMON2608 ANATOMICAL LABELS")
print("=" * 110)

coord = pd.read_csv(
    COORD_FILE,
    sep="\t"
)

ijk2847 = coord[
    [
        "i",
        "j",
        "k",
    ]
].to_numpy(
    dtype=int
)

if ijk2847.shape != (
    2847,
    3
):
    raise RuntimeError(
        ijk2847.shape
    )

row_map = np.load(
    ROW_MAP_FILE,
    allow_pickle=False
).astype(
    int
).reshape(-1)

if row_map.shape != (
    2608,
):
    raise RuntimeError(
        row_map.shape
    )


def sample_mask(path):

    img = nib.load(
        str(path)
    )

    mask = (
        img.get_fdata()
        >
        0
    )

    return mask[
        ijk2847[:, 0],
        ijk2847[:, 1],
        ijk2847[:, 2]
    ]


vst2847 = sample_mask(
    VST_FILE
)

cau2847 = sample_mask(
    CAUDATE_FILE
)

put2847 = sample_mask(
    PUTAMEN_FILE
)

labels2847 = np.full(
    2847,
    "",
    dtype=object
)

labels2847[
    vst2847
] = "VST"

labels2847[
    cau2847
] = "Caudate"

labels2847[
    put2847
] = "Putamen"

labels = labels2847[
    row_map
]

counts = {
    region:
        int(
            np.sum(
                labels == region
            )
        )
    for region in [
        "VST",
        "Caudate",
        "Putamen",
    ]
}

print(
    counts
)

if counts != {
    "VST": 156,
    "Caudate": 970,
    "Putamen": 1482,
}:

    raise RuntimeError(
        f"Unexpected anatomical counts: {counts}"
    )


# =============================================================================
# STEP 4. INDIVIDUAL ETA2
# =============================================================================

print()
print("=" * 110)
print("STEP 4. INDIVIDUAL G1 ANATOMICAL ETA-SQUARED")
print("=" * 110)

rows = []

for i, subject in enumerate(
    subjects,
    start=1
):

    path = (
        IND_ROOT /
        subject /
        f"{subject}_gradients_procrustes.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    G = np.load(
        path,
        allow_pickle=False
    ).astype(
        np.float64
    )

    if G.shape != (
        2608,
        10
    ):
        raise RuntimeError(
            f"{subject}: {G.shape}"
        )

    if not np.all(
        np.isfinite(G)
    ):
        raise RuntimeError(
            f"{subject}: nonfinite gradient."
        )

    g1 = G[
        :,
        0
    ]

    eta2 = eta_squared(
        g1,
        labels
    )

    vst_mean = float(
        np.mean(
            g1[
                labels == "VST"
            ]
        )
    )

    cau_mean = float(
        np.mean(
            g1[
                labels == "Caudate"
            ]
        )
    )

    put_mean = float(
        np.mean(
            g1[
                labels == "Putamen"
            ]
        )
    )

    rows.append({
        "subject":
            subject,

        "group":
            group_map[
                subject
            ],

        "G1_anatomical_eta2":
            eta2,

        "G1_VST_mean_QC":
            vst_mean,

        "G1_caudate_mean_QC":
            cau_mean,

        "G1_putamen_mean_QC":
            put_mean,
    })

    print(
        f"[{i:02d}/64] "
        f"{subject} {group_map[subject]} | "
        f"eta2={eta2:.6f}"
    )


metric_df = pd.DataFrame(
    rows
)

metric_df.to_csv(
    TABLE_OUT /
    "individual_G1_anatomical_eta2.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 5. MERGE COVARIATES
# =============================================================================

print()
print("=" * 110)
print("STEP 5. BUILD 64-ROW MASTER")
print("=" * 110)

clin = pd.read_csv(
    CLIN_FILE,
    sep="\t"
)

motion = pd.read_csv(
    MOTION_FILE,
    sep="\t"
)

clin = clin[
    [
        "subject",
        "group",
        "age",
        "sex",
    ]
].copy()

motion = motion[
    [
        "subject",
        "group",
        "mean_FD_posttrim",
    ]
].copy()

motion = motion.rename(
    columns={
        "group":
            "group_motion"
    }
)

master = metric_df.merge(
    clin,
    on=[
        "subject",
        "group",
    ],
    how="inner",
    validate="one_to_one"
)

master = master.merge(
    motion,
    on="subject",
    how="inner",
    validate="one_to_one"
)

if not np.all(
    master[
        "group"
    ]
    ==
    master[
        "group_motion"
    ]
):

    raise RuntimeError(
        "Group mismatch."
    )

master = master.drop(
    columns=[
        "group_motion"
    ]
)

master[
    "age"
] = pd.to_numeric(
    master[
        "age"
    ],
    errors="coerce"
)

master[
    "mean_FD_posttrim"
] = pd.to_numeric(
    master[
        "mean_FD_posttrim"
    ],
    errors="coerce"
)

master[
    "age_centered"
] = (
    master[
        "age"
    ]
    -
    master[
        "age"
    ].mean()
)

master[
    "mean_FD_posttrim_centered"
] = (
    master[
        "mean_FD_posttrim"
    ]
    -
    master[
        "mean_FD_posttrim"
    ].mean()
)

required = [
    "subject",
    "group",
    "age",
    "sex",
    "mean_FD_posttrim",
    "G1_anatomical_eta2",
]

if (
    len(master) != 64
    or
    master[
        "subject"
    ].nunique() != 64
):

    raise RuntimeError(
        "Master is not exact final64."
    )

if master[
    required
].isna().sum().sum() != 0:

    raise RuntimeError(
        "Missing values in analysis variables."
    )

MASTER_FILE = (
    TABLE_OUT /
    "current64_G1_anatomical_eta2_covariate_master.tsv"
)

master.to_csv(
    MASTER_FILE,
    sep="\t",
    index=False
)


# =============================================================================
# STEP 6. DESCRIPTIVES
# =============================================================================

print()
print("=" * 110)
print("STEP 6. GROUP DESCRIPTIVES")
print("=" * 110)

summary_rows = []

for group in [
    "HC",
    "SZ",
]:

    x = master.loc[
        master[
            "group"
        ]
        ==
        group,
        "G1_anatomical_eta2"
    ].to_numpy(
        dtype=float
    )

    summary_rows.append({
        "group":
            group,

        **summarize(
            x
        ),
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    TABLE_OUT /
    "G1_anatomical_eta2_group_summary.tsv",
    sep="\t",
    index=False
)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# STEP 7. UNADJUSTED GROUP TEST
# =============================================================================

print()
print("=" * 110)
print("STEP 7. UNADJUSTED HC vs SZ")
print("=" * 110)

hc_eta = master.loc[
    master[
        "group"
    ]
    ==
    "HC",
    "G1_anatomical_eta2"
].to_numpy(
    dtype=float
)

sz_eta = master.loc[
    master[
        "group"
    ]
    ==
    "SZ",
    "G1_anatomical_eta2"
].to_numpy(
    dtype=float
)

welch = stats.ttest_ind(
    hc_eta,
    sz_eta,
    equal_var=False
)

mwu = stats.mannwhitneyu(
    hc_eta,
    sz_eta,
    alternative="two-sided"
)

ci_lo, ci_hi = (
    bootstrap_diff(
        hc_eta,
        sz_eta
    )
)

perm_p = permutation_p(
    hc_eta,
    sz_eta
)

g = hedges_g(
    hc_eta,
    sz_eta
)

unadjusted = pd.DataFrame([
    {
        "outcome":
            "G1_anatomical_eta2",

        "contrast":
            "HC_minus_SZ",

        "HC_N":
            len(hc_eta),

        "SZ_N":
            len(sz_eta),

        "HC_mean":
            float(
                np.mean(hc_eta)
            ),

        "HC_SD":
            float(
                np.std(
                    hc_eta,
                    ddof=1
                )
            ),

        "SZ_mean":
            float(
                np.mean(sz_eta)
            ),

        "SZ_SD":
            float(
                np.std(
                    sz_eta,
                    ddof=1
                )
            ),

        "HC_minus_SZ":
            float(
                np.mean(hc_eta)
                -
                np.mean(sz_eta)
            ),

        "bootstrap95CI_lower":
            ci_lo,

        "bootstrap95CI_upper":
            ci_hi,

        "Welch_t":
            float(
                welch.statistic
            ),

        "Welch_df":
            welch_df(
                hc_eta,
                sz_eta
            ),

        "Welch_p":
            float(
                welch.pvalue
            ),

        "Hedges_g_HC_minus_SZ":
            g,

        "Permutation_p":
            perm_p,

        "Mann_Whitney_U":
            float(
                mwu.statistic
            ),

        "Mann_Whitney_p":
            float(
                mwu.pvalue
            ),
    }
])

unadjusted.to_csv(
    TABLE_OUT /
    "G1_anatomical_eta2_unadjusted_HC_vs_SZ.tsv",
    sep="\t",
    index=False
)

print(
    unadjusted.T.to_string(
        header=False
    )
)


# =============================================================================
# STEP 8. ADJUSTED MODEL
# =============================================================================

print()
print("=" * 110)
print("STEP 8. ADJUSTED PRIMARY MODEL")
print("=" * 110)

formula = (
    'G1_anatomical_eta2 ~ '
    'C(group, Treatment(reference="HC")) '
    '+ age_centered '
    '+ C(sex, Treatment(reference="Female")) '
    '+ mean_FD_posttrim_centered'
)

model = smf.ols(
    formula,
    data=master
).fit()

robust = model.get_robustcov_results(
    cov_type="HC3"
)

term = (
    'C(group, Treatment(reference="HC"))[T.SZ]'
)

idx = list(
    model.params.index
).index(
    term
)

ci_hc3 = robust.conf_int()[
    idx
]

# standardized outcome
master[
    "G1_anatomical_eta2_z"
] = (
    master[
        "G1_anatomical_eta2"
    ]
    -
    master[
        "G1_anatomical_eta2"
    ].mean()
) / master[
    "G1_anatomical_eta2"
].std(
    ddof=1
)

z_formula = (
    'G1_anatomical_eta2_z ~ '
    'C(group, Treatment(reference="HC")) '
    '+ age_centered '
    '+ C(sex, Treatment(reference="Female")) '
    '+ mean_FD_posttrim_centered'
)

z_model = smf.ols(
    z_formula,
    data=master
).fit()

z_robust = (
    z_model
    .get_robustcov_results(
        cov_type="HC3"
    )
)

z_idx = list(
    z_model.params.index
).index(
    term
)


# adjusted marginal means
pred_hc = master.copy()
pred_sz = master.copy()

pred_hc[
    "group"
] = "HC"

pred_sz[
    "group"
] = "SZ"

adjusted_hc = float(
    model.predict(
        pred_hc
    ).mean()
)

adjusted_sz = float(
    model.predict(
        pred_sz
    ).mean()
)


adjusted = pd.DataFrame([
    {
        "outcome":
            "G1_anatomical_eta2",

        "model":
            formula,

        "adjusted_HC_mean":
            adjusted_hc,

        "adjusted_SZ_mean":
            adjusted_sz,

        "adjusted_HC_minus_SZ":
            (
                adjusted_hc
                -
                adjusted_sz
            ),

        "beta_SZ_minus_HC":
            float(
                model.params[
                    term
                ]
            ),

        "SE_OLS":
            float(
                model.bse[
                    term
                ]
            ),

        "p_OLS":
            float(
                model.pvalues[
                    term
                ]
            ),

        "SE_HC3":
            float(
                robust.bse[
                    idx
                ]
            ),

        "t_HC3":
            float(
                robust.tvalues[
                    idx
                ]
            ),

        "p_HC3":
            float(
                robust.pvalues[
                    idx
                ]
            ),

        "CI95_HC3_lower":
            float(
                ci_hc3[0]
            ),

        "CI95_HC3_upper":
            float(
                ci_hc3[1]
            ),

        "standardized_beta_SZ_minus_HC":
            float(
                z_model.params[
                    term
                ]
            ),

        "standardized_p_HC3":
            float(
                z_robust.pvalues[
                    z_idx
                ]
            ),

        "R2":
            float(
                model.rsquared
            ),

        "adjusted_R2":
            float(
                model.rsquared_adj
            ),

        "AIC":
            float(
                model.aic
            ),

        "BIC":
            float(
                model.bic
            ),
    }
])

adjusted.to_csv(
    TABLE_OUT /
    "G1_anatomical_eta2_adjusted_group_effect.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 9. FULL COEFFICIENT TABLE
# =============================================================================

coef_rows = []

names = model.model.exog_names
ci_ols = model.conf_int()
ci_rob = robust.conf_int()

for i, name in enumerate(
    names
):

    coef_rows.append({
        "term":
            name,

        "beta":
            float(
                model.params.iloc[i]
            ),

        "SE_OLS":
            float(
                model.bse.iloc[i]
            ),

        "t_OLS":
            float(
                model.tvalues.iloc[i]
            ),

        "p_OLS":
            float(
                model.pvalues.iloc[i]
            ),

        "CI95_OLS_lower":
            float(
                ci_ols.iloc[i, 0]
            ),

        "CI95_OLS_upper":
            float(
                ci_ols.iloc[i, 1]
            ),

        "SE_HC3":
            float(
                robust.bse[i]
            ),

        "t_HC3":
            float(
                robust.tvalues[i]
            ),

        "p_HC3":
            float(
                robust.pvalues[i]
            ),

        "CI95_HC3_lower":
            float(
                ci_rob[i, 0]
            ),

        "CI95_HC3_upper":
            float(
                ci_rob[i, 1]
            ),
    })


coef_df = pd.DataFrame(
    coef_rows
)

coef_df.to_csv(
    TABLE_OUT /
    "G1_anatomical_eta2_adjusted_coefficients.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 10. VIF
# =============================================================================

X = pd.DataFrame(
    model.model.exog,
    columns=model.model.exog_names
)

vif_rows = []

for i, c in enumerate(
    X.columns
):

    if c == "Intercept":
        continue

    vif_rows.append({
        "term":
            c,

        "VIF":
            float(
                variance_inflation_factor(
                    X.values,
                    i
                )
            ),
    })


vif = pd.DataFrame(
    vif_rows
)

vif.to_csv(
    TABLE_OUT /
    "G1_anatomical_eta2_VIF.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 11. DIAGNOSTICS
# =============================================================================

influence = model.get_influence()

diag = master[
    [
        "subject",
        "group",
        "G1_anatomical_eta2",
    ]
].copy()

diag[
    "fitted"
] = model.fittedvalues

diag[
    "residual"
] = model.resid

diag[
    "studentized_residual"
] = (
    influence
    .resid_studentized_external
)

diag[
    "cooks_distance"
] = (
    influence
    .cooks_distance[0]
)

diag.to_csv(
    TABLE_OUT /
    "G1_anatomical_eta2_regression_diagnostics.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 12. FIGURE 1 — RAW DISTRIBUTION
# =============================================================================

fig, ax = plt.subplots(
    figsize=(6.5, 6.2)
)

data = [
    hc_eta,
    sz_eta,
]

positions = [
    1,
    2,
]

ax.boxplot(
    data,
    positions=positions,
    widths=0.48,
    showfliers=False
)

for xpos, values in zip(
    positions,
    data
):

    jitter = RNG_JITTER.normal(
        0,
        0.055,
        size=len(values)
    )

    ax.scatter(
        xpos + jitter,
        values,
        s=32,
        alpha=0.75,
        zorder=3
    )

    mean = np.mean(
        values
    )

    ci = (
        stats.t.ppf(
            0.975,
            len(values) - 1
        )
        *
        stats.sem(
            values
        )
    )

    ax.errorbar(
        xpos,
        mean,
        yerr=ci,
        marker="D",
        markersize=6,
        capsize=5,
        linewidth=1.5,
        zorder=4
    )


ax.set_xticks(
    positions
)

ax.set_xticklabels(
    [
        "HC (n=32)",
        "SZ (n=32)",
    ]
)

ax.set_ylabel(
    "Individual anatomical η²"
)

ax.set_title(
    "Anatomical organization of individual G1"
)

ax.text(
    0.98,
    0.98,
    (
        f"HC − SZ = "
        f"{np.mean(hc_eta)-np.mean(sz_eta):.4f}\n"
        f"Welch p = "
        f"{welch.pvalue:.4g}\n"
        f"Hedges' g = "
        f"{g:.3f}"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round",
        alpha=0.08
    )
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

fig.tight_layout()

fig.savefig(
    FIG_OUT /
    "Figure_individual_G1_anatomical_eta2_HC_vs_SZ.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG_OUT /
    "Figure_individual_G1_anatomical_eta2_HC_vs_SZ.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# STEP 13. FIGURE 2 — ADJUSTED EFFECT
# =============================================================================

beta = float(
    model.params[
        term
    ]
)

ci_lo = float(
    ci_hc3[0]
)

ci_hi = float(
    ci_hc3[1]
)

p_hc3 = float(
    robust.pvalues[
        idx
    ]
)


fig, ax = plt.subplots(
    figsize=(6.5, 4.5)
)

ax.errorbar(
    beta,
    0,
    xerr=np.array([
        [
            beta - ci_lo
        ],
        [
            ci_hi - beta
        ],
    ]),
    marker="o",
    markersize=8,
    capsize=6,
    linewidth=1.8
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_yticks([
    0
])

ax.set_yticklabels([
    "SZ − HC"
])

ax.set_xlabel(
    "Adjusted difference in individual anatomical η²"
)

ax.set_title(
    "Adjusted diagnostic-group effect on G1 anatomical organization"
)

ax.text(
    0.98,
    0.92,
    (
        f"β = {beta:.4f}\n"
        f"95% CI [{ci_lo:.4f}, "
        f"{ci_hi:.4f}]\n"
        f"HC3 p = {p_hc3:.4g}"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top"
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

ax.spines[
    "left"
].set_visible(
    False
)

fig.tight_layout()

fig.savefig(
    FIG_OUT /
    "Figure_adjusted_group_effect_G1_anatomical_eta2.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG_OUT /
    "Figure_adjusted_group_effect_G1_anatomical_eta2.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# STEP 14. FINAL QC
# =============================================================================

all_finite = bool(
    np.all(
        np.isfinite(
            master[
                [
                    "G1_anatomical_eta2",
                    "age",
                    "mean_FD_posttrim",
                ]
            ].to_numpy(
                dtype=float
            )
        )
    )
)

eta_bounds = bool(
    np.all(
        (
            master[
                "G1_anatomical_eta2"
            ]
            >=
            0
        )
        &
        (
            master[
                "G1_anatomical_eta2"
            ]
            <=
            1
        )
    )
)

max_vif = float(
    vif[
        "VIF"
    ].max()
)

final_pass = bool(
    len(master) == 64
    and
    master[
        "subject"
    ].nunique() == 64
    and
    all_finite
    and
    eta_bounds
)


qc = {
    "N":
        64,

    "HC":
        int(
            np.sum(
                master[
                    "group"
                ]
                ==
                "HC"
            )
        ),

    "SZ":
        int(
            np.sum(
                master[
                    "group"
                ]
                ==
                "SZ"
            )
        ),

    "voxels":
        2608,

    "VST_voxels":
        counts[
            "VST"
        ],

    "Caudate_voxels":
        counts[
            "Caudate"
        ],

    "Putamen_voxels":
        counts[
            "Putamen"
        ],

    "eta2_definition":
        "SS_between(VST/Caudate/Putamen) / SS_total",

    "G1_sign_invariant":
        True,

    "covariates":
        [
            "age",
            "sex",
            "mean_FD_posttrim",
        ],

    "primary_adjusted_SE":
        "HC3 heteroskedasticity-robust",

    "max_VIF":
        max_vif,

    "all_eta2_within_0_1":
        eta_bounds,

    "final_check":
        (
            "PASS"
            if final_pass
            else
            "FAIL"
        ),
}

with open(
    QC_OUT /
    "FINAL_G1_ANATOMICAL_ETA2_QC.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        qc,
        f,
        indent=2,
        ensure_ascii=False
    )


# =============================================================================
# FINAL RESULT
# =============================================================================

print()
print("=" * 110)
print("FINAL RESULT")
print("=" * 110)

print()
print("INDIVIDUAL ANATOMICAL ETA2")
print("-" * 100)

print(
    f"HC = "
    f"{np.mean(hc_eta):.6f} ± "
    f"{np.std(hc_eta, ddof=1):.6f}"
)

print(
    f"SZ = "
    f"{np.mean(sz_eta):.6f} ± "
    f"{np.std(sz_eta, ddof=1):.6f}"
)

print(
    f"HC - SZ = "
    f"{np.mean(hc_eta)-np.mean(sz_eta):.6f}"
)

print(
    f"Bootstrap 95% CI = "
    f"[{ci_lo if False else unadjusted.iloc[0]['bootstrap95CI_lower']:.6f}, "
    f"{unadjusted.iloc[0]['bootstrap95CI_upper']:.6f}]"
)

print(
    f"Welch t({unadjusted.iloc[0]['Welch_df']:.3f}) = "
    f"{welch.statistic:.6f}"
)

print(
    f"Welch p = "
    f"{welch.pvalue:.8f}"
)

print(
    f"Hedges' g = "
    f"{g:.6f}"
)

print(
    f"Permutation p = "
    f"{perm_p:.8f}"
)

print(
    f"Mann-Whitney p = "
    f"{mwu.pvalue:.8f}"
)


print()
print("ADJUSTED")
print("-" * 100)

print(
    f"Adjusted HC mean = "
    f"{adjusted_hc:.6f}"
)

print(
    f"Adjusted SZ mean = "
    f"{adjusted_sz:.6f}"
)

print(
    f"SZ - HC beta = "
    f"{beta:.6f}"
)

print(
    f"HC3 95% CI = "
    f"[{ci_lo:.6f}, "
    f"{ci_hi:.6f}]"
)

print(
    f"HC3 p = "
    f"{p_hc3:.8f}"
)

print(
    f"Standardized beta = "
    f"{z_model.params[term]:.6f}"
)


print()
print("FULL ADJUSTED COEFFICIENT TABLE")
print("-" * 100)

print(
    coef_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print()
print("VIF")
print("-" * 100)

print(
    vif.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print()
print(
    "FINAL_CHECK =",
    qc[
        "final_check"
    ]
)

print()
print("TABLES")
print("-" * 100)

for p in [
    TABLE_OUT /
    "individual_G1_anatomical_eta2.tsv",

    TABLE_OUT /
    "current64_G1_anatomical_eta2_covariate_master.tsv",

    TABLE_OUT /
    "G1_anatomical_eta2_group_summary.tsv",

    TABLE_OUT /
    "G1_anatomical_eta2_unadjusted_HC_vs_SZ.tsv",

    TABLE_OUT /
    "G1_anatomical_eta2_adjusted_group_effect.tsv",

    TABLE_OUT /
    "G1_anatomical_eta2_adjusted_coefficients.tsv",

    TABLE_OUT /
    "G1_anatomical_eta2_VIF.tsv",

    TABLE_OUT /
    "G1_anatomical_eta2_regression_diagnostics.tsv",
]:

    print(p)

print()
print("FIGURES")
print("-" * 100)

print(
    FIG_OUT /
    "Figure_individual_G1_anatomical_eta2_HC_vs_SZ.png"
)

print(
    FIG_OUT /
    "Figure_adjusted_group_effect_G1_anatomical_eta2.png"
)
