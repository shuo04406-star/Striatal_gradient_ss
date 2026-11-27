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
from statsmodels.stats.multitest import multipletests
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
    "08_fixed375_normative_bins"
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


# Frozen normative reference used for individual alignment
REF_FILE = (
    ROOT /
    "19_current64_375reference_gradients" /
    "01_reference" /
    "375_group_gradients_mapped_2608x10.npy"
)

# Individual Procrustes-aligned gradients
IND_ROOT = (
    ROOT /
    "19_current64_375reference_gradients" /
    "04_subject_gradients"
)

# 2847 -> common2608 mapping
ROW_MAP_FILE = (
    ROOT /
    "19_current64_375reference_gradients" /
    "01_reference" /
    "row_mapping_2847_to_2608.npy"
)

# Original voxel coordinates
COORD_FILE = (
    ROOT /
    "06_fingerprints" /
    "seed_voxel_coordinates.tsv"
)

# Atlas
ATLAS_ROOT = (
    ROOT /
    "01_inputs" /
    "03_oxford_atlas" /
    "prebuilt_masks_from_old"
)

WHOLE_MASK_FILE = (
    ATLAS_ROOT /
    "oxford_whole_striatum_mask_ref.nii.gz"
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

# Groups
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

# Covariates
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


# =============================================================================
# SETTINGS
# =============================================================================

N_BINS = 10

RNG_JITTER = np.random.default_rng(
    12345
)


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

    pooled = np.sqrt(
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

    if pooled <= 0:
        return np.nan

    d = (
        np.mean(hc)
        -
        np.mean(sz)
    ) / pooled

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


def safe_sample_mask(
    path,
    ijk
):

    img = nib.load(
        str(path)
    )

    dat = (
        img.get_fdata()
        >
        0
    )

    return dat[
        ijk[:, 0],
        ijk[:, 1],
        ijk[:, 2]
    ]


def fit_mixedlm(
    formula,
    data
):

    model = smf.mixedlm(
        formula,
        data=data,
        groups=data[
            "subject"
        ],
        re_formula="1"
    )

    methods = [
        "lbfgs",
        "bfgs",
        "powell",
    ]

    last_error = None

    for method in methods:

        try:

            result = model.fit(
                reml=False,
                method=method,
                maxiter=3000,
                disp=False
            )

            print(
                f"MixedLM method={method} | "
                f"converged={result.converged}"
            )

            if result.converged:
                return result

        except Exception as e:

            last_error = e

            print(
                f"MixedLM {method} failed:",
                repr(e)
            )

    if last_error is not None:
        raise RuntimeError(
            f"MixedLM failed: {last_error}"
        )

    raise RuntimeError(
        "MixedLM did not converge."
    )


# =============================================================================
# STEP 1. INPUT CHECK
# =============================================================================

print("=" * 110)
print("STEP 1. INPUT CHECK")
print("=" * 110)

required = [
    REF_FILE,
    IND_ROOT,
    ROW_MAP_FILE,
    COORD_FILE,
    WHOLE_MASK_FILE,
    VST_FILE,
    CAUDATE_FILE,
    PUTAMEN_FILE,
    HC_FILE,
    SZ_FILE,
    CLIN_FILE,
    MOTION_FILE,
]

for p in required:

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
# STEP 3. COMMON2608 COORDINATES / LABELS
# =============================================================================

print()
print("=" * 110)
print("STEP 3. COMMON2608 SPACE")
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

ijk2608 = ijk2847[
    row_map
]

whole_img = nib.load(
    str(
        WHOLE_MASK_FILE
    )
)

xyz2608 = nib.affines.apply_affine(
    whole_img.affine,
    ijk2608
)


vst2847 = safe_sample_mask(
    VST_FILE,
    ijk2847
)

cau2847 = safe_sample_mask(
    CAUDATE_FILE,
    ijk2847
)

put2847 = safe_sample_mask(
    PUTAMEN_FILE,
    ijk2847
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

labels2608 = labels2847[
    row_map
]

counts = {
    region:
        int(
            np.sum(
                labels2608 == region
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
        counts
    )


# =============================================================================
# STEP 4. FROZEN 375 G1 + GLOBAL PAPER ORIENTATION
# =============================================================================

print()
print("=" * 110)
print("STEP 4. FROZEN 375 NORMATIVE G1")
print("=" * 110)

ref = np.load(
    REF_FILE,
    allow_pickle=False
).astype(
    np.float64
)

if ref.shape != (
    2608,
    10
):
    raise RuntimeError(
        ref.shape
    )

ref_g1_raw = ref[
    :,
    0
]

ref_vst = float(
    np.mean(
        ref_g1_raw[
            labels2608
            ==
            "VST"
        ]
    )
)

ref_cau = float(
    np.mean(
        ref_g1_raw[
            labels2608
            ==
            "Caudate"
        ]
    )
)

ref_put = float(
    np.mean(
        ref_g1_raw[
            labels2608
            ==
            "Putamen"
        ]
    )
)

ref_low = (
    ref_vst
    +
    ref_cau
) / 2.0

GLOBAL_SIGN = (
    1.0
    if ref_put > ref_low
    else
    -1.0
)

ref_g1 = (
    GLOBAL_SIGN
    *
    ref_g1_raw
)

print(
    "GLOBAL_SIGN =",
    GLOBAL_SIGN
)

print(
    "375 VST mean =",
    np.mean(
        ref_g1[
            labels2608
            ==
            "VST"
        ]
    )
)

print(
    "375 Caudate mean =",
    np.mean(
        ref_g1[
            labels2608
            ==
            "Caudate"
        ]
    )
)

print(
    "375 Putamen mean =",
    np.mean(
        ref_g1[
            labels2608
            ==
            "Putamen"
        ]
    )
)


# =============================================================================
# STEP 5. DEFINE FIXED 375 NORMATIVE BINS
# =============================================================================

print()
print("=" * 110)
print("STEP 5. DEFINE FIXED 375 NORMATIVE 10 BINS")
print("=" * 110)

# Stable ascending order:
# Bin1 = lowest normative G1
# Bin10 = highest normative G1
order = np.argsort(
    ref_g1,
    kind="mergesort"
)

bin_labels = np.zeros(
    2608,
    dtype=int
)

split_indices = np.array_split(
    order,
    N_BINS
)

for b, idx in enumerate(
    split_indices,
    start=1
):
    bin_labels[
        idx
    ] = b


if np.any(
    bin_labels == 0
):
    raise RuntimeError(
        "Unassigned voxels."
    )


# voxel-level definition
bin_voxel_df = pd.DataFrame({
    "row2608":
        np.arange(
            2608
        ),

    "row2847":
        row_map,

    "i":
        ijk2608[:, 0],

    "j":
        ijk2608[:, 1],

    "k":
        ijk2608[:, 2],

    "MNI_x":
        xyz2608[:, 0],

    "MNI_y":
        xyz2608[:, 1],

    "MNI_z":
        xyz2608[:, 2],

    "anatomical_label":
        labels2608,

    "frozen375_G1":
        ref_g1,

    "normative_bin":
        bin_labels,
})

bin_voxel_df.to_csv(
    TABLE_OUT /
    "fixed375_normative_bin_definition_2608voxels.tsv",
    sep="\t",
    index=False
)


# Bin summary/composition
composition_rows = []

for b in range(
    1,
    11
):

    idx = (
        bin_labels
        ==
        b
    )

    n = int(
        np.sum(
            idx
        )
    )

    vst_n = int(
        np.sum(
            labels2608[
                idx
            ]
            ==
            "VST"
        )
    )

    cau_n = int(
        np.sum(
            labels2608[
                idx
            ]
            ==
            "Caudate"
        )
    )

    put_n = int(
        np.sum(
            labels2608[
                idx
            ]
            ==
            "Putamen"
        )
    )

    composition_rows.append({
        "bin":
            b,

        "n_voxels":
            n,

        "reference_G1_mean":
            float(
                np.mean(
                    ref_g1[
                        idx
                    ]
                )
            ),

        "reference_G1_min":
            float(
                np.min(
                    ref_g1[
                        idx
                    ]
                )
            ),

        "reference_G1_max":
            float(
                np.max(
                    ref_g1[
                        idx
                    ]
                )
            ),

        "VST_n":
            vst_n,

        "Caudate_n":
            cau_n,

        "Putamen_n":
            put_n,

        "VST_percent":
            100.0
            *
            vst_n
            /
            n,

        "Caudate_percent":
            100.0
            *
            cau_n
            /
            n,

        "Putamen_percent":
            100.0
            *
            put_n
            /
            n,
    })


composition = pd.DataFrame(
    composition_rows
)

composition.to_csv(
    TABLE_OUT /
    "fixed375_normative_bin_anatomical_composition.tsv",
    sep="\t",
    index=False
)


print(
    composition.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =============================================================================
# STEP 6. INDIVIDUAL G1 WITHIN FIXED BINS
# =============================================================================

print()
print("=" * 110)
print("STEP 6. INDIVIDUAL G1 IN FIXED 375 BINS")
print("=" * 110)

long_rows = []

for si, subject in enumerate(
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

    g1 = (
        GLOBAL_SIGN
        *
        G[
            :,
            0
        ]
    )

    # shape-only QC version
    g1_z = (
        g1
        -
        np.mean(
            g1
        )
    ) / np.std(
        g1,
        ddof=0
    )

    vals_print = []

    for b in range(
        1,
        11
    ):

        idx = (
            bin_labels
            ==
            b
        )

        mean_raw = float(
            np.mean(
                g1[
                    idx
                ]
            )
        )

        mean_z = float(
            np.mean(
                g1_z[
                    idx
                ]
            )
        )

        long_rows.append({
            "subject":
                subject,

            "group":
                group_map[
                    subject
                ],

            "bin":
                b,

            "G1_bin_mean":
                mean_raw,

            "G1_bin_mean_withinsubject_z_QC":
                mean_z,

            "n_voxels_bin":
                int(
                    np.sum(
                        idx
                    )
                ),
        })

        vals_print.append(
            mean_raw
        )

    print(
        f"[{si:02d}/64] "
        f"{subject} {group_map[subject]} | "
        f"bin1={vals_print[0]:+.5f} | "
        f"bin10={vals_print[-1]:+.5f}"
    )


long_df = pd.DataFrame(
    long_rows
)

if len(long_df) != (
    64 * 10
):
    raise RuntimeError(
        f"Long rows = {len(long_df)}"
    )

long_df.to_csv(
    TABLE_OUT /
    "individual_G1_fixed375_bins_long.tsv",
    sep="\t",
    index=False
)


# wide convenience table
wide = (
    long_df
    .pivot(
        index=[
            "subject",
            "group",
        ],
        columns="bin",
        values="G1_bin_mean"
    )
    .reset_index()
)

wide.columns = [
    (
        f"G1_bin{int(c)}"
        if isinstance(
            c,
            (int, np.integer)
        )
        else
        c
    )
    for c in wide.columns
]

wide.to_csv(
    TABLE_OUT /
    "individual_G1_fixed375_bins_wide.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 7. MERGE COVARIATES
# =============================================================================

print()
print("=" * 110)
print("STEP 7. MERGE AGE / SEX / FD")
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


subject_cov = clin.merge(
    motion,
    on="subject",
    how="inner",
    validate="one_to_one"
)

if not np.all(
    subject_cov[
        "group"
    ]
    ==
    subject_cov[
        "group_motion"
    ]
):
    raise RuntimeError(
        "Clinical/motion group mismatch."
    )

subject_cov = subject_cov.drop(
    columns=[
        "group_motion"
    ]
)


subject_cov[
    "age"
] = pd.to_numeric(
    subject_cov[
        "age"
    ],
    errors="coerce"
)

subject_cov[
    "mean_FD_posttrim"
] = pd.to_numeric(
    subject_cov[
        "mean_FD_posttrim"
    ],
    errors="coerce"
)


subject_cov = subject_cov[
    subject_cov[
        "subject"
    ].isin(
        subjects
    )
].copy()


if (
    len(subject_cov) != 64
    or
    subject_cov[
        "subject"
    ].nunique() != 64
):
    raise RuntimeError(
        "Covariate table is not exact final64."
    )


subject_cov[
    "age_centered"
] = (
    subject_cov[
        "age"
    ]
    -
    subject_cov[
        "age"
    ].mean()
)

subject_cov[
    "mean_FD_posttrim_centered"
] = (
    subject_cov[
        "mean_FD_posttrim"
    ]
    -
    subject_cov[
        "mean_FD_posttrim"
    ].mean()
)


analysis = long_df.merge(
    subject_cov[
        [
            "subject",
            "age",
            "sex",
            "mean_FD_posttrim",
            "age_centered",
            "mean_FD_posttrim_centered",
        ]
    ],
    on="subject",
    how="inner",
    validate="many_to_one"
)


if len(
    analysis
) != 640:
    raise RuntimeError(
        f"Analysis rows = {len(analysis)}"
    )

if analysis[
    [
        "G1_bin_mean",
        "age",
        "sex",
        "mean_FD_posttrim",
    ]
].isna().sum().sum() != 0:

    raise RuntimeError(
        "Missing analysis values."
    )


analysis[
    "bin"
] = analysis[
    "bin"
].astype(
    int
)

analysis.to_csv(
    TABLE_OUT /
    "current64_fixed375_bin_covariate_long.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 8. RAW GROUP PROFILE SUMMARY
# =============================================================================

print()
print("=" * 110)
print("STEP 8. GROUP × BIN DESCRIPTIVES")
print("=" * 110)

summary_rows = []

for b in range(
    1,
    11
):

    for group in [
        "HC",
        "SZ",
    ]:

        x = analysis.loc[
            (
                analysis[
                    "bin"
                ]
                ==
                b
            )
            &
            (
                analysis[
                    "group"
                ]
                ==
                group
            ),
            "G1_bin_mean"
        ].to_numpy(
            dtype=float
        )

        summary_rows.append({
            "bin":
                b,

            "group":
                group,

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

            "SEM":
                float(
                    stats.sem(
                        x
                    )
                ),
        })


profile_summary = pd.DataFrame(
    summary_rows
)

profile_summary.to_csv(
    TABLE_OUT /
    "fixed375_bin_group_profile_summary.tsv",
    sep="\t",
    index=False
)

print(
    profile_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# STEP 9. MIXED MODEL — GROUP × BIN
# =============================================================================

print()
print("=" * 110)
print("STEP 9. MIXED MODEL: GROUP × BIN")
print("=" * 110)

# Bin treated as categorical.
# Random intercept = subject.
#
# Full model:
# G1 ~ Group * Bin + covariates + (1|subject)
#
# Reduced model removes Group × Bin interaction.
# Both fitted using ML for likelihood-ratio comparison.

full_formula = (
    'G1_bin_mean ~ '
    'C(group, Treatment(reference="HC")) '
    '* C(bin, Treatment(reference=1)) '
    '+ age_centered '
    '+ C(sex, Treatment(reference="Female")) '
    '+ mean_FD_posttrim_centered'
)

reduced_formula = (
    'G1_bin_mean ~ '
    'C(group, Treatment(reference="HC")) '
    '+ C(bin, Treatment(reference=1)) '
    '+ age_centered '
    '+ C(sex, Treatment(reference="Female")) '
    '+ mean_FD_posttrim_centered'
)


print()
print("FULL:")
print(
    full_formula
)

full = fit_mixedlm(
    full_formula,
    analysis
)

print()
print("REDUCED:")
print(
    reduced_formula
)

reduced = fit_mixedlm(
    reduced_formula,
    analysis
)


# =============================================================================
# STEP 10. OMNIBUS GROUP × BIN INTERACTION
# =============================================================================

print()
print("=" * 110)
print("STEP 10. OMNIBUS GROUP × BIN INTERACTION")
print("=" * 110)

lrt_stat = float(
    2.0
    *
    (
        full.llf
        -
        reduced.llf
    )
)

df_diff = int(
    len(
        full.fe_params
    )
    -
    len(
        reduced.fe_params
    )
)

lrt_p = float(
    stats.chi2.sf(
        lrt_stat,
        df_diff
    )
)

interaction_result = pd.DataFrame([
    {
        "comparison":
            "Full Group*Bin vs reduced no-interaction",

        "full_logLik":
            float(
                full.llf
            ),

        "reduced_logLik":
            float(
                reduced.llf
            ),

        "LRT_chi2":
            lrt_stat,

        "df":
            df_diff,

        "p":
            lrt_p,

        "full_AIC":
            float(
                full.aic
            ),

        "reduced_AIC":
            float(
                reduced.aic
            ),

        "full_converged":
            bool(
                full.converged
            ),

        "reduced_converged":
            bool(
                reduced.converged
            ),
    }
])

interaction_result.to_csv(
    TABLE_OUT /
    "fixed375_bins_GroupXBin_mixedmodel_LRT.tsv",
    sep="\t",
    index=False
)

print(
    interaction_result.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# STEP 11. FULL MIXED MODEL FIXED-EFFECT TABLE
# =============================================================================

fe = full.fe_params
se = full.bse_fe

fixed_rows = []

for name in fe.index:

    beta = float(
        fe[
            name
        ]
    )

    s = float(
        se[
            name
        ]
    )

    z = (
        beta / s
    )

    p = float(
        2.0
        *
        stats.norm.sf(
            abs(z)
        )
    )

    fixed_rows.append({
        "term":
            name,

        "beta":
            beta,

        "SE":
            s,

        "z":
            z,

        "p":
            p,

        "CI95_lower":
            beta
            -
            1.96
            *
            s,

        "CI95_upper":
            beta
            +
            1.96
            *
            s,
    })


fixed_effects = pd.DataFrame(
    fixed_rows
)

fixed_effects.to_csv(
    TABLE_OUT /
    "fixed375_bins_full_mixedmodel_fixed_effects.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 12. BIN-WISE HC vs SZ CONTRASTS
# =============================================================================

print()
print("=" * 110)
print("STEP 12. BIN-WISE ADJUSTED HC vs SZ CONTRASTS")
print("=" * 110)

contrast_rows = []

group_term = (
    'C(group, Treatment(reference="HC"))[T.SZ]'
)

for b in range(
    1,
    11
):

    d = analysis[
        analysis[
            "bin"
        ]
        ==
        b
    ].copy()

    formula = (
        'G1_bin_mean ~ '
        'C(group, Treatment(reference="HC")) '
        '+ age_centered '
        '+ C(sex, Treatment(reference="Female")) '
        '+ mean_FD_posttrim_centered'
    )

    model = smf.ols(
        formula,
        data=d
    ).fit()

    robust = model.get_robustcov_results(
        cov_type="HC3"
    )

    idx = list(
        model.params.index
    ).index(
        group_term
    )

    ci = robust.conf_int()[
        idx
    ]

    hc_x = d.loc[
        d[
            "group"
        ]
        ==
        "HC",
        "G1_bin_mean"
    ].to_numpy(
        dtype=float
    )

    sz_x = d.loc[
        d[
            "group"
        ]
        ==
        "SZ",
        "G1_bin_mean"
    ].to_numpy(
        dtype=float
    )

    welch = stats.ttest_ind(
        hc_x,
        sz_x,
        equal_var=False
    )

    # adjusted marginal means
    phc = d.copy()
    psz = d.copy()

    phc[
        "group"
    ] = "HC"

    psz[
        "group"
    ] = "SZ"

    adj_hc = float(
        model.predict(
            phc
        ).mean()
    )

    adj_sz = float(
        model.predict(
            psz
        ).mean()
    )

    contrast_rows.append({
        "bin":
            b,

        "n_voxels":
            int(
                composition.loc[
                    composition[
                        "bin"
                    ]
                    ==
                    b,
                    "n_voxels"
                ].iloc[
                    0
                ]
            ),

        "reference_G1_mean":
            float(
                composition.loc[
                    composition[
                        "bin"
                    ]
                    ==
                    b,
                    "reference_G1_mean"
                ].iloc[
                    0
                ]
            ),

        "HC_raw_mean":
            float(
                np.mean(
                    hc_x
                )
            ),

        "SZ_raw_mean":
            float(
                np.mean(
                    sz_x
                )
            ),

        "HC_minus_SZ_raw":
            float(
                np.mean(
                    hc_x
                )
                -
                np.mean(
                    sz_x
                )
            ),

        "Welch_p_unadjusted":
            float(
                welch.pvalue
            ),

        "Hedges_g_HC_minus_SZ":
            hedges_g(
                hc_x,
                sz_x
            ),

        "adjusted_HC_mean":
            adj_hc,

        "adjusted_SZ_mean":
            adj_sz,

        "beta_SZ_minus_HC":
            float(
                model.params[
                    group_term
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
                ci[
                    0
                ]
            ),

        "CI95_HC3_upper":
            float(
                ci[
                    1
                ]
            ),
    })


contrasts = pd.DataFrame(
    contrast_rows
)

reject, qvals, _, _ = multipletests(
    contrasts[
        "p_HC3"
    ].to_numpy(
        dtype=float
    ),
    alpha=0.05,
    method="fdr_bh"
)

contrasts[
    "FDR_BH_q"
] = qvals

contrasts[
    "FDR_BH_significant"
] = reject


contrasts.to_csv(
    TABLE_OUT /
    "fixed375_bins_binwise_HC_vs_SZ_adjusted_FDR.tsv",
    sep="\t",
    index=False
)

print(
    contrasts.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# STEP 13. OPTIONAL SHAPE-ONLY MIXED MODEL QC
# =============================================================================

print()
print("=" * 110)
print("STEP 13. SHAPE-ONLY WITHIN-SUBJECT-Z PROFILE QC")
print("=" * 110)

z_full_formula = (
    'G1_bin_mean_withinsubject_z_QC ~ '
    'C(group, Treatment(reference="HC")) '
    '* C(bin, Treatment(reference=1)) '
    '+ age_centered '
    '+ C(sex, Treatment(reference="Female")) '
    '+ mean_FD_posttrim_centered'
)

z_reduced_formula = (
    'G1_bin_mean_withinsubject_z_QC ~ '
    'C(group, Treatment(reference="HC")) '
    '+ C(bin, Treatment(reference=1)) '
    '+ age_centered '
    '+ C(sex, Treatment(reference="Female")) '
    '+ mean_FD_posttrim_centered'
)

z_full = fit_mixedlm(
    z_full_formula,
    analysis
)

z_reduced = fit_mixedlm(
    z_reduced_formula,
    analysis
)

z_lrt = float(
    2.0
    *
    (
        z_full.llf
        -
        z_reduced.llf
    )
)

z_df = int(
    len(
        z_full.fe_params
    )
    -
    len(
        z_reduced.fe_params
    )
)

z_p = float(
    stats.chi2.sf(
        z_lrt,
        z_df
    )
)

shape_sensitivity = pd.DataFrame([
    {
        "analysis":
            "within-subject z-scored G1 profile",

        "LRT_chi2":
            z_lrt,

        "df":
            z_df,

        "GroupXBin_p":
            z_p,

        "role":
            "shape-only sensitivity QC",
    }
])

shape_sensitivity.to_csv(
    TABLE_OUT /
    "fixed375_bins_shape_only_sensitivity.tsv",
    sep="\t",
    index=False
)

print(
    shape_sensitivity.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# STEP 14. FIGURE 1 — ANATOMICAL COMPOSITION
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        9.0,
        5.5
    )
)

x = composition[
    "bin"
].to_numpy()

bottom = np.zeros(
    len(
        composition
    )
)

for column, label in [
    (
        "VST_percent",
        "VST"
    ),
    (
        "Caudate_percent",
        "Caudate"
    ),
    (
        "Putamen_percent",
        "Putamen"
    ),
]:

    values = composition[
        column
    ].to_numpy()

    ax.bar(
        x,
        values,
        bottom=bottom,
        label=label
    )

    bottom = (
        bottom + values
    )


ax.set_ylim(
    0,
    100
)

ax.set_xticks(
    np.arange(
        1,
        11
    )
)

ax.set_xlabel(
    "Fixed 375 normative G1 bin"
)

ax.set_ylabel(
    "Anatomical composition (%)"
)

ax.set_title(
    "Anatomical composition of fixed normative G1 bins"
)

ax.legend(
    frameon=False,
    ncol=3
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
    "Figure_fixed375_bin_anatomical_composition.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG_OUT /
    "Figure_fixed375_bin_anatomical_composition.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# STEP 15. FIGURE 2 — HC/SZ PROFILE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        9.0,
        6.0
    )
)

for group in [
    "HC",
    "SZ",
]:

    d = (
        profile_summary[
            profile_summary[
                "group"
            ]
            ==
            group
        ]
        .sort_values(
            "bin"
        )
    )

    ax.errorbar(
        d[
            "bin"
        ],
        d[
            "mean"
        ],
        yerr=d[
            "SEM"
        ],
        marker="o",
        capsize=3,
        linewidth=2,
        label=group
    )


ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    alpha=0.5
)

ax.set_xticks(
    np.arange(
        1,
        11
    )
)

ax.set_xlabel(
    "Fixed 375 normative G1 bin"
)

ax.set_ylabel(
    "Mean individual G1"
)

ax.set_title(
    "HC and SZ profiles along the fixed 375 normative G1 axis"
)

ax.legend(
    frameon=False
)

ax.text(
    0.02,
    0.98,
    (
        f"Group × Bin LRT:\n"
        f"χ²({df_diff}) = {lrt_stat:.2f}, "
        f"p = {lrt_p:.4g}"
    ),
    transform=ax.transAxes,
    ha="left",
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
    "Figure_HC_SZ_fixed375_normative_bin_profiles.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG_OUT /
    "Figure_HC_SZ_fixed375_normative_bin_profiles.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# STEP 16. FIGURE 3 — BIN-WISE ADJUSTED EFFECTS
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        9.0,
        5.8
    )
)

x = contrasts[
    "bin"
].to_numpy(
    dtype=float
)

beta = contrasts[
    "beta_SZ_minus_HC"
].to_numpy(
    dtype=float
)

lo = contrasts[
    "CI95_HC3_lower"
].to_numpy(
    dtype=float
)

hi = contrasts[
    "CI95_HC3_upper"
].to_numpy(
    dtype=float
)

yerr = np.vstack([
    beta - lo,
    hi - beta,
])

ax.errorbar(
    x,
    beta,
    yerr=yerr,
    marker="o",
    capsize=4,
    linewidth=1.8
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

for _, r in contrasts.iterrows():

    if bool(
        r[
            "FDR_BH_significant"
        ]
    ):

        ax.text(
            r[
                "bin"
            ],
            r[
                "CI95_HC3_upper"
            ],
            "*",
            ha="center",
            va="bottom",
            fontsize=14
        )


ax.set_xticks(
    np.arange(
        1,
        11
    )
)

ax.set_xlabel(
    "Fixed 375 normative G1 bin"
)

ax.set_ylabel(
    "Adjusted SZ − HC difference in G1"
)

ax.set_title(
    "Bin-wise diagnostic-group effects along the normative G1 axis"
)

ax.text(
    0.02,
    0.98,
    "* FDR-BH q < .05",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10
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
    "Figure_binwise_adjusted_SZ_minus_HC_effects_FDR.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG_OUT /
    "Figure_binwise_adjusted_SZ_minus_HC_effects_FDR.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# STEP 17. FINAL QC
# =============================================================================

bin_sizes = (
    composition[
        "n_voxels"
    ].to_numpy(
        dtype=int
    )
)

reference_bin_means = (
    composition[
        "reference_G1_mean"
    ].to_numpy(
        dtype=float
    )
)

bins_monotonic = bool(
    np.all(
        np.diff(
            reference_bin_means
        )
        >
        0
    )
)

bin_size_ok = bool(
    np.max(
        bin_sizes
    )
    -
    np.min(
        bin_sizes
    )
    <=
    1
)

subject_bin_counts = (
    analysis
    .groupby(
        "subject"
    )[
        "bin"
    ]
    .nunique()
)

all_subjects_have_10_bins = bool(
    np.all(
        subject_bin_counts
        ==
        10
    )
)

all_finite = bool(
    np.all(
        np.isfinite(
            analysis[
                [
                    "G1_bin_mean",
                    "G1_bin_mean_withinsubject_z_QC",
                    "age",
                    "mean_FD_posttrim",
                ]
            ].to_numpy(
                dtype=float
            )
        )
    )
)

final_pass = bool(
    len(
        analysis
    )
    ==
    640
    and
    analysis[
        "subject"
    ].nunique()
    ==
    64
    and
    bins_monotonic
    and
    bin_size_ok
    and
    all_subjects_have_10_bins
    and
    all_finite
    and
    full.converged
    and
    reduced.converged
)


qc = {
    "N_subjects":
        64,

    "HC":
        32,

    "SZ":
        32,

    "N_bins":
        10,

    "total_voxels":
        2608,

    "bin_definition":
        (
            "Frozen 375 normative G1 mapped to the "
            "common2608 space, sorted from lowest to highest "
            "G1 and divided into 10 approximately equal-sized "
            "spatial bins. The identical voxel-to-bin assignment "
            "was used for all 64 participants."
        ),

    "reference":
        str(
            REF_FILE
        ),

    "global_sign":
        GLOBAL_SIGN,

    "bin_sizes":
        bin_sizes.tolist(),

    "bin_reference_means_monotonically_increasing":
        bins_monotonic,

    "all_subjects_have_10_bins":
        all_subjects_have_10_bins,

    "mixed_model":
        (
            "G1_bin_mean ~ Group * categorical Bin + "
            "Age + Sex + mean_FD_posttrim + "
            "random intercept for Subject"
        ),

    "mixed_model_GroupXBin_LRT_chi2":
        lrt_stat,

    "mixed_model_GroupXBin_df":
        df_diff,

    "mixed_model_GroupXBin_p":
        lrt_p,

    "shape_only_sensitivity_GroupXBin_p":
        z_p,

    "binwise_multiple_testing":
        "Benjamini-Hochberg FDR across 10 adjusted HC-vs-SZ bin contrasts",

    "n_FDR_significant_bins":
        int(
            np.sum(
                contrasts[
                    "FDR_BH_significant"
                ]
            )
        ),

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
    "FINAL_FIXED375_NORMATIVE_BINS_QC.json",
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
print("FIXED BIN DEFINITION")
print("-" * 100)

print(
    composition[
        [
            "bin",
            "n_voxels",
            "reference_G1_mean",
            "VST_percent",
            "Caudate_percent",
            "Putamen_percent",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print()
print("PRIMARY MIXED MODEL")
print("-" * 100)

print(
    f"Group × Bin LRT: "
    f"chi2({df_diff}) = "
    f"{lrt_stat:.6f}"
)

print(
    f"p = "
    f"{lrt_p:.8f}"
)

print(
    "Full converged =",
    full.converged
)

print(
    "Reduced converged =",
    reduced.converged
)


print()
print("BIN-WISE ADJUSTED HC vs SZ")
print("-" * 100)

print(
    contrasts[
        [
            "bin",
            "HC_raw_mean",
            "SZ_raw_mean",
            "beta_SZ_minus_HC",
            "CI95_HC3_lower",
            "CI95_HC3_upper",
            "p_HC3",
            "FDR_BH_q",
            "FDR_BH_significant",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print()
print("SHAPE-ONLY SENSITIVITY")
print("-" * 100)

print(
    f"Within-subject z-scored "
    f"Group × Bin p = "
    f"{z_p:.8f}"
)


print()
print(
    "FDR significant bins =",
    contrasts.loc[
        contrasts[
            "FDR_BH_significant"
        ],
        "bin"
    ].tolist()
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
    "fixed375_normative_bin_definition_2608voxels.tsv",

    TABLE_OUT /
    "fixed375_normative_bin_anatomical_composition.tsv",

    TABLE_OUT /
    "individual_G1_fixed375_bins_long.tsv",

    TABLE_OUT /
    "individual_G1_fixed375_bins_wide.tsv",

    TABLE_OUT /
    "current64_fixed375_bin_covariate_long.tsv",

    TABLE_OUT /
    "fixed375_bin_group_profile_summary.tsv",

    TABLE_OUT /
    "fixed375_bins_GroupXBin_mixedmodel_LRT.tsv",

    TABLE_OUT /
    "fixed375_bins_full_mixedmodel_fixed_effects.tsv",

    TABLE_OUT /
    "fixed375_bins_binwise_HC_vs_SZ_adjusted_FDR.tsv",

    TABLE_OUT /
    "fixed375_bins_shape_only_sensitivity.tsv",
]:

    print(p)


print()
print("FIGURES")
print("-" * 100)

for p in [
    FIG_OUT /
    "Figure_fixed375_bin_anatomical_composition.png",

    FIG_OUT /
    "Figure_HC_SZ_fixed375_normative_bin_profiles.png",

    FIG_OUT /
    "Figure_binwise_adjusted_SZ_minus_HC_effects_FDR.png",
]:

    print(p)
