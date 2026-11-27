#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib

from scipy import stats
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
    "01_anatomical_separation"
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


# Individual Procrustes gradients
IND_ROOT = (
    ROOT /
    "19_current64_375reference_gradients" /
    "04_subject_gradients"
)

# Frozen 375 reference mapped to exact common2608 space
REF_FILE = (
    ROOT /
    "19_current64_375reference_gradients" /
    "01_reference" /
    "375_group_gradients_mapped_2608x10.npy"
)

# exact 2847 -> 2608 mapping
ROW_MAP_FILE = (
    ROOT /
    "19_current64_375reference_gradients" /
    "01_reference" /
    "row_mapping_2847_to_2608.npy"
)

# Original 375 voxel order
COORD_FILE = (
    ROOT /
    "06_fingerprints" /
    "seed_voxel_coordinates.tsv"
)

# Group lists
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

# Anatomical masks
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


# =============================================================================
# SETTINGS
# =============================================================================

RNG = np.random.default_rng(20260825)

N_BOOT = 20000
N_PERM = 50000


# =============================================================================
# HELPERS
# =============================================================================

def read_subject_list(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        x = [
            line.strip()
            for line in f
            if line.strip()
            and not line.startswith("#")
        ]

    return x


def corr(a, b):

    a = np.asarray(
        a,
        dtype=float
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=float
    ).reshape(-1)

    return float(
        np.corrcoef(
            a,
            b
        )[0, 1]
    )


def cohens_d_independent(
    x,
    y
):
    """
    Effect direction:
        x - y

    Here:
        HC - SZ
    """

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

    sx2 = np.var(
        x,
        ddof=1
    )

    sy2 = np.var(
        y,
        ddof=1
    )

    pooled_sd = np.sqrt(
        (
            (nx - 1) * sx2
            +
            (ny - 1) * sy2
        )
        /
        (nx + ny - 2)
    )

    d = (
        np.mean(x)
        -
        np.mean(y)
    ) / pooled_sd

    return float(d)


def hedges_g_independent(
    x,
    y
):

    d = cohens_d_independent(
        x,
        y
    )

    nx = len(x)
    ny = len(y)

    df = (
        nx
        +
        ny
        -
        2
    )

    correction = (
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
        correction * d
    )


def bootstrap_mean_difference(
    hc,
    sz,
    n_boot=20000
):
    """
    Stratified bootstrap.
    Difference = HC mean - SZ mean.
    """

    hc = np.asarray(
        hc,
        dtype=float
    )

    sz = np.asarray(
        sz,
        dtype=float
    )

    boot = np.empty(
        n_boot,
        dtype=float
    )

    for i in range(
        n_boot
    ):

        bhc = RNG.choice(
            hc,
            size=len(hc),
            replace=True
        )

        bsz = RNG.choice(
            sz,
            size=len(sz),
            replace=True
        )

        boot[i] = (
            np.mean(bhc)
            -
            np.mean(bsz)
        )

    lo, hi = np.percentile(
        boot,
        [
            2.5,
            97.5,
        ]
    )

    return (
        float(lo),
        float(hi)
    )


def permutation_pvalue(
    hc,
    sz,
    n_perm=50000
):
    """
    Two-sided permutation test on mean difference.
    """

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
        n_perm
    ):

        perm = RNG.permutation(
            pooled
        )

        diff = (
            np.mean(
                perm[:n_hc]
            )
            -
            np.mean(
                perm[n_hc:]
            )
        )

        if abs(diff) >= abs(
            observed
        ):
            extreme += 1

    p = (
        extreme + 1
    ) / (
        n_perm + 1
    )

    return float(p)


def summarize_group(
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

required = [
    IND_ROOT,
    REF_FILE,
    ROW_MAP_FILE,
    COORD_FILE,
    HC_FILE,
    SZ_FILE,
    VST_FILE,
    CAUDATE_FILE,
    PUTAMEN_FILE,
]

for p in required:

    print(
        f"{p} -> {p.exists()}"
    )

    if not p.exists():
        raise FileNotFoundError(
            p
        )


# =============================================================================
# STEP 2. SUBJECTS
# =============================================================================

print()
print("=" * 110)
print("STEP 2. SUBJECT LISTS")
print("=" * 110)

hc = read_subject_list(
    HC_FILE
)

sz = read_subject_list(
    SZ_FILE
)

if len(hc) != 32:
    raise RuntimeError(
        f"HC N = {len(hc)}"
    )

if len(sz) != 32:
    raise RuntimeError(
        f"SZ N = {len(sz)}"
    )

if set(hc) & set(sz):
    raise RuntimeError(
        "HC/SZ overlap detected."
    )

subjects = hc + sz

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

print(
    "TOTAL =",
    len(subjects)
)


# =============================================================================
# STEP 3. BUILD 2608 ANATOMICAL LABEL VECTOR
# =============================================================================

print()
print("=" * 110)
print("STEP 3. BUILD COMMON2608 ANATOMICAL LABELS")
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
        f"Bad coordinate shape: {ijk2847.shape}"
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
        f"row_map shape = {row_map.shape}"
    )


def sample_mask(
    path
):

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

labels2608 = labels2847[
    row_map
]

if np.any(
    labels2608 == ""
):
    raise RuntimeError(
        "Unassigned common2608 voxels."
    )

counts = pd.Series(
    labels2608
).value_counts()

print(counts)

expected = {
    "VST": 156,
    "Caudate": 970,
    "Putamen": 1482,
}

for label, n in expected.items():

    actual = int(
        np.sum(
            labels2608
            ==
            label
        )
    )

    if actual != n:
        raise RuntimeError(
            f"{label}: expected {n}, "
            f"got {actual}"
        )


# =============================================================================
# STEP 4. DETERMINE ONE GLOBAL PAPER ORIENTATION
# =============================================================================

print()
print("=" * 110)
print("STEP 4. GLOBAL G1 ORIENTATION")
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
        f"Reference shape = {ref.shape}"
    )

ref_g1 = ref[
    :,
    0
].copy()

ref_vst = np.mean(
    ref_g1[
        labels2608 == "VST"
    ]
)

ref_cau = np.mean(
    ref_g1[
        labels2608 == "Caudate"
    ]
)

ref_put = np.mean(
    ref_g1[
        labels2608 == "Putamen"
    ]
)

ref_low = np.mean(
    [
        ref_vst,
        ref_cau,
    ]
)

# One GLOBAL sign for everybody.
# We do NOT orient each participant separately.
if ref_put > ref_low:

    GLOBAL_SIGN = 1.0

else:

    GLOBAL_SIGN = -1.0


ref_g1 = (
    GLOBAL_SIGN
    *
    ref_g1
)

print(
    "Reference VST mean     =",
    np.mean(
        ref_g1[
            labels2608 == "VST"
        ]
    )
)

print(
    "Reference Caudate mean =",
    np.mean(
        ref_g1[
            labels2608 == "Caudate"
        ]
    )
)

print(
    "Reference Putamen mean =",
    np.mean(
        ref_g1[
            labels2608 == "Putamen"
        ]
    )
)

print(
    "GLOBAL_SIGN =",
    GLOBAL_SIGN
)

print(
    "Convention: low G1 = VST/caudate; "
    "high G1 = putamen"
)


# =============================================================================
# STEP 5. INDIVIDUAL METRICS
# =============================================================================

print()
print("=" * 110)
print("STEP 5. INDIVIDUAL G1 ANATOMICAL SEPARATION")
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
            f"{subject}: gradient shape = {G.shape}"
        )

    if not np.all(
        np.isfinite(G)
    ):
        raise RuntimeError(
            f"{subject}: nonfinite gradient values."
        )

    g1 = (
        GLOBAL_SIGN
        *
        G[:, 0]
    )

    vst_mean = float(
        np.mean(
            g1[
                labels2608
                ==
                "VST"
            ]
        )
    )

    cau_mean = float(
        np.mean(
            g1[
                labels2608
                ==
                "Caudate"
            ]
        )
    )

    put_mean = float(
        np.mean(
            g1[
                labels2608
                ==
                "Putamen"
            ]
        )
    )

    low_mean = float(
        np.mean(
            [
                vst_mean,
                cau_mean,
            ]
        )
    )

    separation = float(
        put_mean
        -
        low_mean
    )

    # QC only:
    # correspondence after the already-completed Procrustes alignment
    r_ref = corr(
        g1,
        ref_g1
    )

    rows.append({
        "subject":
            subject,

        "group":
            group_map[
                subject
            ],

        "G1_VST_mean":
            vst_mean,

        "G1_caudate_mean":
            cau_mean,

        "G1_putamen_mean":
            put_mean,

        "G1_low_VST_caudate_mean":
            low_mean,

        "G1_separation":
            separation,

        "G1_vs_375_reference_r_QC":
            r_ref,
    })

    print(
        f"[{i:02d}/64] "
        f"{subject} | "
        f"{group_map[subject]:2s} | "
        f"VST={vst_mean:+.6f} | "
        f"CAU={cau_mean:+.6f} | "
        f"PUT={put_mean:+.6f} | "
        f"SEP={separation:+.6f} | "
        f"r375={r_ref:.3f}"
    )


df = pd.DataFrame(
    rows
)

df.to_csv(
    TABLE_OUT /
    "individual_G1_anatomical_separation.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 6. GROUP SUMMARY
# =============================================================================

print()
print("=" * 110)
print("STEP 6. GROUP SUMMARY")
print("=" * 110)

metric_cols = [
    "G1_VST_mean",
    "G1_caudate_mean",
    "G1_putamen_mean",
    "G1_low_VST_caudate_mean",
    "G1_separation",
]

summary_rows = []

for metric in metric_cols:

    for group in [
        "HC",
        "SZ",
    ]:

        x = df.loc[
            df["group"] == group,
            metric
        ].to_numpy(
            dtype=float
        )

        s = summarize_group(
            x
        )

        summary_rows.append({
            "metric":
                metric,

            "group":
                group,

            **s,
        })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    TABLE_OUT /
    "G1_anatomical_metrics_group_summary.tsv",
    sep="\t",
    index=False
)

print(
    summary_df[
        summary_df[
            "metric"
        ]
        ==
        "G1_separation"
    ].to_string(
        index=False
    )
)


# =============================================================================
# STEP 7. PRIMARY HC vs SZ TEST
# =============================================================================

print()
print("=" * 110)
print("STEP 7. PRIMARY HC vs SZ TEST: G1_separation")
print("=" * 110)

hc_sep = df.loc[
    df["group"] == "HC",
    "G1_separation"
].to_numpy(
    dtype=float
)

sz_sep = df.loc[
    df["group"] == "SZ",
    "G1_separation"
].to_numpy(
    dtype=float
)

welch = stats.ttest_ind(
    hc_sep,
    sz_sep,
    equal_var=False
)

# Welch-Satterthwaite df
nx = len(hc_sep)
ny = len(sz_sep)

vx = np.var(
    hc_sep,
    ddof=1
)

vy = np.var(
    sz_sep,
    ddof=1
)

welch_df = (
    (
        vx / nx
        +
        vy / ny
    ) ** 2
    /
    (
        (
            (vx / nx) ** 2
            /
            (nx - 1)
        )
        +
        (
            (vy / ny) ** 2
            /
            (ny - 1)
        )
    )
)

mean_diff = float(
    np.mean(hc_sep)
    -
    np.mean(sz_sep)
)

hedges_g = hedges_g_independent(
    hc_sep,
    sz_sep
)

ci_lo, ci_hi = bootstrap_mean_difference(
    hc_sep,
    sz_sep,
    n_boot=N_BOOT
)

perm_p = permutation_pvalue(
    hc_sep,
    sz_sep,
    n_perm=N_PERM
)

mw = stats.mannwhitneyu(
    hc_sep,
    sz_sep,
    alternative="two-sided"
)

primary_stats = pd.DataFrame([
    {
        "outcome":
            "G1_separation",

        "contrast":
            "HC_minus_SZ",

        "HC_N":
            len(hc_sep),

        "SZ_N":
            len(sz_sep),

        "HC_mean":
            float(
                np.mean(hc_sep)
            ),

        "HC_SD":
            float(
                np.std(
                    hc_sep,
                    ddof=1
                )
            ),

        "SZ_mean":
            float(
                np.mean(sz_sep)
            ),

        "SZ_SD":
            float(
                np.std(
                    sz_sep,
                    ddof=1
                )
            ),

        "mean_difference_HC_minus_SZ":
            mean_diff,

        "bootstrap95CI_lower":
            ci_lo,

        "bootstrap95CI_upper":
            ci_hi,

        "Welch_t":
            float(
                welch.statistic
            ),

        "Welch_df":
            float(
                welch_df
            ),

        "Welch_p_two_sided":
            float(
                welch.pvalue
            ),

        "Hedges_g_HC_minus_SZ":
            hedges_g,

        "permutation_p_two_sided":
            perm_p,

        "permutation_N":
            N_PERM,

        "Mann_Whitney_U":
            float(
                mw.statistic
            ),

        "Mann_Whitney_p_two_sided":
            float(
                mw.pvalue
            ),
    }
])

primary_stats.to_csv(
    TABLE_OUT /
    "PRIMARY_HC_vs_SZ_G1_separation_stats.tsv",
    sep="\t",
    index=False
)

print(
    primary_stats.T.to_string(
        header=False
    )
)


# =============================================================================
# STEP 8. REGIONAL MEANS: DESCRIPTIVE GROUP CONTRASTS
# =============================================================================

regional_rows = []

for metric in [
    "G1_VST_mean",
    "G1_caudate_mean",
    "G1_putamen_mean",
]:

    hc_x = df.loc[
        df["group"] == "HC",
        metric
    ].to_numpy(
        dtype=float
    )

    sz_x = df.loc[
        df["group"] == "SZ",
        metric
    ].to_numpy(
        dtype=float
    )

    test = stats.ttest_ind(
        hc_x,
        sz_x,
        equal_var=False
    )

    regional_rows.append({
        "metric":
            metric,

        "HC_mean":
            float(
                np.mean(hc_x)
            ),

        "HC_SD":
            float(
                np.std(
                    hc_x,
                    ddof=1
                )
            ),

        "SZ_mean":
            float(
                np.mean(sz_x)
            ),

        "SZ_SD":
            float(
                np.std(
                    sz_x,
                    ddof=1
                )
            ),

        "HC_minus_SZ":
            float(
                np.mean(hc_x)
                -
                np.mean(sz_x)
            ),

        "Welch_t_descriptive":
            float(
                test.statistic
            ),

        "Welch_p_unadjusted_descriptive":
            float(
                test.pvalue
            ),

        "Hedges_g_HC_minus_SZ":
            hedges_g_independent(
                hc_x,
                sz_x
            ),
    })


regional_stats = pd.DataFrame(
    regional_rows
)

regional_stats.to_csv(
    TABLE_OUT /
    "regional_G1_means_HC_vs_SZ_DESCRIPTIVE.tsv",
    sep="\t",
    index=False
)

print()
print(
    "Regional tests are DESCRIPTIVE/secondary; "
    "G1_separation is the prespecified primary outcome."
)


# =============================================================================
# STEP 9. FIGURE 1 — INDIVIDUAL SEPARATION
# =============================================================================

fig, ax = plt.subplots(
    figsize=(6.5, 6.2)
)

groups = [
    "HC",
    "SZ",
]

positions = [
    1,
    2,
]

data = [
    hc_sep,
    sz_sep,
]

bp = ax.boxplot(
    data,
    positions=positions,
    widths=0.48,
    showfliers=False,
    patch_artist=False
)

# deterministic jitter
plot_rng = np.random.default_rng(
    12345
)

for xpos, x in zip(
    positions,
    data
):

    jitter = plot_rng.normal(
        loc=0,
        scale=0.055,
        size=len(x)
    )

    ax.scatter(
        xpos + jitter,
        x,
        s=32,
        alpha=0.75,
        zorder=3
    )

    mean = np.mean(x)

    sem = stats.sem(x)

    ci = stats.t.ppf(
        0.975,
        len(x) - 1
    ) * sem

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


ax.axhline(
    0,
    linewidth=1,
    linestyle="--",
    alpha=0.5
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
    "G1 anatomical separation\n"
    "Putamen − mean(VST, Caudate)"
)

ax.set_title(
    "Individual G1 anatomical separation"
)

annotation = (
    f"HC − SZ = {mean_diff:.4f}\n"
    f"Welch p = {welch.pvalue:.4g}\n"
    f"Hedges' g = {hedges_g:.3f}"
)

ax.text(
    0.98,
    0.98,
    annotation,
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
    "Figure_G1_separation_HC_vs_SZ.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG_OUT /
    "Figure_G1_separation_HC_vs_SZ.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# STEP 10. FIGURE 2 — REGIONAL G1 PROFILE
# =============================================================================

regions = [
    "VST",
    "Caudate",
    "Putamen",
]

region_cols = [
    "G1_VST_mean",
    "G1_caudate_mean",
    "G1_putamen_mean",
]

fig, ax = plt.subplots(
    figsize=(7.0, 5.8)
)

xpos = np.arange(
    len(regions)
)

for group in [
    "HC",
    "SZ",
]:

    means = []
    sems = []

    for c in region_cols:

        x = df.loc[
            df["group"] == group,
            c
        ].to_numpy(
            dtype=float
        )

        means.append(
            np.mean(x)
        )

        sems.append(
            stats.sem(x)
        )

    ax.errorbar(
        xpos,
        means,
        yerr=sems,
        marker="o",
        capsize=4,
        linewidth=1.8,
        label=group
    )


ax.axhline(
    0,
    linewidth=1,
    linestyle="--",
    alpha=0.5
)

ax.set_xticks(
    xpos
)

ax.set_xticklabels(
    regions
)

ax.set_ylabel(
    "Mean individual G1 value"
)

ax.set_xlabel(
    "Striatal anatomical subdivision"
)

ax.set_title(
    "Individual G1 anatomical profile"
)

ax.legend(
    frameon=False
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
    "Figure_regional_G1_profile_HC_vs_SZ.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG_OUT /
    "Figure_regional_G1_profile_HC_vs_SZ.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# STEP 11. QC
# =============================================================================

print()
print("=" * 110)
print("STEP 11. FINAL QC")
print("=" * 110)

group_counts = (
    df["group"]
    .value_counts()
    .to_dict()
)

finite_metrics = bool(
    np.all(
        np.isfinite(
            df[
                metric_cols
            ].to_numpy(
                dtype=float
            )
        )
    )
)

# Expectation is not imposed as PASS criterion.
# We only report how many individuals show the paper-direction separation > 0.
hc_positive = int(
    np.sum(
        hc_sep > 0
    )
)

sz_positive = int(
    np.sum(
        sz_sep > 0
    )
)

qc = {
    "N_total":
        len(df),

    "N_HC":
        int(
            group_counts.get(
                "HC",
                0
            )
        ),

    "N_SZ":
        int(
            group_counts.get(
                "SZ",
                0
            )
        ),

    "common_voxels":
        2608,

    "VST_voxels":
        int(
            np.sum(
                labels2608 == "VST"
            )
        ),

    "Caudate_voxels":
        int(
            np.sum(
                labels2608 == "Caudate"
            )
        ),

    "Putamen_voxels":
        int(
            np.sum(
                labels2608 == "Putamen"
            )
        ),

    "global_orientation_sign":
        GLOBAL_SIGN,

    "all_metrics_finite":
        finite_metrics,

    "HC_with_positive_separation":
        hc_positive,

    "SZ_with_positive_separation":
        sz_positive,

    "primary_test":
        "Welch independent-samples t-test, two-sided",

    "primary_effect_direction":
        "HC minus SZ",

    "Welch_p":
        float(
            welch.pvalue
        ),

    "Hedges_g":
        hedges_g,

    "permutation_p":
        perm_p,

    "final_check":
        (
            "PASS"
            if (
                len(df) == 64
                and
                group_counts.get(
                    "HC",
                    0
                ) == 32
                and
                group_counts.get(
                    "SZ",
                    0
                ) == 32
                and
                finite_metrics
            )
            else
            "FAIL"
        ),
}

with open(
    QC_OUT /
    "FINAL_G1_ANATOMICAL_SEPARATION_QC.json",
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
print("PRIMARY OUTCOME")
print("-" * 90)

print(
    f"HC: "
    f"{np.mean(hc_sep):.6f} ± "
    f"{np.std(hc_sep, ddof=1):.6f}"
)

print(
    f"SZ: "
    f"{np.mean(sz_sep):.6f} ± "
    f"{np.std(sz_sep, ddof=1):.6f}"
)

print(
    f"HC - SZ = "
    f"{mean_diff:.6f}"
)

print(
    f"Bootstrap 95% CI = "
    f"[{ci_lo:.6f}, {ci_hi:.6f}]"
)

print(
    f"Welch t({welch_df:.3f}) = "
    f"{welch.statistic:.6f}"
)

print(
    f"Welch p = "
    f"{welch.pvalue:.8f}"
)

print(
    f"Hedges' g = "
    f"{hedges_g:.6f}"
)

print(
    f"Permutation p = "
    f"{perm_p:.8f}"
)

print(
    f"HC positive separation = "
    f"{hc_positive}/32"
)

print(
    f"SZ positive separation = "
    f"{sz_positive}/32"
)

print()
print("REGIONAL DESCRIPTIVES")
print("-" * 90)

print(
    regional_stats.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print()
print(
    "FINAL_CHECK =",
    qc["final_check"]
)

print()
print("OUTPUT TABLES")
print("-" * 90)

for p in [
    TABLE_OUT /
    "individual_G1_anatomical_separation.tsv",

    TABLE_OUT /
    "G1_anatomical_metrics_group_summary.tsv",

    TABLE_OUT /
    "PRIMARY_HC_vs_SZ_G1_separation_stats.tsv",

    TABLE_OUT /
    "regional_G1_means_HC_vs_SZ_DESCRIPTIVE.tsv",
]:

    print(p)

print()
print("OUTPUT FIGURES")
print("-" * 90)

for p in [
    FIG_OUT /
    "Figure_G1_separation_HC_vs_SZ.png",

    FIG_OUT /
    "Figure_regional_G1_profile_HC_vs_SZ.png",
]:

    print(p)
