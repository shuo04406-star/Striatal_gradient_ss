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
    "04_regional_ordering"
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

REF_FILE = (
    ROOT /
    "19_current64_375reference_gradients" /
    "01_reference" /
    "375_group_gradients_mapped_2608x10.npy"
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


def fisher_table(
    df,
    outcome
):
    """
    Table rows:
        HC, SZ
    columns:
        ordering yes, ordering no
    """

    hc_yes = int(
        np.sum(
            (
                df["group"] == "HC"
            )
            &
            (
                df[outcome] == 1
            )
        )
    )

    hc_no = int(
        np.sum(
            (
                df["group"] == "HC"
            )
            &
            (
                df[outcome] == 0
            )
        )
    )

    sz_yes = int(
        np.sum(
            (
                df["group"] == "SZ"
            )
            &
            (
                df[outcome] == 1
            )
        )
    )

    sz_no = int(
        np.sum(
            (
                df["group"] == "SZ"
            )
            &
            (
                df[outcome] == 0
            )
        )
    )

    table = np.array(
        [
            [
                hc_yes,
                hc_no,
            ],
            [
                sz_yes,
                sz_no,
            ],
        ],
        dtype=int
    )

    odds_ratio, p = stats.fisher_exact(
        table,
        alternative="two-sided"
    )

    return {
        "HC_yes":
            hc_yes,

        "HC_no":
            hc_no,

        "HC_percent_yes":
            100.0 * hc_yes / 32,

        "SZ_yes":
            sz_yes,

        "SZ_no":
            sz_no,

        "SZ_percent_yes":
            100.0 * sz_yes / 32,

        "odds_ratio_HC_vs_SZ":
            float(odds_ratio),

        "Fisher_exact_p_two_sided":
            float(p),
    }


# =============================================================================
# STEP 1. INPUT CHECK
# =============================================================================

print("=" * 110)
print("STEP 1. INPUT CHECK")
print("=" * 110)

for p in [
    IND_ROOT,
    REF_FILE,
    ROW_MAP_FILE,
    COORD_FILE,
    VST_FILE,
    CAUDATE_FILE,
    PUTAMEN_FILE,
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
# STEP 2. SUBJECTS
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

subjects = hc + sz

group_map = {
    s: "HC"
    for s in hc
}

group_map.update({
    s: "SZ"
    for s in sz
})

print("HC =", len(hc))
print("SZ =", len(sz))


# =============================================================================
# STEP 3. BUILD 2608 LABELS
# =============================================================================

print()
print("=" * 110)
print("STEP 3. COMMON2608 ANATOMICAL LABELS")
print("=" * 110)

coord = pd.read_csv(
    COORD_FILE,
    sep="\t"
)

ijk = coord[
    [
        "i",
        "j",
        "k",
    ]
].to_numpy(
    dtype=int
)

if ijk.shape != (
    2847,
    3
):
    raise RuntimeError(
        ijk.shape
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
        ijk[:, 0],
        ijk[:, 1],
        ijk[:, 2]
    ]


vst = sample_mask(
    VST_FILE
)

cau = sample_mask(
    CAUDATE_FILE
)

put = sample_mask(
    PUTAMEN_FILE
)

labels2847 = np.full(
    2847,
    "",
    dtype=object
)

labels2847[
    vst
] = "VST"

labels2847[
    cau
] = "Caudate"

labels2847[
    put
] = "Putamen"

labels = labels2847[
    row_map
]

counts = {
    "VST":
        int(
            np.sum(
                labels == "VST"
            )
        ),

    "Caudate":
        int(
            np.sum(
                labels == "Caudate"
            )
        ),

    "Putamen":
        int(
            np.sum(
                labels == "Putamen"
            )
        ),
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
# STEP 4. GLOBAL PAPER ORIENTATION
# =============================================================================

print()
print("=" * 110)
print("STEP 4. GLOBAL G1 ORIENTATION")
print("=" * 110)

ref = np.load(
    REF_FILE,
    allow_pickle=False
).astype(
    float
)

if ref.shape != (
    2608,
    10
):
    raise RuntimeError(
        ref.shape
    )

ref_g1 = ref[
    :,
    0
]

ref_vst = float(
    np.mean(
        ref_g1[
            labels == "VST"
        ]
    )
)

ref_cau = float(
    np.mean(
        ref_g1[
            labels == "Caudate"
        ]
    )
)

ref_put = float(
    np.mean(
        ref_g1[
            labels == "Putamen"
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

print(
    "GLOBAL_SIGN =",
    GLOBAL_SIGN
)

print(
    "Convention:"
)

print(
    "lower G1 = VST/Caudate; "
    "higher G1 = Putamen"
)


# =============================================================================
# STEP 5. INDIVIDUAL ORDERING
# =============================================================================

print()
print("=" * 110)
print("STEP 5. INDIVIDUAL REGIONAL ORDERING")
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
        float
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
            f"{subject}: nonfinite values."
        )

    g1 = (
        GLOBAL_SIGN
        *
        G[
            :,
            0
        ]
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

    low_mean = float(
        (
            vst_mean
            +
            cau_mean
        )
        /
        2.0
    )

    # Original strict ordering
    strict = int(
        (
            vst_mean
            <
            cau_mean
        )
        and
        (
            cau_mean
            <
            put_mean
        )
    )

    # More theoretically stable two-end ordering
    robust = int(
        low_mean
        <
        put_mean
    )

    # Useful QC subconditions
    vst_lt_cau = int(
        vst_mean
        <
        cau_mean
    )

    cau_lt_put = int(
        cau_mean
        <
        put_mean
    )

    vst_lt_put = int(
        vst_mean
        <
        put_mean
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

        "VST_lt_Caudate":
            vst_lt_cau,

        "Caudate_lt_Putamen":
            cau_lt_put,

        "VST_lt_Putamen":
            vst_lt_put,

        "strict_VST_lt_Caudate_lt_Putamen":
            strict,

        "robust_meanVSTCaudate_lt_Putamen":
            robust,

        # margins only for QC / interpretation
        "margin_Caudate_minus_VST":
            (
                cau_mean
                -
                vst_mean
            ),

        "margin_Putamen_minus_Caudate":
            (
                put_mean
                -
                cau_mean
            ),

        "margin_Putamen_minus_low":
            (
                put_mean
                -
                low_mean
            ),
    })

    print(
        f"[{i:02d}/64] "
        f"{subject} {group_map[subject]} | "
        f"VST={vst_mean:+.5f} | "
        f"CAU={cau_mean:+.5f} | "
        f"PUT={put_mean:+.5f} | "
        f"strict={strict} | "
        f"robust={robust}"
    )


df = pd.DataFrame(
    rows
)

df.to_csv(
    TABLE_OUT /
    "individual_G1_regional_ordering.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# STEP 6. GROUP COUNTS
# =============================================================================

print()
print("=" * 110)
print("STEP 6. GROUP ORDERING COUNTS")
print("=" * 110)

summary_rows = []

outcomes = [
    (
        "strict_VST_lt_Caudate_lt_Putamen",
        "Strict: VST < Caudate < Putamen"
    ),
    (
        "robust_meanVSTCaudate_lt_Putamen",
        "Robust: mean(VST,Caudate) < Putamen"
    ),
    (
        "VST_lt_Caudate",
        "VST < Caudate"
    ),
    (
        "Caudate_lt_Putamen",
        "Caudate < Putamen"
    ),
    (
        "VST_lt_Putamen",
        "VST < Putamen"
    ),
]

for outcome, label in outcomes:

    for group in [
        "HC",
        "SZ",
    ]:

        d = df[
            df[
                "group"
            ]
            ==
            group
        ]

        yes = int(
            d[
                outcome
            ].sum()
        )

        summary_rows.append({
            "outcome":
                outcome,

            "label":
                label,

            "group":
                group,

            "N":
                len(d),

            "yes_n":
                yes,

            "no_n":
                int(
                    len(d)
                    -
                    yes
                ),

            "yes_percent":
                float(
                    100.0
                    *
                    yes
                    /
                    len(d)
                ),
        })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    TABLE_OUT /
    "G1_regional_ordering_group_counts.tsv",
    sep="\t",
    index=False
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# =============================================================================
# STEP 7. FISHER EXACT
# =============================================================================

print()
print("=" * 110)
print("STEP 7. FISHER EXACT TESTS")
print("=" * 110)

test_rows = []

for outcome, label in outcomes:

    result = fisher_table(
        df,
        outcome
    )

    test_rows.append({
        "outcome":
            outcome,

        "label":
            label,

        **result,
    })


tests = pd.DataFrame(
    test_rows
)

# FDR is included only across the 5 ordering tests as secondary bookkeeping.
# The strict ordering test remains the manuscript-replication test.
pvals = tests[
    "Fisher_exact_p_two_sided"
].to_numpy(
    dtype=float
)

order = np.argsort(
    pvals
)

ranked = pvals[
    order
]

m = len(
    pvals
)

q_ranked = np.minimum.accumulate(
    (
        ranked
        *
        m
        /
        np.arange(
            1,
            m + 1
        )
    )[::-1]
)[::-1]

q_ranked = np.minimum(
    q_ranked,
    1.0
)

qvals = np.empty_like(
    q_ranked
)

qvals[
    order
] = q_ranked

tests[
    "FDR_BH_q"
] = qvals

tests.to_csv(
    TABLE_OUT /
    "G1_regional_ordering_Fisher_tests.tsv",
    sep="\t",
    index=False
)

print(
    tests.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# =============================================================================
# STEP 8. PRIMARY RESULTS
# =============================================================================

strict_row = tests[
    tests[
        "outcome"
    ]
    ==
    "strict_VST_lt_Caudate_lt_Putamen"
].iloc[0]

robust_row = tests[
    tests[
        "outcome"
    ]
    ==
    "robust_meanVSTCaudate_lt_Putamen"
].iloc[0]


# =============================================================================
# STEP 9. FIGURE 1 — STRICT ORDERING PERCENTAGE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(6.2, 5.5)
)

groups = [
    "HC",
    "SZ",
]

percents = [
    float(
        strict_row[
            "HC_percent_yes"
        ]
    ),
    float(
        strict_row[
            "SZ_percent_yes"
        ]
    ),
]

bars = ax.bar(
    groups,
    percents,
    width=0.55
)

for bar, pct, n in zip(
    bars,
    percents,
    [
        int(
            strict_row[
                "HC_yes"
            ]
        ),
        int(
            strict_row[
                "SZ_yes"
            ]
        ),
    ]
):

    ax.text(
        bar.get_x()
        +
        bar.get_width() / 2,
        pct + 2,
        f"{n}/32\n({pct:.1f}%)",
        ha="center",
        va="bottom"
    )


ax.set_ylim(
    0,
    100
)

ax.set_ylabel(
    "Participants meeting strict ordering (%)"
)

ax.set_title(
    "Individual G1 regional ordering"
)

ax.text(
    0.98,
    0.98,
    (
        "VST < Caudate < Putamen\n"
        f"Fisher p = "
        f"{strict_row['Fisher_exact_p_two_sided']:.4g}"
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

fig.tight_layout()

fig.savefig(
    FIG_OUT /
    "Figure_strict_G1_regional_ordering_HC_vs_SZ.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG_OUT /
    "Figure_strict_G1_regional_ordering_HC_vs_SZ.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# STEP 10. FIGURE 2 — INDIVIDUAL REGIONAL PROFILES
# =============================================================================

fig, ax = plt.subplots(
    figsize=(7.5, 6.0)
)

x = np.array(
    [
        0,
        1,
        2,
    ]
)

for _, r in df.iterrows():

    y = [
        r[
            "G1_VST_mean"
        ],
        r[
            "G1_caudate_mean"
        ],
        r[
            "G1_putamen_mean"
        ],
    ]

    ax.plot(
        x,
        y,
        linewidth=0.7,
        alpha=0.18
    )


# group means
for group in [
    "HC",
    "SZ",
]:

    d = df[
        df[
            "group"
        ]
        ==
        group
    ]

    means = [
        d[
            "G1_VST_mean"
        ].mean(),

        d[
            "G1_caudate_mean"
        ].mean(),

        d[
            "G1_putamen_mean"
        ].mean(),
    ]

    ax.plot(
        x,
        means,
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=group
    )


ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    alpha=0.5
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        "VST",
        "Caudate",
        "Putamen",
    ]
)

ax.set_ylabel(
    "Mean individual G1"
)

ax.set_title(
    "Individual striatal G1 regional profiles"
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
    "Figure_individual_G1_regional_profiles.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG_OUT /
    "Figure_individual_G1_regional_profiles.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# STEP 11. QC
# =============================================================================

robust_total = int(
    df[
        "robust_meanVSTCaudate_lt_Putamen"
    ].sum()
)

strict_total = int(
    df[
        "strict_VST_lt_Caudate_lt_Putamen"
    ].sum()
)

finite = bool(
    np.all(
        np.isfinite(
            df[
                [
                    "G1_VST_mean",
                    "G1_caudate_mean",
                    "G1_putamen_mean",
                ]
            ].to_numpy(
                dtype=float
            )
        )
    )
)

final_pass = bool(
    len(df) == 64
    and
    df[
        "subject"
    ].nunique() == 64
    and
    finite
)


qc = {
    "N":
        64,

    "HC":
        32,

    "SZ":
        32,

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

    "strict_definition":
        "VST < Caudate < Putamen",

    "robust_definition":
        "mean(VST,Caudate) < Putamen",

    "strict_total_yes":
        strict_total,

    "robust_total_yes":
        robust_total,

    "global_orientation_sign":
        GLOBAL_SIGN,

    "all_values_finite":
        finite,

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
    "FINAL_G1_REGIONAL_ORDERING_QC.json",
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
print("STRICT ORDERING: VST < CAUDATE < PUTAMEN")
print("-" * 100)

print(
    f"HC = "
    f"{int(strict_row['HC_yes'])}/32 "
    f"({strict_row['HC_percent_yes']:.2f}%)"
)

print(
    f"SZ = "
    f"{int(strict_row['SZ_yes'])}/32 "
    f"({strict_row['SZ_percent_yes']:.2f}%)"
)

print(
    "Odds ratio HC vs SZ =",
    strict_row[
        "odds_ratio_HC_vs_SZ"
    ]
)

print(
    "Fisher exact p =",
    strict_row[
        "Fisher_exact_p_two_sided"
    ]
)


print()
print("ROBUST ORDERING: mean(VST, CAUDATE) < PUTAMEN")
print("-" * 100)

print(
    f"HC = "
    f"{int(robust_row['HC_yes'])}/32 "
    f"({robust_row['HC_percent_yes']:.2f}%)"
)

print(
    f"SZ = "
    f"{int(robust_row['SZ_yes'])}/32 "
    f"({robust_row['SZ_percent_yes']:.2f}%)"
)

print(
    "Fisher exact p =",
    robust_row[
        "Fisher_exact_p_two_sided"
    ]
)


print()
print("SUBORDERINGS")
print("-" * 100)

for outcome in [
    "VST_lt_Caudate",
    "Caudate_lt_Putamen",
    "VST_lt_Putamen",
]:

    r = tests[
        tests[
            "outcome"
        ]
        ==
        outcome
    ].iloc[0]

    print(
        f"{r['label']}: "
        f"HC {int(r['HC_yes'])}/32 "
        f"vs SZ {int(r['SZ_yes'])}/32 | "
        f"Fisher p={r['Fisher_exact_p_two_sided']:.6f}"
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
    "individual_G1_regional_ordering.tsv",

    TABLE_OUT /
    "G1_regional_ordering_group_counts.tsv",

    TABLE_OUT /
    "G1_regional_ordering_Fisher_tests.tsv",
]:

    print(p)


print()
print("FIGURES")
print("-" * 100)

print(
    FIG_OUT /
    "Figure_strict_G1_regional_ordering_HC_vs_SZ.png"
)

print(
    FIG_OUT /
    "Figure_individual_G1_regional_profiles.png"
)
