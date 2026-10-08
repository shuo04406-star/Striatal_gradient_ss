#!/usr/bin/env bash
set -euo pipefail

: "${STRIATAL_ANALYSIS_ROOT:?Set STRIATAL_ANALYSIS_ROOT to the private analysis root}"
BASE="${STRIATAL_ANALYSIS_ROOT}"

FCROOT="${FCROOT:-$BASE/17_current64_375matched_fingerprints/04_uniform_z_for_normative_analysis}"
COV="${COV:-$BASE/20_current64_reward_stage_fc/05_covariate_adjusted/covariate_master_64.tsv}"
G1="${G1:-$BASE/34_G1_orientation_FINAL/01_canonical_reference/g1_voxel_table_with_bins_CURRENT.tsv}"
MAP="${MAP:-$BASE/19_current64_375reference_gradients/01_reference/row_mapping_2847_to_2608.npy}"

ANTI="${ANTI:-$BASE/24_stage_family_reanalysis_v2/tables/anticipation_ROIs_43.tsv}"
OUTCOME="${OUTCOME:-$BASE/24_stage_family_reanalysis_v2/tables/outcome_ROIs_19.tsv}"
EXCLUDED="${EXCLUDED:-$BASE/24_stage_family_reanalysis_v2/tables/midbrain_excluded_from_two_family_3.tsv}"

PARENT="${PARENT:-$BASE/24_normative_G1_FC_deviation_FIXED/CURRENTbin_parent_validation/02_subject_bin_metrics_G1_bin_current.tsv}"

OUT="${OUT:-$BASE/24_normative_G1_FC_deviation_FIXED/Bin10_subfamily_FINAL}"
mkdir -p "$OUT"

export FCROOT COV G1 MAP ANTI OUTCOME EXCLUDED PARENT OUT

python3 - <<'PY' 2>&1 | tee "$OUT/run.log"

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

FCROOT  = Path(os.environ["FCROOT"])
COVFILE = Path(os.environ["COV"])
G1FILE  = Path(os.environ["G1"])
MAPFILE = Path(os.environ["MAP"])

ANTI_FILE = Path(os.environ["ANTI"])
OUT_FILE  = Path(os.environ["OUTCOME"])
EXC_FILE  = Path(os.environ["EXCLUDED"])

PARENT = Path(os.environ["PARENT"])
OUT    = Path(os.environ["OUT"])

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 100)

print("="*110)
print("FINAL BIN10 SUBFAMILY DECOMPOSITION")
print("G1_bin_current | 285 Putamen voxels | 65-ROI normative profile")
print("="*110)


# =============================================================================
# 1. ROI LABELS
# =============================================================================

print("\n"+"="*110)
print("STEP 1. LOCK 65 ROI -> 14 SUBFAMILY LABELS")
print("="*110)

anti = pd.read_csv(ANTI_FILE, sep="\t")
outc = pd.read_csv(OUT_FILE, sep="\t")
exc  = pd.read_csv(EXC_FILE, sep="\t")


def prep(d, stage):

    d = d.copy()

    d["roi_index0"] = pd.to_numeric(
        d["roi_index0"],
        errors="raise"
    ).astype(int)

    if "subfamily" not in d.columns:
        if stage == "Not_assigned":
            d["subfamily"] = "Midbrain_pallidal"
        else:
            raise RuntimeError(
                f"{stage}: subfamily column missing"
            )

    if "brainnetome_label" not in d.columns:
        d["brainnetome_label"] = [
            f"ROI_{x:02d}"
            for x in d["roi_index0"]
        ]

    d["stage"] = stage

    return d[
        [
            "roi_index0",
            "brainnetome_label",
            "subfamily",
            "stage"
        ]
    ]


roi = pd.concat(
    [
        prep(anti, "Anticipation"),
        prep(outc, "Outcome"),
        prep(exc, "Not_assigned")
    ],
    ignore_index=True
).sort_values("roi_index0").reset_index(drop=True)


if len(roi) != 65:
    raise RuntimeError(
        f"Expected 65 ROI rows; got {len(roi)}"
    )

if set(roi.roi_index0) != set(range(65)):
    raise RuntimeError(
        "ROI indices do not exactly cover 0..64"
    )


manifest = (
    roi.groupby(
        ["subfamily","stage"],
        as_index=False
    )
    .agg(
        N_ROIs=("roi_index0","nunique")
    )
    .sort_values(
        ["stage","subfamily"]
    )
)

print(manifest.to_string(index=False))

manifest.to_csv(
    OUT/"01_subfamily_manifest.tsv",
    sep="\t",
    index=False
)


assigned = manifest[
    manifest.stage != "Not_assigned"
]

if assigned.subfamily.nunique() != 13:
    raise RuntimeError(
        f"Expected 13 Anticipation/Outcome assigned subfamilies; "
        f"found {assigned.subfamily.nunique()}"
    )

print()
print("13 ASSIGNED + 1 NOT-ASSIGNED SUBFAMILY QC: PASS")


# =============================================================================
# 2. COVARIATES
# =============================================================================

print("\n"+"="*110)
print("STEP 2. LOAD 64 SUBJECTS")
print("="*110)

cov = pd.read_csv(
    COVFILE,
    sep="\t"
).copy()

cov["subject"] = cov["subject"].astype(str)
cov["group"] = cov["group"].astype(str).str.upper()

cov["age"] = pd.to_numeric(
    cov["age"],
    errors="raise"
)

cov["meanFD"] = pd.to_numeric(
    cov["meanFD"],
    errors="raise"
)

cov["sex_num"] = (
    cov["sex"].astype(str).str.lower()
    .map({
        "female":0,
        "f":0,
        "male":1,
        "m":1
    })
)

if cov.sex_num.isna().any():
    raise RuntimeError("Bad sex coding")

cov["group_sz"] = (
    cov.group == "SZ"
).astype(int)

cov["age_c"] = (
    cov.age - cov.age.mean()
)

cov["sex_c"] = (
    cov.sex_num - cov.sex_num.mean()
)

cov["meanFD_c"] = (
    cov.meanFD - cov.meanFD.mean()
)

if len(cov) != 64:
    raise RuntimeError("Expected 64 subjects")

print(
    cov.groupby("group").subject.nunique()
)


# =============================================================================
# 3. CURRENT BIN10
# =============================================================================

print("\n"+"="*110)
print("STEP 3. LOCK CURRENT BIN10")
print("="*110)

g1 = pd.read_csv(
    G1FILE,
    sep="\t"
)

full_bin = pd.to_numeric(
    g1["G1_bin_current"],
    errors="raise"
).astype(int).to_numpy()

mapping = np.asarray(
    np.load(MAPFILE)
).reshape(-1).astype(int)

common_bin = full_bin[mapping]

bin10 = (
    common_bin == 10
)

print(
    "Bin10 common voxels =",
    int(bin10.sum())
)

if int(bin10.sum()) != 285:
    raise RuntimeError(
        f"Expected 285 CURRENT Bin10 voxels; "
        f"got {bin10.sum()}"
    )


# =============================================================================
# 4. SUBJECT BIN10 65-ROI FINGERPRINT
# =============================================================================

print("\n"+"="*110)
print("STEP 4. LOAD BIN10 65-ROI FC FINGERPRINT")
print("="*110)

fingerprints = {}

for sid in cov.subject:

    p = (
        FCROOT
        / f"{sid}_fingerprint_z_2608x65.npy"
    )

    if not p.exists():
        raise FileNotFoundError(p)

    a = np.asarray(
        np.load(p),
        dtype=float
    )

    if a.shape == (65,2608):
        a = a.T

    if a.shape != (2608,65):
        raise RuntimeError(
            f"{sid}: wrong FC shape {a.shape}"
        )

    fingerprints[sid] = (
        a[bin10,:].mean(axis=0)
    )

print(
    "Loaded subjects =",
    len(fingerprints)
)


# =============================================================================
# 5. HC NORMATIVE TEMPLATE + EXACT PEARSON-r DECOMPOSITION
#
# Pearson r across 65 ROIs:
#
#   r = sum_i z_subject(i) * z_template(i) / 64
#
# Therefore each ROI has an exact additive contribution to r.
# =============================================================================

print("\n"+"="*110)
print("STEP 5. EXACTLY DECOMPOSE BIN10 PROFILE SIMILARITY")
print("="*110)

hc = cov.loc[
    cov.group=="HC",
    "subject"
].tolist()

hc_mean = np.stack(
    [
        fingerprints[s]
        for s in hc
    ],
    axis=0
).mean(axis=0)


def decompose_r(x, template):

    x = np.asarray(x,float)
    t = np.asarray(template,float)

    zx = (
        (x-x.mean())
        / x.std(ddof=1)
    )

    zt = (
        (t-t.mean())
        / t.std(ddof=1)
    )

    contribution = (
        zx * zt / 64.0
    )

    direct = float(
        np.corrcoef(
            x,
            t
        )[0,1]
    )

    reconstructed = float(
        contribution.sum()
    )

    if abs(
        direct-reconstructed
    ) > 1e-10:
        raise RuntimeError(
            "Pearson-r decomposition failed"
        )

    return direct, contribution


subject_rows = []
roi_rows = []

for _,cr in cov.iterrows():

    sid = cr.subject

    x = fingerprints[sid]

    if cr.group == "HC":

        other_hc = [
            s for s in hc
            if s != sid
        ]

        template = np.stack(
            [
                fingerprints[s]
                for s in other_hc
            ],
            axis=0
        ).mean(axis=0)

        reference = "HC_LOO"

    else:

        template = hc_mean
        reference = "HC32"

    r,contribution = decompose_r(
        x,
        template
    )

    subject_rows.append({
        "subject":sid,
        "group":cr.group,
        "group_sz":cr.group_sz,
        "profile_r":r,
        "profile_z":
            np.arctanh(
                np.clip(
                    r,
                    -0.999999,
                    0.999999
                )
            ),
        "age_c":cr.age_c,
        "sex_c":cr.sex_c,
        "meanFD_c":cr.meanFD_c,
        "reference":reference
    })

    for j in range(65):

        rr = roi.iloc[j]

        roi_rows.append({
            "subject":sid,
            "group":cr.group,
            "group_sz":cr.group_sz,
            "age_c":cr.age_c,
            "sex_c":cr.sex_c,
            "meanFD_c":cr.meanFD_c,

            "roi_index0":j,
            "brainnetome_label":
                rr.brainnetome_label,
            "subfamily":
                rr.subfamily,
            "stage":
                rr.stage,

            "r_contribution":
                contribution[j]
        })


subject = pd.DataFrame(
    subject_rows
)

long = pd.DataFrame(
    roi_rows
)


# =============================================================================
# 6. REPRODUCE CURRENT-BIN PARENT BIN10
# =============================================================================

print("\n"+"="*110)
print("STEP 6. REPRODUCE CURRENT-BIN BIN10")
print("="*110)

parent = pd.read_csv(
    PARENT,
    sep="\t"
)

parent = (
    parent[parent.bin==10]
    [["subject","profile_r"]]
    .rename(
        columns={
            "profile_r":
                "parent_profile_r"
        }
    )
)

qc = subject.merge(
    parent,
    on="subject",
    how="left"
)

qc["absdiff"] = (
    qc.profile_r
    - qc.parent_profile_r
).abs()

maxdiff = qc.absdiff.max()

print(
    "Maximum profile-r difference =",
    f"{maxdiff:.12g}"
)

if maxdiff > 1e-10:
    raise RuntimeError(
        "CURRENT-bin parent effect "
        "was not exactly reproduced."
    )

print("CURRENT BIN10 REPRODUCTION: PASS")


# =============================================================================
# 7. FULL RAW-r EFFECT
#
# This is NOT replacing the parent Fisher-z inference.
# It exists because Pearson r can be decomposed exactly.
# =============================================================================

print("\n"+"="*110)
print("STEP 7. FULL BIN10 EFFECT ON DECOMPOSABLE r SCALE")
print("="*110)

m_total = smf.ols(
    "profile_r ~ "
    "group_sz + age_c + sex_c + meanFD_c",
    data=subject
).fit(
    cov_type="HC3"
)

total_beta = float(
    m_total.params["group_sz"]
)

total_se = float(
    m_total.bse["group_sz"]
)

total_p = float(
    m_total.pvalues["group_sz"]
)

total = pd.DataFrame([{
    "beta_SZ_minus_HC_profile_r":
        total_beta,
    "SE_HC3":
        total_se,
    "p_HC3":
        total_p,
    "HC_mean_r":
        subject.loc[
            subject.group=="HC",
            "profile_r"
        ].mean(),
    "SZ_mean_r":
        subject.loc[
            subject.group=="SZ",
            "profile_r"
        ].mean()
}])

print(
    total.to_string(
        index=False
    )
)

total.to_csv(
    OUT/"02_Bin10_full_effect_r_scale.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 8. SUM EXACT ROI CONTRIBUTIONS WITHIN EACH SUBFAMILY
# =============================================================================

print("\n"+"="*110)
print("STEP 8. 14-SUBFAMILY DRIVER ANALYSIS")
print("="*110)

sub_subject = (
    long.groupby(
        [
            "subject",
            "group",
            "group_sz",
            "age_c",
            "sex_c",
            "meanFD_c",
            "subfamily",
            "stage"
        ],
        as_index=False
    )
    .agg(
        contribution_sum=(
            "r_contribution",
            "sum"
        ),
        contribution_mean_per_ROI=(
            "r_contribution",
            "mean"
        ),
        n_ROIs=(
            "roi_index0",
            "nunique"
        )
    )
)

sub_rows = []

for (
    subfamily,
    stage
),ds in sub_subject.groupby(
    ["subfamily","stage"]
):

    m = smf.ols(
        "contribution_sum ~ "
        "group_sz + age_c + sex_c + meanFD_c",
        data=ds
    ).fit(
        cov_type="HC3"
    )

    beta = float(
        m.params["group_sz"]
    )

    se = float(
        m.bse["group_sz"]
    )

    p = float(
        m.pvalues["group_sz"]
    )

    nroi = int(
        ds.n_ROIs.iloc[0]
    )

    sub_rows.append({
        "subfamily":subfamily,
        "stage":stage,
        "N_ROIs":nroi,

        "HC_mean_contribution":
            ds.loc[
                ds.group=="HC",
                "contribution_sum"
            ].mean(),

        "SZ_mean_contribution":
            ds.loc[
                ds.group=="SZ",
                "contribution_sum"
            ].mean(),

        "beta_SZ_minus_HC":
            beta,

        "beta_per_ROI":
            beta / nroi,

        "SE_HC3":
            se,

        "p_raw":
            p,

        "CI95_low":
            beta - 1.96*se,

        "CI95_high":
            beta + 1.96*se
    })


res = pd.DataFrame(
    sub_rows
)


# FDR ONLY ACROSS THE 14 ASSIGNED SUBFAMILIES
res["q_FDR_13_assigned"] = np.nan

mask = (
    res.stage != "Not_assigned"
)

res.loc[
    mask,
    "q_FDR_13_assigned"
] = multipletests(
    res.loc[
        mask,
        "p_raw"
    ],
    method="fdr_bh"
)[1]


# Signed proportion of the total r-scale group effect.
#
# Since total effect is negative:
# positive % = contributes to lower similarity in SZ
# negative % = offsets/opposes the overall effect
res["signed_share_of_total_pct"] = (
    100
    * res["beta_SZ_minus_HC"]
    / total_beta
)


res = res.sort_values(
    "beta_SZ_minus_HC",
    ascending=True
).reset_index(drop=True)


res.to_csv(
    OUT/"03_Bin10_subfamily_drivers_FINAL.tsv",
    sep="\t",
    index=False
)


print()
print(
    "Negative beta = contributes to LOWER "
    "Bin10 normative-profile similarity in SZ."
)

print()

print(
    res[
        [
            "subfamily",
            "stage",
            "N_ROIs",
            "beta_SZ_minus_HC",
            "beta_per_ROI",
            "signed_share_of_total_pct",
            "p_raw",
            "q_FDR_13_assigned"
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# 9. EXACT SUM CHECK
# =============================================================================

print("\n"+"="*110)
print("STEP 9. EXACT ADDITIVE QC")
print("="*110)

sum_beta = float(
    res.beta_SZ_minus_HC.sum()
)

print(
    f"Whole profile beta       = {total_beta:+.12f}"
)

print(
    f"Sum of all subfamilies   = {sum_beta:+.12f}"
)

print(
    f"Absolute difference      = "
    f"{abs(total_beta-sum_beta):.3e}"
)

if abs(
    total_beta-sum_beta
) > 1e-9:

    raise RuntimeError(
        "Subfamily beta decomposition does not sum "
        "to whole-profile beta."
    )

print(
    "ADDITIVE DECOMPOSITION QC: PASS"
)


# =============================================================================
# 10. STAGE-SUM CHECK — DESCRIPTION ONLY
# =============================================================================

print("\n"+"="*110)
print("STEP 10. STAGE-LEVEL CONTRIBUTION SUMMARY")
print("="*110)

stage_sum = (
    sub_subject.groupby(
        [
            "subject",
            "group",
            "group_sz",
            "age_c",
            "sex_c",
            "meanFD_c",
            "stage"
        ],
        as_index=False
    )
    .agg(
        contribution_sum=(
            "contribution_sum",
            "sum"
        )
    )
)

stage_rows=[]

for stage,ds in stage_sum.groupby(
    "stage"
):

    m=smf.ols(
        "contribution_sum ~ "
        "group_sz + age_c + sex_c + meanFD_c",
        data=ds
    ).fit(
        cov_type="HC3"
    )

    stage_rows.append({
        "stage":stage,
        "beta_SZ_minus_HC":
            float(
                m.params["group_sz"]
            ),
        "p_HC3":
            float(
                m.pvalues["group_sz"]
            )
    })


stage_res=pd.DataFrame(
    stage_rows
)

stage_res.to_csv(
    OUT/"04_stage_contribution_summary.tsv",
    sep="\t",
    index=False
)

print(
    stage_res.to_string(
        index=False
    )
)


# =============================================================================
# 11. ROI DRIVER RANKING — EXPLORATORY
# =============================================================================

print("\n"+"="*110)
print("STEP 11. ROI DRIVER RANKING — EXPLORATORY")
print("="*110)

roi_rows=[]

for (
    idx,
    label,
    sf,
    stage
),ds in long.groupby(
    [
        "roi_index0",
        "brainnetome_label",
        "subfamily",
        "stage"
    ]
):

    m=smf.ols(
        "r_contribution ~ "
        "group_sz + age_c + sex_c + meanFD_c",
        data=ds
    ).fit(
        cov_type="HC3"
    )

    roi_rows.append({
        "roi_index0":idx,
        "brainnetome_label":label,
        "subfamily":sf,
        "stage":stage,
        "beta_SZ_minus_HC":
            float(
                m.params["group_sz"]
            ),
        "p_raw":
            float(
                m.pvalues["group_sz"]
            )
    })


roi_res=pd.DataFrame(
    roi_rows
)

roi_res["q_FDR_65"] = multipletests(
    roi_res.p_raw,
    method="fdr_bh"
)[1]

roi_res = roi_res.sort_values(
    "beta_SZ_minus_HC",
    ascending=True
)

roi_res.to_csv(
    OUT/"05_Bin10_ROI_driver_ranking_EXPLORATORY.tsv",
    sep="\t",
    index=False
)

print()
print("TOP 15 NEGATIVE ROI DRIVERS:")

print(
    roi_res[
        [
            "brainnetome_label",
            "subfamily",
            "stage",
            "beta_SZ_minus_HC",
            "p_raw",
            "q_FDR_65"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# =============================================================================
# FINAL
# =============================================================================

print("\n"+"="*110)
print("FINAL — SUBFAMILY RANKING")
print("="*110)

print()

print(
    res[
        res.stage!="Not_assigned"
    ][
        [
            "subfamily",
            "stage",
            "N_ROIs",
            "beta_SZ_minus_HC",
            "signed_share_of_total_pct",
            "p_raw",
            "q_FDR_13_assigned"
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "Interpretation:"
)

print(
    "More negative beta = stronger contribution "
    "to the reduced Bin10 normative profile similarity in SZ."
)

print(
    "q_FDR_13_assigned controls multiplicity across the 14 predefined "
    "reward subfamilies."
)

print(
    "The parent inferential result remains the previously "
    "validated Fisher-z Bin10 effect; this table decomposes "
    "the Pearson-r effect exactly."
)

print()
print("OUTPUT:",OUT)

PY

STATUS=${PIPESTATUS[0]}

if [ "$STATUS" -ne 0 ]; then
    echo
    echo "ANALYSIS STOPPED WITH ERROR."
    echo "SSH REMAINS OPEN."
    echo "LOG: $OUT/run.log"
    exit "$STATUS"
fi

echo
echo "======================================================================"
echo "BIN10 SUBFAMILY ANALYSIS COMPLETE"
echo "======================================================================"

echo
echo "FULL BIN10 r-SCALE EFFECT:"
column -t -s $'\t' \
"$OUT/02_Bin10_full_effect_r_scale.tsv"

echo
echo "14 SUBFAMILY DRIVERS:"
column -t -s $'\t' \
"$OUT/03_Bin10_subfamily_drivers_FINAL.tsv"

echo
echo "TOP 15 ROI DRIVERS:"
head -n 16 \
"$OUT/05_Bin10_ROI_driver_ranking_EXPLORATORY.tsv" \
| column -t -s $'\t'

echo
echo "OUTPUT:"
echo "$OUT"
