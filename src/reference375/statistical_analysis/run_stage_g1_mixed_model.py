#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import re
import warnings

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


# =============================================================================
# 0. PATHS
# =============================================================================

BASE_VALUE = os.environ.get("STRIATAL_ANALYSIS_ROOT")
if not BASE_VALUE:
    raise RuntimeError("Set STRIATAL_ANALYSIS_ROOT to the private analysis root")
BASE = Path(BASE_VALUE).resolve()

PREV = BASE / "24_stage_family_reanalysis_v2"

FP_ROOT = BASE / "06_fingerprints"

OUT = BASE / "25_subject_level_stage_g1_model1"
OUT.mkdir(parents=True, exist_ok=True)

TAB = OUT / "tables"
QC  = OUT / "qc"

TAB.mkdir(exist_ok=True)
QC.mkdir(exist_ok=True)

META_FILE = (
    PREV /
    "tables/roi_metadata_stage_subfamily_v2.tsv"
)

BIN_FILE = (
    PREV /
    "tables/voxel_G1_and_10bin.tsv"
)

GROUP_FC_FILE = (
    BASE /
    "06_fingerprints/group_mean_voxel_by_roi_fisherz.npy"
)

EXPECTED_N_SUBJECTS = 375
EXPECTED_N_VOX = 2847
EXPECTED_N_ROI = 65


# =============================================================================
# 1. LOAD FROZEN ROI DEFINITIONS
# =============================================================================

if not META_FILE.exists():
    raise FileNotFoundError(META_FILE)

meta = pd.read_csv(
    META_FILE,
    sep="\t"
)

anti_idx = (
    meta.loc[
        meta["stage"].eq("Anticipation"),
        "roi_index0"
    ]
    .astype(int)
    .to_numpy()
)

out_idx = (
    meta.loc[
        meta["stage"].eq("Outcome"),
        "roi_index0"
    ]
    .astype(int)
    .to_numpy()
)

excluded_idx = (
    meta.loc[
        meta["include_two_family"].eq(0),
        "roi_index0"
    ]
    .astype(int)
    .to_numpy()
)

assert len(anti_idx) == 43
assert len(out_idx) == 19

assert set(excluded_idx.tolist()) == {62, 63, 64}

assert len(
    set(anti_idx.tolist()) &
    set(out_idx.tolist())
) == 0


print("=" * 100)
print("FROZEN STAGE DEFINITION")
print("=" * 100)

print(
    f"Anticipation = {len(anti_idx)} ROIs"
)

print(
    f"Outcome      = {len(out_idx)} ROIs"
)

print(
    "Excluded from stage = "
    "ROI63 VTA, ROI64 SNr, ROI65 VP/VeP"
)


# =============================================================================
# 2. LOAD FROZEN G1 BINS
# =============================================================================

if not BIN_FILE.exists():
    raise FileNotFoundError(BIN_FILE)

bin_df = pd.read_csv(
    BIN_FILE,
    sep="\t"
)

if len(bin_df) != EXPECTED_N_VOX:
    raise RuntimeError(
        f"Unexpected voxel count in bin file: "
        f"{len(bin_df)}"
    )

g1_bins = (
    bin_df["G1_bin"]
    .to_numpy()
)

if np.isnan(g1_bins).any():
    raise RuntimeError(
        "NaN found in G1 bin assignment."
    )

g1_bins = g1_bins.astype(int)

assert set(np.unique(g1_bins)) == set(range(1, 11))

print("\n" + "=" * 100)
print("G1 BIN COUNTS")
print("=" * 100)

for b in range(1, 11):
    print(
        f"Bin {b:2d}: "
        f"{np.sum(g1_bins == b)} voxels"
    )


# =============================================================================
# 3. DISCOVER INDIVIDUAL 2847 x 65 FINGERPRINTS
# =============================================================================

subject_pattern = re.compile(
    r"(sub-[A-Za-z0-9]+)",
    flags=re.IGNORECASE
)


def inspect_shape(p):
    try:
        arr = np.load(
            p,
            mmap_mode="r"
        )
        return tuple(arr.shape)
    except Exception:
        return None


def candidate_score(p):
    s = str(p).lower()
    name = p.name.lower()

    score = 0

    if "fisherz" in name:
        score += 20
    elif "fisher" in name:
        score += 15

    if "voxel_by_roi" in name:
        score += 15

    if "fingerprint" in name:
        score += 10

    if "fc" in name:
        score += 4

    if "group" in s:
        score -= 100

    if "mean" in name:
        score -= 50

    return score


candidates = []

print("\n" + "=" * 100)
print("SEARCHING INDIVIDUAL FINGERPRINTS")
print("=" * 100)

for p in FP_ROOT.rglob("*.npy"):

    m = subject_pattern.search(
        str(p)
    )

    if m is None:
        continue

    shp = inspect_shape(p)

    if shp not in [
        (EXPECTED_N_VOX, EXPECTED_N_ROI),
        (EXPECTED_N_ROI, EXPECTED_N_VOX)
    ]:
        continue

    sub = m.group(1).lower()

    candidates.append({
        "subject": sub,
        "file": str(p),
        "shape": str(shp),
        "score": candidate_score(p),
    })


cand_df = pd.DataFrame(candidates)

if len(cand_df) == 0:
    raise RuntimeError(
        "\nNo individual 2847 x 65 fingerprint "
        f"files found under:\n{FP_ROOT}"
    )


cand_df.to_csv(
    QC / "all_individual_fingerprint_candidates.tsv",
    sep="\t",
    index=False
)


# Choose highest-scoring file for each subject.
cand_df = cand_df.sort_values(
    ["subject", "score", "file"],
    ascending=[True, False, True]
)

selected = (
    cand_df
    .groupby("subject", as_index=False)
    .first()
)

selected = selected.sort_values(
    "subject"
).reset_index(drop=True)

selected.to_csv(
    QC / "selected_individual_fingerprints.tsv",
    sep="\t",
    index=False
)


print(
    "Unique subjects discovered:",
    len(selected)
)

print("\nFirst 10 selected files:")

for _, r in selected.head(10).iterrows():
    print(
        r["subject"],
        "->",
        r["file"]
    )


# Do NOT silently analyse a different sample.
if len(selected) != EXPECTED_N_SUBJECTS:
    raise RuntimeError(
        "\nSTOP: expected exactly "
        f"{EXPECTED_N_SUBJECTS} subjects, "
        f"but discovered {len(selected)}.\n"
        "Check:\n"
        f"{QC / 'selected_individual_fingerprints.tsv'}"
    )


# =============================================================================
# 4. BUILD SUBJECT x STAGE x G1 LONG TABLE
# =============================================================================

long_rows = []

# Also reconstruct the group mean from all selected subjects
# to verify that we selected the same frozen 375 fingerprints.
sum_fc = np.zeros(
    (EXPECTED_N_VOX, EXPECTED_N_ROI),
    dtype=np.float64
)

count_fc = np.zeros(
    (EXPECTED_N_VOX, EXPECTED_N_ROI),
    dtype=np.int32
)


for k, row in selected.iterrows():

    sub = row["subject"]
    p = Path(row["file"])

    arr = np.load(p).astype(float)

    if arr.shape == (
        EXPECTED_N_ROI,
        EXPECTED_N_VOX
    ):
        arr = arr.T

    if arr.shape != (
        EXPECTED_N_VOX,
        EXPECTED_N_ROI
    ):
        raise RuntimeError(
            f"{sub}: unexpected shape {arr.shape}"
        )

    finite = np.isfinite(arr)

    sum_fc[finite] += arr[finite]
    count_fc[finite] += 1

    # ------------------------------------------------------------
    # 10 bins x 65 ROIs for this subject
    # ------------------------------------------------------------

    bin_roi = np.full(
        (10, EXPECTED_N_ROI),
        np.nan,
        dtype=float
    )

    for b in range(1, 11):

        vox = (
            g1_bins == b
        )

        bin_roi[b - 1, :] = np.nanmean(
            arr[vox, :],
            axis=0
        )

    # ------------------------------------------------------------
    # Stage means
    #
    # IMPORTANT:
    # 63-65 are not present in either index set.
    # ------------------------------------------------------------

    anti_profile = np.nanmean(
        bin_roi[:, anti_idx],
        axis=1
    )

    outcome_profile = np.nanmean(
        bin_roi[:, out_idx],
        axis=1
    )

    for b in range(1, 11):

        long_rows.append({
            "Subject": sub,
            "G1_bin": b,
            "G1c": b - 5.5,
            "Stage": "Outcome",
            "FC": outcome_profile[b - 1],
        })

        long_rows.append({
            "Subject": sub,
            "G1_bin": b,
            "G1c": b - 5.5,
            "Stage": "Anticipation",
            "FC": anti_profile[b - 1],
        })

    if (
        (k + 1) % 25 == 0
        or
        (k + 1) == len(selected)
    ):
        print(
            f"[{k + 1:03d}/{len(selected)}] "
            f"{sub}"
        )


long_df = pd.DataFrame(
    long_rows
)

long_df.to_csv(
    TAB / "subject_stage_G1_long.tsv",
    sep="\t",
    index=False
)


print("\n" + "=" * 100)
print("LONG TABLE")
print("=" * 100)

print("Rows    :", len(long_df))
print(
    "Subjects:",
    long_df["Subject"].nunique()
)
print(
    "Stages  :",
    sorted(long_df["Stage"].unique())
)
print(
    "G1 bins :",
    sorted(long_df["G1_bin"].unique())
)

expected_rows = (
    EXPECTED_N_SUBJECTS *
    10 *
    2
)

if len(long_df) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} rows, "
        f"got {len(long_df)}."
    )

if long_df["FC"].isna().any():
    raise RuntimeError(
        "NaN FC values detected in long table."
    )


# =============================================================================
# 5. QC: RECONSTRUCT GROUP MEAN
# =============================================================================

individual_group_mean = np.divide(
    sum_fc,
    count_fc,
    out=np.full_like(
        sum_fc,
        np.nan,
        dtype=float
    ),
    where=count_fc > 0
)

np.save(
    QC / "reconstructed_375_group_mean.npy",
    individual_group_mean
)


qc_rows = []

if GROUP_FC_FILE.exists():

    frozen_group = np.load(
        GROUP_FC_FILE
    ).astype(float)

    if frozen_group.shape == (
        EXPECTED_N_ROI,
        EXPECTED_N_VOX
    ):
        frozen_group = frozen_group.T

    if frozen_group.shape != (
        EXPECTED_N_VOX,
        EXPECTED_N_ROI
    ):
        raise RuntimeError(
            "Frozen group FC has unexpected shape: "
            f"{frozen_group.shape}"
        )

    valid = (
        np.isfinite(frozen_group) &
        np.isfinite(individual_group_mean)
    )

    r = np.corrcoef(
        frozen_group[valid].ravel(),
        individual_group_mean[valid].ravel()
    )[0, 1]

    rmse = np.sqrt(
        np.mean(
            (
                frozen_group[valid] -
                individual_group_mean[valid]
            ) ** 2
        )
    )

    max_abs = np.max(
        np.abs(
            frozen_group[valid] -
            individual_group_mean[valid]
        )
    )

    qc_rows.append({
        "n_subjects": len(selected),
        "correlation_with_frozen_group_mean": r,
        "RMSE": rmse,
        "max_absolute_difference": max_abs,
    })

    print("\n" + "=" * 100)
    print("GROUP-MEAN RECONSTRUCTION QC")
    print("=" * 100)

    print(
        "Correlation:",
        f"{r:.10f}"
    )

    print(
        "RMSE       :",
        f"{rmse:.10g}"
    )

    print(
        "Max abs diff:",
        f"{max_abs:.10g}"
    )

    if r < 0.99:
        raise RuntimeError(
            "\nSTOP: reconstructed individual group mean "
            "does not sufficiently match the frozen "
            "group mean fingerprint."
        )


pd.DataFrame(qc_rows).to_csv(
    QC / "group_mean_reconstruction_QC.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 6. DESCRIPTIVE GROUP PROFILE
# =============================================================================

desc = (
    long_df
    .groupby(
        ["Stage", "G1_bin"],
        as_index=False
    )
    .agg(
        mean_FC=("FC", "mean"),
        SD=("FC", "std"),
        N=("FC", "size")
    )
)

desc["SEM"] = (
    desc["SD"] /
    np.sqrt(desc["N"])
)

desc.to_csv(
    TAB / "subject_level_stage_G1_descriptives.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 7. MODEL 1
#
# FC ~ Stage * G1c + (1 | Subject)
#
# Outcome is reference stage.
#
# Therefore:
#
# Intercept
#   = Outcome FC at middle of G1
#
# G1c
#   = Outcome G1 slope
#
# Stage[T.Anticipation]
#   = Anticipation - Outcome difference at G1 midpoint
#
# Stage[T.Anticipation]:G1c
#   = Anticipation slope - Outcome slope
#
# THIS INTERACTION IS THE PRIMARY TEST.
# =============================================================================

formula = (
    'FC ~ '
    'C(Stage, Treatment(reference="Outcome")) '
    '* G1c'
)

print("\n" + "=" * 100)
print("MODEL 1")
print("=" * 100)

print(
    "FC ~ Stage * G1c + (1 | Subject)"
)

print(
    "Reference stage = Outcome"
)

print(
    "G1c = G1_bin - 5.5"
)

print(
    "N subjects =",
    long_df["Subject"].nunique()
)

print(
    "N observations =",
    len(long_df)
)


model = smf.mixedlm(
    formula,
    data=long_df,
    groups=long_df["Subject"],
    re_formula="1"
)


# Try stable optimizers in order.
fit = None
fit_method = None
last_error = None

for method in [
    "lbfgs",
    "bfgs",
    "powell"
]:

    try:

        candidate = model.fit(
            reml=False,
            method=method,
            maxiter=3000,
            disp=False
        )

        fit = candidate
        fit_method = method
        break

    except Exception as e:
        last_error = e
        print(
            f"Optimizer {method} failed:",
            repr(e)
        )


if fit is None:
    raise RuntimeError(
        f"All optimizers failed. Last error: "
        f"{last_error}"
    )


# =============================================================================
# 8. SAVE FULL MODEL SUMMARY
# =============================================================================

summary_text = []

summary_text.append(
    "MODEL 1"
)

summary_text.append(
    "=" * 100
)

summary_text.append(
    "FC ~ Stage * G1c + (1 | Subject)"
)

summary_text.append(
    "Outcome = reference stage"
)

summary_text.append(
    "G1c = G1_bin - 5.5"
)

summary_text.append(
    f"N subjects = "
    f"{long_df['Subject'].nunique()}"
)

summary_text.append(
    f"N observations = {len(long_df)}"
)

summary_text.append(
    f"Optimizer = {fit_method}"
)

summary_text.append("")

summary_text.append(
    fit.summary().as_text()
)

summary_file = (
    OUT / "MODEL1_full_summary.txt"
)

summary_file.write_text(
    "\n".join(summary_text),
    encoding="utf-8"
)


# =============================================================================
# 9. FIXED EFFECT TABLE
# =============================================================================

fe = fit.fe_params

bse = fit.bse_fe

pvals = fit.pvalues.loc[
    fe.index
]

ci = fit.conf_int().loc[
    fe.index
]


fixed = pd.DataFrame({
    "term": fe.index,
    "beta": fe.values,
    "SE": bse.values,
    "CI95_low": ci[0].values,
    "CI95_high": ci[1].values,
    "p": pvals.values,
})


def meaning(term):

    if term == "Intercept":
        return (
            "Outcome mean FC at G1 midpoint"
        )

    if term == "G1c":
        return (
            "Outcome G1 slope"
        )

    if (
        "C(Stage" in term
        and ":G1c" not in term
    ):
        return (
            "Anticipation minus Outcome "
            "at G1 midpoint"
        )

    if ":G1c" in term:
        return (
            "PRIMARY: Anticipation slope "
            "minus Outcome slope "
            "(Stage x G1 interaction)"
        )

    return ""


fixed["interpretation"] = (
    fixed["term"]
    .apply(meaning)
)

fixed.to_csv(
    TAB / "MODEL1_fixed_effects.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 10. DERIVE STAGE SLOPES
# =============================================================================

interaction_terms = [
    x
    for x in fe.index
    if ":G1c" in x
]

stage_terms = [
    x
    for x in fe.index
    if (
        "C(Stage" in x
        and ":G1c" not in x
    )
]


if len(interaction_terms) != 1:
    raise RuntimeError(
        "Could not uniquely identify "
        "Stage x G1 interaction term."
    )

interaction_term = (
    interaction_terms[0]
)

outcome_slope = (
    fe["G1c"]
)

interaction_beta = (
    fe[interaction_term]
)

anticipation_slope = (
    outcome_slope +
    interaction_beta
)

interaction_p = (
    fit.pvalues[
        interaction_term
    ]
)

interaction_ci = (
    fit.conf_int()
    .loc[interaction_term]
    .to_numpy()
)


key = pd.DataFrame([{
    "N_subjects":
        long_df["Subject"].nunique(),

    "N_observations":
        len(long_df),

    "Outcome_slope":
        outcome_slope,

    "Anticipation_slope":
        anticipation_slope,

    "Anti_minus_Outcome_slope_difference":
        interaction_beta,

    "interaction_CI95_low":
        interaction_ci[0],

    "interaction_CI95_high":
        interaction_ci[1],

    "Stage_x_G1_p":
        interaction_p,

    "optimizer":
        fit_method,

    "converged":
        bool(fit.converged),
}])

key.to_csv(
    TAB / "MODEL1_KEY_RESULT.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 11. FINAL REPORT
# =============================================================================

print("\n" + "=" * 100)
print("MODEL 1 FIXED EFFECTS")
print("=" * 100)

print(
    fixed.to_string(
        index=False
    )
)

print("\n" + "=" * 100)
print("MODEL 1 KEY RESULT")
print("=" * 100)

print(
    f"Outcome slope       = "
    f"{outcome_slope:.8f}"
)

print(
    f"Anticipation slope  = "
    f"{anticipation_slope:.8f}"
)

print(
    f"Anti - Outcome slope difference "
    f"= {interaction_beta:.8f}"
)

print(
    f"95% CI = "
    f"[{interaction_ci[0]:.8f}, "
    f"{interaction_ci[1]:.8f}]"
)

print(
    f"Stage x G1 p = "
    f"{interaction_p:.10g}"
)

print(
    f"Converged = "
    f"{fit.converged}"
)

print(
    f"Optimizer = "
    f"{fit_method}"
)


if interaction_p < 0.05:

    if interaction_beta > 0:

        print(
            "\nINTERPRETATION:"
        )

        print(
            "Significant Stage x G1 interaction."
        )

        print(
            "Anticipation has a more positive / "
            "less negative G1 slope than Outcome."
        )

    else:

        print(
            "\nINTERPRETATION:"
        )

        print(
            "Significant Stage x G1 interaction."
        )

        print(
            "Anticipation has a more negative "
            "G1 slope than Outcome."
        )

else:

    print(
        "\nINTERPRETATION:"
    )

    print(
        "Stage x G1 interaction is not "
        "significant at alpha = .05."
    )


print("\nOutputs:")
print(
    TAB / "subject_stage_G1_long.tsv"
)
print(
    TAB / "subject_level_stage_G1_descriptives.tsv"
)
print(
    TAB / "MODEL1_fixed_effects.tsv"
)
print(
    TAB / "MODEL1_KEY_RESULT.tsv"
)
print(
    OUT / "MODEL1_full_summary.txt"
)
print(
    QC / "group_mean_reconstruction_QC.tsv"
)

print("\nDONE.")
