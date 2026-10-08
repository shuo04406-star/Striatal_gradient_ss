#!/usr/bin/env bash
set -euo pipefail

: "${STRIATAL_ANALYSIS_ROOT:?Set STRIATAL_ANALYSIS_ROOT to the private analysis root}"
BASE="${STRIATAL_ANALYSIS_ROOT}"

FCROOT="${FCROOT:-$BASE/17_current64_375matched_fingerprints/04_uniform_z_for_normative_analysis}"
COV="${COV:-$BASE/20_current64_reward_stage_fc/05_covariate_adjusted/covariate_master_64.tsv}"
G1="${G1:-$BASE/34_G1_orientation_FINAL/01_canonical_reference/g1_voxel_table_with_bins_CURRENT.tsv}"
MAP="${MAP:-$BASE/19_current64_375reference_gradients/01_reference/row_mapping_2847_to_2608.npy}"

OUT="${OUT:-$BASE/24_normative_G1_FC_deviation_FIXED/CURRENTbin_parent_validation}"
mkdir -p "$OUT"

export FCROOT COV G1 MAP OUT

python3 - <<'PY' 2>&1 | tee "$OUT/run.log"

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.regression.mixed_linear_model import MixedLM

warnings.filterwarnings("default")

FCROOT = Path(os.environ["FCROOT"])
COVFILE = Path(os.environ["COV"])
G1FILE = Path(os.environ["G1"])
MAPFILE = Path(os.environ["MAP"])
OUT = Path(os.environ["OUT"])

print("="*110)
print("CURRENT-BIN PARENT EFFECT VALIDATION")
print("All analyses locked to G1_bin_current")
print("="*110)


# =============================================================================
# 1. COVARIATES
# =============================================================================

print("\n"+"="*110)
print("STEP 1. VALIDATED 64-SUBJECT COVARIATES")
print("="*110)

cov = pd.read_csv(COVFILE, sep="\t").copy()

cov["subject"] = cov["subject"].astype(str)
cov["group"] = cov["group"].astype(str).str.upper()

cov["age"] = pd.to_numeric(cov["age"], errors="raise")
cov["meanFD"] = pd.to_numeric(cov["meanFD"], errors="raise")

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
    raise RuntimeError("Unrecognized sex values.")

cov["group_sz"] = (cov.group=="SZ").astype(int)

cov["age_c"] = cov.age - cov.age.mean()
cov["sex_c"] = cov.sex_num - cov.sex_num.mean()
cov["meanFD_c"] = cov.meanFD - cov.meanFD.mean()

if len(cov) != 64 or cov.subject.nunique()!=64:
    raise RuntimeError("Expected exactly 64 unique subjects.")

print(cov.groupby("group").subject.nunique())
print("Age mean     =", cov.age.mean())
print("Mean FD mean =", cov.meanFD.mean())


# =============================================================================
# 2. LOCK CURRENT BIN
# =============================================================================

print("\n"+"="*110)
print("STEP 2. LOCK G1_bin_current")
print("="*110)

g1 = pd.read_csv(G1FILE, sep="\t")

if len(g1) != 2847:
    raise RuntimeError(
        f"Expected 2847 canonical voxels; got {len(g1)}"
    )

BINCOL = "G1_bin_current"

if BINCOL not in g1.columns:
    raise RuntimeError(f"{BINCOL} absent.")

current_full = pd.to_numeric(
    g1[BINCOL],
    errors="raise"
).astype(int).to_numpy()

if set(np.unique(current_full)) != set(range(1,11)):
    raise RuntimeError("Current bins are not 1..10.")

mapping = np.asarray(
    np.load(MAPFILE)
).reshape(-1).astype(int)

if len(mapping) != 2608:
    raise RuntimeError("Mapping is not length 2608.")

current_common = current_full[mapping]

print("Current-bin counts on full 2847:")
for b in range(1,11):
    print(
        f"  Bin {b:2d}: "
        f"{np.sum(current_full==b):4d}"
    )

print()
print("Current-bin counts on common 2608:")
for b in range(1,11):
    print(
        f"  Bin {b:2d}: "
        f"{np.sum(current_common==b):4d}"
    )


# =============================================================================
# 3. AUDIT OLD RECOMPUTED EQUAL-COUNT BINS
# =============================================================================

print("\n"+"="*110)
print("STEP 3. WHY DID THE PREVIOUS QC DIFFER?")
print("="*110)

def equal_count_bins(x, n_bins=10):
    x=np.asarray(x,float)
    order=np.argsort(x,kind="mergesort")
    b=np.empty(len(x),dtype=int)

    labels=(
        np.floor(
            np.arange(len(x))*n_bins/len(x)
        ).astype(int)+1
    )

    labels=np.clip(
        labels,
        1,
        n_bins
    )

    b[order]=labels
    return b


# Find canonical G1 numeric coordinate, excluding bin columns.
g1_value_col = None

preferred = [
    "G1",
    "g1",
    "G1_value",
    "g1_value",
    "gradient1",
    "gradient_1"
]

for c in preferred:
    if c in g1.columns and "bin" not in c.lower():
        try:
            x=pd.to_numeric(g1[c],errors="coerce")
            if x.notna().all() and x.std()>0:
                g1_value_col=c
                break
        except:
            pass

if g1_value_col is None:

    candidates=[]

    for c in g1.columns:

        if "bin" in c.lower():
            continue

        if "g1" not in c.lower() and "gradient" not in c.lower():
            continue

        try:
            x=pd.to_numeric(g1[c],errors="coerce")

            if x.notna().all() and x.std()>0:
                candidates.append(c)

        except:
            pass

    if len(candidates)==1:
        g1_value_col=candidates[0]


if g1_value_col is not None:

    g1_value=pd.to_numeric(
        g1[g1_value_col],
        errors="raise"
    ).to_numpy(float)

    old_full=equal_count_bins(
        g1_value,
        10
    )

    old_common=old_full[mapping]

    full_changed=int(
        np.sum(
            old_full != current_full
        )
    )

    common_changed=int(
        np.sum(
            old_common != current_common
        )
    )

    old10=set(
        np.where(old_common==10)[0]
    )

    current10=set(
        np.where(current_common==10)[0]
    )

    overlap=len(
        old10 & current10
    )

    union=len(
        old10 | current10
    )

    jaccard=(
        overlap/union
        if union else np.nan
    )

    print("Old script G1 coordinate column:",g1_value_col)

    print(
        "Full voxels assigned to a different bin =",
        full_changed,
        "/ 2847"
    )

    print(
        "Common voxels assigned to a different bin =",
        common_changed,
        "/ 2608"
    )

    print(
        "Old Bin10 N     =",
        len(old10)
    )

    print(
        "Current Bin10 N =",
        len(current10)
    )

    print(
        "Bin10 overlap   =",
        overlap
    )

    print(
        "Bin10 Jaccard   =",
        jaccard
    )

    pd.DataFrame({
        "canonical_row_0based":
            np.arange(2847),
        "old_equal_count_bin":
            old_full,
        "G1_bin_current":
            current_full,
        "same_bin":
            old_full==current_full
    }).to_csv(
        OUT/"01_old_vs_current_bin_audit.tsv",
        sep="\t",
        index=False
    )

else:

    print(
        "Could not uniquely identify continuous G1 column."
    )

    print(
        "This does not affect the CURRENT-bin reanalysis."
    )


# =============================================================================
# 4. LOAD 64 FC MATRICES AND COLLAPSE USING CURRENT BIN MEMBERSHIP
# =============================================================================

print("\n"+"="*110)
print("STEP 4. 2608 × 65 → CURRENT 10 × 65")
print("="*110)

subjects=cov.subject.tolist()

F=np.empty(
    (64,10,65),
    dtype=float
)

for i,sid in enumerate(subjects):

    p=FCROOT/f"{sid}_fingerprint_z_2608x65.npy"

    if not p.exists():
        raise FileNotFoundError(p)

    a=np.asarray(
        np.load(p),
        dtype=float
    )

    if a.shape==(65,2608):
        a=a.T

    if a.shape!=(2608,65):
        raise RuntimeError(
            f"{sid}: shape={a.shape}"
        )

    if not np.isfinite(a).all():
        raise RuntimeError(
            f"{sid}: nonfinite values"
        )

    for b in range(1,11):

        mask=current_common==b

        F[i,b-1,:]=(
            a[mask,:].mean(axis=0)
        )

    print(
        f"[{i+1:02d}/64] {sid}"
    )


# =============================================================================
# 5. HC NORMATIVE REFERENCE
# =============================================================================

print("\n"+"="*110)
print("STEP 5. REBUILD CURRENT-BIN NORMATIVE METRICS")
print("="*110)

hc_idx=np.where(
    cov.group.to_numpy()=="HC"
)[0]

sz_idx=np.where(
    cov.group.to_numpy()=="SZ"
)[0]

if len(hc_idx)!=32 or len(sz_idx)!=32:
    raise RuntimeError("Expected HC=32, SZ=32.")

hc_F=F[hc_idx]

mu_hc=hc_F.mean(axis=0)

sd_hc=hc_F.std(
    axis=0,
    ddof=1
)

sd_hc=np.where(
    sd_hc<1e-8,
    1e-8,
    sd_hc
)

Z=np.empty_like(F)

profile_r=np.empty(
    (64,10),
    dtype=float
)


def corr_safe(a,b):

    if np.std(a)<1e-12 or np.std(b)<1e-12:
        return np.nan

    return float(
        np.corrcoef(a,b)[0,1]
    )


# SZ uses all 32 HC.
for i in sz_idx:

    Z[i]=(
        F[i]-mu_hc
    )/sd_hc

    for b in range(10):

        profile_r[i,b]=corr_safe(
            F[i,b],
            mu_hc[b]
        )


# HC uses leave-one-out HC reference.
for i in hc_idx:

    others=hc_idx[
        hc_idx!=i
    ]

    mu=F[others].mean(axis=0)

    sd=F[others].std(
        axis=0,
        ddof=1
    )

    sd=np.where(
        sd<1e-8,
        1e-8,
        sd
    )

    Z[i]=(
        F[i]-mu
    )/sd

    for b in range(10):

        profile_r[i,b]=corr_safe(
            F[i,b],
            mu[b]
        )


profile_z=np.arctanh(
    np.clip(
        profile_r,
        -0.999999,
        0.999999
    )
)

rms=np.sqrt(
    np.mean(
        Z**2,
        axis=2
    )
)

burden=np.mean(
    np.abs(Z)>1.96,
    axis=2
)

signed_mean=np.mean(
    Z,
    axis=2
)


# =============================================================================
# 6. LONG TABLE
# =============================================================================

rows=[]

for i,sid in enumerate(subjects):

    cr=cov.iloc[i]

    for b in range(1,11):

        rows.append({
            "subject":sid,
            "group":cr.group,
            "group_sz":cr.group_sz,
            "bin":b,
            "rms_deviation":
                rms[i,b-1],
            "profile_r":
                profile_r[i,b-1],
            "profile_z":
                profile_z[i,b-1],
            "abnormal_burden":
                burden[i,b-1],
            "signed_mean_z":
                signed_mean[i,b-1],
            "age_c":
                cr.age_c,
            "sex_c":
                cr.sex_c,
            "meanFD_c":
                cr.meanFD_c
        })

d=pd.DataFrame(rows)

d.to_csv(
    OUT/"02_subject_bin_metrics_G1_bin_current.tsv",
    sep="\t",
    index=False
)


# =============================================================================
# 7. EXACT ORIGINAL MIXED-MODEL LRT
# =============================================================================

print("\n"+"="*110)
print("STEP 6. ORIGINAL MIXED-MODEL GROUP × BIN TESTS")
print("="*110)


def fit_mixed(formula,data,label):

    model=MixedLM.from_formula(
        formula,
        groups="subject",
        data=data
    )

    last=None

    for method in [
        "lbfgs",
        "bfgs",
        "cg",
        "powell"
    ]:

        try:

            res=model.fit(
                reml=False,
                method=method,
                maxiter=3000,
                disp=False
            )

            if np.isfinite(res.llf):
                return res,method

        except Exception as e:
            last=e

    raise RuntimeError(
        f"{label} MixedLM failed: {last}"
    )


def lrt(full,reduced):

    stat=max(
        0.0,
        2.0*(
            full.llf
            - reduced.llf
        )
    )

    df=max(
        1,
        int(
            round(
                len(full.fe_params)
                - len(reduced.fe_params)
            )
        )
    )

    p=float(
        chi2.sf(
            stat,
            df
        )
    )

    return stat,df,p


model_rows=[]

for metric,role in [
    ("rms_deviation","PRIMARY"),
    ("profile_z","SECONDARY"),
    ("abnormal_burden","SECONDARY")
]:

    full_formula=(
        f"{metric} ~ "
        "group_sz*C(bin) "
        "+ age_c + sex_c + meanFD_c"
    )

    reduced_formula=(
        f"{metric} ~ "
        "group_sz + C(bin) "
        "+ age_c + sex_c + meanFD_c"
    )

    full,m1=fit_mixed(
        full_formula,
        d,
        metric+"_full"
    )

    red,m0=fit_mixed(
        reduced_formula,
        d,
        metric+"_reduced"
    )

    stat,dfx,p=lrt(
        full,
        red
    )

    model_rows.append({
        "role":role,
        "metric":metric,
        "effect":"Group_x_Bin",
        "chi2":stat,
        "df":dfx,
        "p":p,
        "full_method":m1,
        "reduced_method":m0
    })

models=pd.DataFrame(
    model_rows
)

models.to_csv(
    OUT/"03_mixed_model_omnibus_G1_bin_current.tsv",
    sep="\t",
    index=False
)

print(
    models.to_string(
        index=False
    )
)


# =============================================================================
# 8. ROBUST PROFILE GROUP × BIN
# =============================================================================

print("\n"+"="*110)
print("STEP 7. PROFILE SIMILARITY ROBUSTNESS")
print("="*110)

FORMULA=(
    "profile_z ~ "
    "group_sz*C(bin) "
    "+ age_c + sex_c + meanFD_c"
)


def interaction_names(result):

    return [
        n for n in result.params.index
        if "group_sz" in n
        and "C(bin)" in n
    ]


def joint_wald(result,names):

    allnames=list(
        result.params.index
    )

    R=np.zeros(
        (
            len(names),
            len(allnames)
        )
    )

    for i,n in enumerate(names):

        R[
            i,
            allnames.index(n)
        ]=1

    beta=np.asarray(
        result.params,
        dtype=float
    )

    V=np.asarray(
        result.cov_params(),
        dtype=float
    )

    x=R@beta
    C=R@V@R.T

    W=float(
        x.T
        @ np.linalg.pinv(C)
        @ x
    )

    df=len(names)

    p=float(
        chi2.sf(
            W,
            df
        )
    )

    return W,df,p


gee=smf.gee(
    FORMULA,
    groups="subject",
    data=d,
    family=sm.families.Gaussian(),
    cov_struct=Exchangeable()
).fit()

gee_terms=interaction_names(
    gee
)

WG,dfG,pG=joint_wald(
    gee,
    gee_terms
)


ols=smf.ols(
    FORMULA,
    data=d
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups":d.subject,
        "use_correction":True
    }
)

ols_terms=interaction_names(
    ols
)

WO,dfO,pO=joint_wald(
    ols,
    ols_terms
)


robust=pd.DataFrame([
    {
        "analysis":"GEE_exchangeable",
        "effect":"Group_x_Bin",
        "chi2":WG,
        "df":dfG,
        "p":pG
    },
    {
        "analysis":"OLS_subject_clustered",
        "effect":"Group_x_Bin",
        "chi2":WO,
        "df":dfO,
        "p":pO
    }
])

robust.to_csv(
    OUT/"04_profile_similarity_robust_omnibus_G1_bin_current.tsv",
    sep="\t",
    index=False
)

print(
    robust.to_string(
        index=False
    )
)


# =============================================================================
# 9. ADJUSTED PROFILE CONTRASTS IN ALL TEN BINS
# =============================================================================

print("\n"+"="*110)
print("STEP 8. PROFILE SIMILARITY BIN CONTRASTS")
print("="*110)

names=list(
    ols.params.index
)

beta=np.asarray(
    ols.params,
    dtype=float
)

V=np.asarray(
    ols.cov_params(),
    dtype=float
)

gidx=names.index(
    "group_sz"
)

rows=[]

for b in range(1,11):

    L=np.zeros(
        len(names)
    )

    L[gidx]=1.0

    if b>1:

        candidates=[
            n for n in names
            if "group_sz" in n
            and "C(bin)" in n
            and (
                f"[T.{b}]" in n
                or f"[T.{float(b)}]" in n
            )
        ]

        if len(candidates)!=1:
            raise RuntimeError(
                f"Bin {b} interaction lookup failed: "
                f"{candidates}"
            )

        L[
            names.index(
                candidates[0]
            )
        ]=1.0

    est=float(
        L@beta
    )

    var=float(
        L@V@L.T
    )

    se=np.sqrt(
        max(
            var,
            0
        )
    )

    z=(
        est/se
        if se>0
        else np.nan
    )

    p=float(
        2*(
            1-sm.stats.norm.cdf(abs(z))
        )
    ) if False else None

    # scipy-free normal survival via statsmodels/scipy unavailable shortcut:
    from scipy.stats import norm

    p=float(
        2*norm.sf(abs(z))
    )

    rows.append({
        "bin":b,
        "contrast":"SZ_minus_HC",
        "estimate_profile_z":
            est,
        "SE_cluster":
            se,
        "z":
            z,
        "p_raw":
            p,
        "CI95_low":
            est-1.96*se,
        "CI95_high":
            est+1.96*se,
        "HC_mean_r":
            d.loc[
                (d.group=="HC")
                & (d.bin==b),
                "profile_r"
            ].mean(),
        "SZ_mean_r":
            d.loc[
                (d.group=="SZ")
                & (d.bin==b),
                "profile_r"
            ].mean()
    })


post=pd.DataFrame(
    rows
)

post["p_FDR_BH"]=multipletests(
    post.p_raw,
    method="fdr_bh"
)[1]

post["FDR_q_lt_05"]=(
    post.p_FDR_BH<0.05
)

post.to_csv(
    OUT/"05_profile_bin_contrasts_G1_bin_current.tsv",
    sep="\t",
    index=False
)

print(
    post.to_string(
        index=False
    )
)


# =============================================================================
# 10. FINAL DECISION
# =============================================================================

b10=post[
    post.bin==10
].iloc[0]

profile_mixed_p=float(
    models.loc[
        models.metric=="profile_z",
        "p"
    ].iloc[0]
)

rms_p=float(
    models.loc[
        models.metric=="rms_deviation",
        "p"
    ].iloc[0]
)

burden_p=float(
    models.loc[
        models.metric=="abnormal_burden",
        "p"
    ].iloc[0]
)


print("\n"+"="*110)
print("FINAL CURRENT-BIN DECISION")
print("="*110)

print(
    f"RMS mixed Group×Bin p       = {rms_p:.10g}"
)

print(
    f"Burden mixed Group×Bin p    = {burden_p:.10g}"
)

print(
    f"Profile mixed Group×Bin p   = {profile_mixed_p:.10g}"
)

print(
    f"Profile GEE p               = {pG:.10g}"
)

print(
    f"Profile cluster-OLS p       = {pO:.10g}"
)

print()

print(
    f"CURRENT Bin10 HC mean r     = {b10.HC_mean_r:.6f}"
)

print(
    f"CURRENT Bin10 SZ mean r     = {b10.SZ_mean_r:.6f}"
)

print(
    f"CURRENT Bin10 SZ-HC z beta  = "
    f"{b10.estimate_profile_z:+.6f}"
)

print(
    f"CURRENT Bin10 raw p         = {b10.p_raw:.10g}"
)

print(
    f"CURRENT Bin10 FDR q         = {b10.p_FDR_BH:.10g}"
)

print()

if (
    profile_mixed_p < .05
    and pG < .05
    and pO < .05
    and b10.p_FDR_BH < .05
):
    print(
        "STATUS: CURRENT-BIN PARENT EFFECT CONFIRMED."
    )

    print(
        "It is now valid to decompose CURRENT Bin10."
    )

else:

    print(
        "STATUS: CURRENT-BIN PARENT EFFECT NOT FULLY CONFIRMED."
    )

    print(
        "Do NOT interpret the subfamily/ROI decomposition yet."
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
echo "CURRENT-BIN VALIDATION COMPLETE"
echo "======================================================================"

echo
echo "1. MIXED MODELS"
column -t -s $'\t' \
"$OUT/03_mixed_model_omnibus_G1_bin_current.tsv"

echo
echo "2. PROFILE ROBUSTNESS"
column -t -s $'\t' \
"$OUT/04_profile_similarity_robust_omnibus_G1_bin_current.tsv"

echo
echo "3. PROFILE BIN CONTRASTS"
column -t -s $'\t' \
"$OUT/05_profile_bin_contrasts_G1_bin_current.tsv"

echo
echo "OUTPUT:"
echo "$OUT"
