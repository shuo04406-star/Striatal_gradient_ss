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

OUT="${OUT:-$BASE/24_normative_G1_FC_deviation_FIXED/stage_specific_profile_similarity_FINAL_LOCKED}"
mkdir -p "$OUT"

export FCROOT COV G1 MAP ANTI OUTCOME EXCLUDED OUT

python3 - <<'PY' 2>&1 | tee "$OUT/run.log"

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

FCROOT = Path(os.environ["FCROOT"])
COVFILE = Path(os.environ["COV"])
G1FILE = Path(os.environ["G1"])
MAPFILE = Path(os.environ["MAP"])
ANTI_FILE = Path(os.environ["ANTI"])
OUTCOME_FILE = Path(os.environ["OUTCOME"])
EXCLUDED_FILE = Path(os.environ["EXCLUDED"])
OUT = Path(os.environ["OUT"])

print("="*110)
print("FINAL LOCKED STAGE-SPECIFIC NORMATIVE PROFILE ANALYSIS")
print("="*110)

# ============================================================
# 0. INPUT EXISTENCE
# ============================================================

for p in [
    FCROOT, COVFILE, G1FILE, MAPFILE,
    ANTI_FILE, OUTCOME_FILE, EXCLUDED_FILE
]:
    if not p.exists():
        raise FileNotFoundError(p)

# ============================================================
# 1. LOCK 43 / 19 / 3 ROI FAMILIES
# ============================================================

print("\n"+"="*110)
print("STEP 1. LOCK ROI FAMILIES")
print("="*110)

anti = pd.read_csv(ANTI_FILE, sep="\t")
outc = pd.read_csv(OUTCOME_FILE, sep="\t")
exc  = pd.read_csv(EXCLUDED_FILE, sep="\t")

for name,d in [
    ("Anticipation",anti),
    ("Outcome",outc),
    ("Excluded",exc)
]:
    if "roi_index0" not in d.columns:
        raise RuntimeError(
            f"{name}: roi_index0 missing; columns={list(d.columns)}"
        )

anti_idx = set(pd.to_numeric(anti["roi_index0"]).astype(int))
out_idx  = set(pd.to_numeric(outc["roi_index0"]).astype(int))
exc_idx  = set(pd.to_numeric(exc["roi_index0"]).astype(int))

print("Anticipation =", len(anti_idx))
print("Outcome      =", len(out_idx))
print("Excluded     =", len(exc_idx))

assert len(anti_idx) == 43
assert len(out_idx) == 19
assert len(exc_idx) == 3

if anti_idx & out_idx:
    raise RuntimeError("Anti/Outcome overlap")
if anti_idx & exc_idx:
    raise RuntimeError("Anti/Excluded overlap")
if out_idx & exc_idx:
    raise RuntimeError("Outcome/Excluded overlap")

if anti_idx | out_idx | exc_idx != set(range(65)):
    raise RuntimeError("43+19+3 does not exactly cover ROI indices 0..64")

print("ROI FAMILY QC: PASS")

# ============================================================
# 2. VALIDATED 64-PERSON COVARIATES
# ============================================================

print("\n"+"="*110)
print("STEP 2. LOAD VALIDATED 64-SUBJECT COVARIATES")
print("="*110)

cov = pd.read_csv(COVFILE, sep="\t").copy()

cov["subject"] = cov["subject"].astype(str)
cov["group"] = cov["group"].astype(str).str.upper()
cov["age"] = pd.to_numeric(cov["age"], errors="raise")
cov["meanFD"] = pd.to_numeric(cov["meanFD"], errors="raise")

cov["sex_num"] = (
    cov["sex"].astype(str).str.lower()
    .map({
        "male":1, "m":1,
        "female":0, "f":0
    })
)

if cov["sex_num"].isna().any():
    raise RuntimeError("Unrecognized sex coding")

cov["group_sz"] = (cov["group"]=="SZ").astype(int)

cov["age_c"] = cov["age"] - cov["age"].mean()
cov["sex_c"] = cov["sex_num"] - cov["sex_num"].mean()
cov["meanFD_c"] = cov["meanFD"] - cov["meanFD"].mean()

if len(cov) != 64:
    raise RuntimeError(f"Expected 64 subjects; found {len(cov)}")

if cov["subject"].nunique() != 64:
    raise RuntimeError("Subject IDs are not unique")

print(cov.groupby("group")["subject"].nunique())
print(f"Age mean       = {cov.age.mean():.6f}")
print(f"Mean FD mean   = {cov.meanFD.mean():.9f}")
print(f"Male proportion= {cov.sex_num.mean():.6f}")

# ============================================================
# 3. EXPLICIT CURRENT G1 BIN — NO DISCOVERY
# ============================================================

print("\n"+"="*110)
print("STEP 3. LOCK G1_bin_current")
print("="*110)

g1 = pd.read_csv(G1FILE, sep="\t")

if len(g1) != 2847:
    raise RuntimeError(
        f"Expected 2847 canonical voxels; found {len(g1)}"
    )

BIN_COL = "G1_bin_current"

if BIN_COL not in g1.columns:
    raise RuntimeError(
        f"{BIN_COL} not present; columns={list(g1.columns)}"
    )

full_bins = pd.to_numeric(
    g1[BIN_COL],
    errors="raise"
).astype(int).to_numpy()

if set(np.unique(full_bins)) != set(range(1,11)):
    raise RuntimeError(
        f"Unexpected {BIN_COL} values: {sorted(np.unique(full_bins))}"
    )

mapping = np.asarray(
    np.load(MAPFILE)
).reshape(-1).astype(int)

if len(mapping) != 2608:
    raise RuntimeError(
        f"Expected mapping length 2608; found {len(mapping)}"
    )

if mapping.min() < 0 or mapping.max() >= 2847:
    raise RuntimeError(
        f"Mapping range invalid: {mapping.min()}..{mapping.max()}"
    )

common_bins = full_bins[mapping]

print("Locked column =", BIN_COL)
print("Canonical voxels = 2847")
print("Common voxels    = 2608")

for b in range(1,11):
    print(
        f"Bin {b:2d}: {(common_bins==b).sum():4d} common voxels"
    )

# ============================================================
# 4. LOAD UNIFORM FISHER-Z FC → 10 × 65
# ============================================================

print("\n"+"="*110)
print("STEP 4. LOAD UNIFORM FISHER-Z FC")
print("="*110)

fc = {}

for i,sid in enumerate(cov["subject"],1):

    p = FCROOT / f"{sid}_fingerprint_z_2608x65.npy"

    if not p.exists():
        raise FileNotFoundError(p)

    a = np.asarray(np.load(p), dtype=float)

    if a.shape == (65,2608):
        a = a.T

    if a.shape != (2608,65):
        raise RuntimeError(
            f"{sid}: expected 2608×65, got {a.shape}"
        )

    if not np.isfinite(a).all():
        raise RuntimeError(f"{sid}: non-finite FC values")

    bm = np.empty((10,65),dtype=float)

    for b in range(1,11):
        bm[b-1,:] = a[common_bins==b,:].mean(axis=0)

    fc[sid] = bm

    print(f"[{i:02d}/64] {sid}")

print("FC LOAD QC: PASS")

# ============================================================
# 5. HC NORMATIVE REFERENCE
# ============================================================

print("\n"+"="*110)
print("STEP 5. BUILD HC NORMATIVE PROFILES")
print("="*110)

hc_subjects = cov.loc[
    cov["group"]=="HC","subject"
].tolist()

if len(hc_subjects) != 32:
    raise RuntimeError("Expected 32 HC")

hc_mean = np.stack(
    [fc[s] for s in hc_subjects],
    axis=0
).mean(axis=0)

stage_cols = {
    "Anticipation": np.array(sorted(anti_idx),dtype=int),
    "Outcome": np.array(sorted(out_idx),dtype=int)
}

def safe_corr(a,b):
    a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float)

    if np.std(a)==0 or np.std(b)==0:
        return np.nan

    return float(np.corrcoef(a,b)[0,1])

def fisher_z(r):
    return float(
        np.arctanh(
            np.clip(r,-0.999999,0.999999)
        )
    )

rows=[]

for _,cr in cov.iterrows():

    sid=cr["subject"]
    grp=cr["group"]

    if grp=="HC":
        others=[s for s in hc_subjects if s != sid]

        template=np.stack(
            [fc[s] for s in others],
            axis=0
        ).mean(axis=0)

        reference="HC_LOO"

    else:
        template=hc_mean
        reference="HC32"

    for b in range(1,11):

        for stage,cols in stage_cols.items():

            r=safe_corr(
                fc[sid][b-1,cols],
                template[b-1,cols]
            )

            rows.append({
                "subject":sid,
                "group":grp,
                "group_sz":int(grp=="SZ"),
                "bin":b,
                "stage":stage,
                "stage_out":int(stage=="Outcome"),
                "n_roi":len(cols),
                "profile_r":r,
                "profile_z":fisher_z(r),
                "reference":reference,
                "age_c":cr["age_c"],
                "sex_c":cr["sex_c"],
                "meanFD_c":cr["meanFD_c"]
            })

d=pd.DataFrame(rows)

if len(d) != 1280:
    raise RuntimeError(
        f"Expected 1280 rows; found {len(d)}"
    )

if d["profile_z"].isna().any():
    raise RuntimeError("NaN profile similarity detected")

d.to_csv(
    OUT/"stage_specific_subject_bin_metrics.tsv",
    sep="\t",
    index=False
)

print("Rows     =",len(d))
print("Subjects =",d.subject.nunique())
print("PROFILE CALCULATION QC: PASS")

# ============================================================
# 6. DESCRIPTIVES
# ============================================================

desc=(
    d.groupby(
        ["stage","group","bin"],
        as_index=False
    )
    .agg(
        N=("subject","nunique"),
        mean_r=("profile_r","mean"),
        sd_r=("profile_r","std"),
        mean_z=("profile_z","mean"),
        sd_z=("profile_z","std")
    )
)

desc.to_csv(
    OUT/"stage_specific_descriptives.tsv",
    sep="\t",
    index=False
)

# ============================================================
# HELPER: JOINT WALD
# ============================================================

def joint_wald(model,names):

    allnames=list(model.params.index)

    if len(names)==0:
        raise RuntimeError("No terms supplied to Wald test")

    R=np.zeros((len(names),len(allnames)))

    for i,n in enumerate(names):

        if n not in allnames:
            raise RuntimeError(f"Parameter missing: {n}")

        R[i,allnames.index(n)]=1

    beta=np.asarray(model.params,dtype=float)
    V=np.asarray(model.cov_params(),dtype=float)

    x=R@beta
    C=R@V@R.T

    W=float(
        x.T @ np.linalg.pinv(C) @ x
    )

    df=len(names)
    p=float(chi2.sf(W,df))

    return W,df,p

# ============================================================
# 7. GROUP × BIN × STAGE — MAIN TEST
# ============================================================

print("\n"+"="*110)
print("STEP 7. GROUP × BIN × STAGE")
print("="*110)

FORMULA=(
    "profile_z ~ "
    "group_sz*C(bin)*stage_out "
    "+ age_c + sex_c + meanFD_c"
)

gee=smf.gee(
    FORMULA,
    groups="subject",
    data=d,
    family=sm.families.Gaussian(),
    cov_struct=Exchangeable()
).fit()

gee_terms=[
    n for n in gee.params.index
    if "group_sz" in n
    and "C(bin)" in n
    and "stage_out" in n
]

if len(gee_terms)!=9:
    raise RuntimeError(
        f"Expected 9 GEE 3-way terms; got {len(gee_terms)}"
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
        "groups":d["subject"],
        "use_correction":True
    }
)

ols_terms=[
    n for n in ols.params.index
    if "group_sz" in n
    and "C(bin)" in n
    and "stage_out" in n
]

if len(ols_terms)!=9:
    raise RuntimeError(
        f"Expected 9 cluster-OLS 3-way terms; got {len(ols_terms)}"
    )

WO,dfO,pO=joint_wald(
    ols,
    ols_terms
)

omnibus=pd.DataFrame([
    {
        "analysis":"GEE_exchangeable",
        "effect":"Group_x_Bin_x_Stage",
        "chi2":WG,
        "df":dfG,
        "p":pG
    },
    {
        "analysis":"OLS_subject_clustered",
        "effect":"Group_x_Bin_x_Stage",
        "chi2":WO,
        "df":dfO,
        "p":pO
    }
])

omnibus.to_csv(
    OUT/"01_Group_Bin_Stage_omnibus.tsv",
    sep="\t",
    index=False
)

print(omnibus.to_string(index=False))

# ============================================================
# 8. GROUP × BIN WITHIN EACH STAGE
# ============================================================

print("\n"+"="*110)
print("STEP 8. GROUP × BIN WITHIN ANTICIPATION / OUTCOME")
print("="*110)

stage_rows=[]

for stage in ["Anticipation","Outcome"]:

    ds=d[d["stage"]==stage].copy()

    m=smf.ols(
        "profile_z ~ "
        "group_sz*C(bin) "
        "+ age_c + sex_c + meanFD_c",
        data=ds
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":ds["subject"],
            "use_correction":True
        }
    )

    terms=[
        n for n in m.params.index
        if "group_sz" in n
        and "C(bin)" in n
    ]

    W,dfx,p=joint_wald(m,terms)

    stage_rows.append({
        "stage":stage,
        "effect":"Group_x_Bin",
        "chi2":W,
        "df":dfx,
        "p":p
    })

stage_res=pd.DataFrame(stage_rows)

stage_res["p_FDR_BH"]=multipletests(
    stage_res["p"],
    method="fdr_bh"
)[1]

stage_res.to_csv(
    OUT/"02_Within_stage_Group_Bin.tsv",
    sep="\t",
    index=False
)

print(stage_res.to_string(index=False))

# ============================================================
# 9. BIN10 DIRECT GROUP × STAGE
# ============================================================

print("\n"+"="*110)
print("STEP 9. BIN10 DIRECT GROUP × STAGE DIFFERENCE-OF-DIFFERENCES")
print("="*110)

b10=d[d["bin"]==10].copy()

m10=smf.ols(
    "profile_z ~ "
    "group_sz*stage_out "
    "+ age_c + sex_c + meanFD_c",
    data=b10
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups":b10["subject"],
        "use_correction":True
    }
)

term="group_sz:stage_out"

if term not in m10.params.index:
    term="stage_out:group_sz"

if term not in m10.params.index:
    raise RuntimeError(
        "Could not find Bin10 Group×Stage interaction term"
    )

did_est=float(m10.params[term])
did_se=float(m10.bse[term])
did_z=float(m10.tvalues[term])
did_p=float(m10.pvalues[term])

did=pd.DataFrame([{
    "bin":10,
    "effect":
        "(SZ-HC)_Outcome_minus_(SZ-HC)_Anticipation",
    "estimate_profile_z":did_est,
    "SE_cluster":did_se,
    "z":did_z,
    "p":did_p,
    "CI95_low":did_est-1.96*did_se,
    "CI95_high":did_est+1.96*did_se
}])

did.to_csv(
    OUT/"03_Bin10_Group_x_Stage_DID.tsv",
    sep="\t",
    index=False
)

print(did.to_string(index=False))

# ============================================================
# 10. BIN10 WITHIN-STAGE SZ–HC
# ============================================================

print("\n"+"="*110)
print("STEP 10. BIN10 SZ-HC WITHIN EACH STAGE")
print("="*110)

contrast_rows=[]

for stage in ["Anticipation","Outcome"]:

    ds=b10[b10["stage"]==stage].copy()

    m=smf.ols(
        "profile_z ~ "
        "group_sz + age_c + sex_c + meanFD_c",
        data=ds
    ).fit(cov_type="HC3")

    beta=float(m.params["group_sz"])
    se=float(m.bse["group_sz"])
    p=float(m.pvalues["group_sz"])

    contrast_rows.append({
        "stage":stage,
        "contrast":"SZ_minus_HC",
        "estimate_profile_z":beta,
        "SE_HC3":se,
        "t_HC3":float(m.tvalues["group_sz"]),
        "p_raw":p,
        "CI95_low":beta-1.96*se,
        "CI95_high":beta+1.96*se,
        "HC_mean_r":
            ds.loc[ds.group=="HC","profile_r"].mean(),
        "SZ_mean_r":
            ds.loc[ds.group=="SZ","profile_r"].mean()
    })

contrasts=pd.DataFrame(contrast_rows)

contrasts["p_FDR_BH"]=multipletests(
    contrasts["p_raw"],
    method="fdr_bh"
)[1]

contrasts["FDR_q_lt_05"]=(
    contrasts["p_FDR_BH"] < 0.05
)

contrasts.to_csv(
    OUT/"04_Bin10_within_stage_SZ_HC.tsv",
    sep="\t",
    index=False
)

print(contrasts.to_string(index=False))

# ============================================================
# FINAL
# ============================================================

print("\n"+"="*110)
print("FINAL KEY RESULTS")
print("="*110)

print(
    f"GEE Group×Bin×Stage p       = {pG:.10g}"
)
print(
    f"Cluster OLS 3-way p         = {pO:.10g}"
)
print(
    f"Bin10 Group×Stage DID p     = {did_p:.10g}"
)
print(
    f"Bin10 DID estimate          = {did_est:+.6f}"
)

print()
print("DID direction:")
print(
    "negative = SZ-vs-HC profile reduction is stronger for Outcome"
)
print(
    "positive = SZ-vs-HC profile reduction is stronger for Anticipation"
)

print()
print("Output:", OUT)

PY

STATUS=${PIPESTATUS[0]}

echo
echo "======================================================================"

if [ "$STATUS" -ne 0 ]; then
    echo "ANALYSIS STOPPED WITH ERROR"
    echo "SSH WILL REMAIN OPEN."
    echo "Log:"
    echo "$OUT/run.log"
    exit "$STATUS"
fi

echo "ANALYSIS COMPLETED"
echo "======================================================================"

echo
echo "1. GROUP × BIN × STAGE"
column -t -s $'\t' "$OUT/01_Group_Bin_Stage_omnibus.tsv"

echo
echo "2. WITHIN-STAGE GROUP × BIN"
column -t -s $'\t' "$OUT/02_Within_stage_Group_Bin.tsv"

echo
echo "3. BIN10 DIRECT GROUP × STAGE"
column -t -s $'\t' "$OUT/03_Bin10_Group_x_Stage_DID.tsv"

echo
echo "4. BIN10 WITHIN-STAGE SZ-HC"
column -t -s $'\t' "$OUT/04_Bin10_within_stage_SZ_HC.tsv"

echo
echo "ALL OUTPUTS:"
echo "$OUT"
