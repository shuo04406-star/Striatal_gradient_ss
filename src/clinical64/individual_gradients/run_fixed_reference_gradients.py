#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CURRENT64 exact 375-reference individual gradient pipeline

MAIN ANALYSIS
-------------
1. Load final exact current64 Fisher-z fingerprints:
       64 x 2608 x 65

2. Load frozen 375 group gradients:
       2847 x 10

3. Apply the already validated exact 2847 -> 2608 row mapping:
       375 reference = 2608 x 10

4. For each current64 subject:
       2608 x 65 Fisher-z fingerprint
           ->
       BrainSpace GradientMaps
           n_components = 10
           approach      = "dm"
           kernel        = "normalized_angle"
           sparsity      = 0.0
           random_state  = 0
           no explicit diffusion_time
           ->
       raw individual gradients = 2608 x 10

5. Replicate final 375 individual-gradient orientation:
       component-wise sign orientation to fixed group reference
       NO permutation
       NO Procrustes yet

6. Replicate final 375 Procrustes pipeline:
       - sign-oriented individual gradients are first cast to float32,
         matching the saved intermediate file used by the original pipeline
       - column-center individual G1-G10
       - column-center fixed 375 reference G1-G10
       - solve orthogonal Procrustes using all 10 components
       - no scaling
       - reflection allowed
       - no post-Procrustes sign flip

7. Save raw, sign-oriented, and Procrustes gradients plus full 10x10
   component correspondence QC.

IMPORTANT
---------
The current64 group gradient is computed ONLY as a QC/replication result.
It is NOT used as the individual-gradient reference.
"""

from pathlib import Path
import os
import re
import json
import hashlib
import warnings

# Keep numerical libraries from launching many nested threads.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import nibabel as nib

from brainspace.gradient import GradientMaps

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

ROOT_VALUE = os.environ.get("STRIATAL_ANALYSIS_ROOT")
if not ROOT_VALUE:
    raise RuntimeError("Set STRIATAL_ANALYSIS_ROOT to the private analysis root")
ROOT = Path(ROOT_VALUE).resolve()

CURRENT_ROOT = (
    ROOT /
    "17_current64_375matched_fingerprints"
)

STACK_FILE = (
    CURRENT_ROOT /
    "05_group" /
    "current64_fingerprints_z_2608x65.npy"
)

CURRENT_GROUP_MEAN_FILE = (
    CURRENT_ROOT /
    "05_group" /
    "group_mean_fingerprint_z_2608x65.npy"
)

MANIFEST_FILE = (
    CURRENT_ROOT /
    "04_qc" /
    "375matched_fingerprint_manifest.tsv"
)

# Frozen final 375 reference.
REF375_GRADIENT_FILE = (
    ROOT /
    "07_group_gradient" /
    "group_gradients.npy"
)

REF375_LAMBDA_FILE = (
    ROOT /
    "07_group_gradient" /
    "group_lambdas.npy"
)

# Exact validated 2847 -> 2608 row mapping.
ROW_MAPPING_FILE = (
    ROOT /
    "09_375HC_fingerprint_LOO_benchmark" /
    "01_inputs" /
    "row_mapping_2847_to_2608.npy"
)

# 375 voxel-coordinate table.
COORD375_FILE = (
    ROOT /
    "06_fingerprints" /
    "seed_voxel_coordinates.tsv"
)

# Exact current 2608 voxel order.
CURRENT2608_IJK_FILE = (
    ROOT /
    "07_fingerprints_pass64" /
    "final_analysis_mask_2608_voxel_ijk.npy"
)

# The 2608 analysis mask used in the current dataset.
CURRENT2608_MASK_FILE = Path(
    os.environ.get("STRIATAL_CURRENT2608_MASK", "inputs/current64_analysis_mask.nii.gz")
).resolve()

OUT_ROOT = (
    ROOT /
    "19_current64_375reference_gradients"
)

INPUT_OUT = OUT_ROOT / "01_reference"
SCRIPT_OUT = OUT_ROOT / "02_scripts"
QC_OUT = OUT_ROOT / "03_qc"
SUBJECT_OUT = OUT_ROOT / "04_subject_gradients"
GROUP_QC_OUT = OUT_ROOT / "05_current64_group_gradient_QC"

for d in [
    OUT_ROOT,
    INPUT_OUT,
    SCRIPT_OUT,
    QC_OUT,
    SUBJECT_OUT,
    GROUP_QC_OUT,
]:
    d.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================================
# FROZEN PARAMETERS
# =============================================================================

N_SUBJECTS = 64
N_VOXELS_OLD = 2847
N_VOXELS = 2608
N_ROIS = 65
N_COMPONENTS = 10

KERNEL = "normalized_angle"
APPROACH = "dm"
SPARSITY = 0.0
RANDOM_STATE = 0

# Frozen reference SHA256 values already audited.
EXPECTED_REF_GRADIENT_SHA256 = (
    "db28fd70f558776c4c4d57dbc9b1534ffdcf381009507a9a052898b446d42a37"
)

EXPECTED_REF_LAMBDA_SHA256 = (
    "b36c304dbbbd9305bd429b76f515190529da892191e2eeec618ee77373719c94"
)


# =============================================================================
# HELPERS
# =============================================================================

def sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def get_subject_id(text):
    m = re.search(
        r"(sub-[A-Za-z0-9]+)",
        str(text)
    )

    if m is None:
        raise ValueError(
            f"Cannot parse subject ID from: {text}"
        )

    return m.group(1)


def safe_corr(a, b):
    a = np.asarray(
        a,
        dtype=np.float64
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=np.float64
    ).reshape(-1)

    good = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    a = a[good]
    b = b[good]

    if (
        len(a) < 3
        or
        np.std(a) == 0
        or
        np.std(b) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def component_corr_matrix(A, B):
    """
    A: n_voxels x p
    B: n_voxels x q

    Return:
        p x q Pearson correlation matrix.
    """

    A = np.asarray(
        A,
        dtype=np.float64
    )

    B = np.asarray(
        B,
        dtype=np.float64
    )

    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"Voxel mismatch in correlation matrix: "
            f"{A.shape} vs {B.shape}"
        )

    Ac = (
        A
        -
        A.mean(
            axis=0,
            keepdims=True
        )
    )

    Bc = (
        B
        -
        B.mean(
            axis=0,
            keepdims=True
        )
    )

    ssA = np.sqrt(
        np.sum(
            Ac ** 2,
            axis=0
        )
    )

    ssB = np.sqrt(
        np.sum(
            Bc ** 2,
            axis=0
        )
    )

    denom = np.outer(
        ssA,
        ssB
    )

    C = np.full(
        (
            A.shape[1],
            B.shape[1]
        ),
        np.nan,
        dtype=np.float64
    )

    good = denom > 0

    numerator = (
        Ac.T @ Bc
    )

    C[good] = (
        numerator[good]
        /
        denom[good]
    )

    return np.clip(
        C,
        -1.0,
        1.0
    )


def fit_gradient(X):
    """
    EXACT final-375 GradientMaps call.

    Note:
    diffusion_time is intentionally NOT explicitly passed,
    matching the final frozen 375 script that was exactly reproduced.
    """

    X = np.asarray(
        X,
        dtype=np.float64
    )

    if X.shape != (
        N_VOXELS,
        N_ROIS
    ):
        raise ValueError(
            f"Fingerprint shape is not "
            f"{N_VOXELS} x {N_ROIS}: "
            f"{X.shape}"
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "Fingerprint contains NaN/Inf."
        )

    gm = GradientMaps(
        n_components=N_COMPONENTS,
        approach=APPROACH,
        kernel=KERNEL,
        random_state=RANDOM_STATE,
    )

    gm.fit(
        X,
        sparsity=SPARSITY
    )

    grad = np.asarray(
        gm.gradients_,
        dtype=np.float64
    )

    lambdas = np.asarray(
        gm.lambdas_,
        dtype=np.float64
    ).reshape(-1)

    # Robust orientation check.
    if (
        grad.shape[0] != N_VOXELS
        and
        grad.shape[1] == N_VOXELS
    ):
        grad = grad.T

    if grad.shape != (
        N_VOXELS,
        N_COMPONENTS
    ):
        raise ValueError(
            f"Unexpected gradient shape: "
            f"{grad.shape}"
        )

    if len(lambdas) < N_COMPONENTS:
        raise ValueError(
            f"Not enough lambdas: "
            f"{lambdas.shape}"
        )

    lambdas = lambdas[
        :N_COMPONENTS
    ]

    if not np.all(
        np.isfinite(grad)
    ):
        raise ValueError(
            "Gradient contains NaN/Inf."
        )

    if not np.all(
        np.isfinite(lambdas)
    ):
        raise ValueError(
            "Lambdas contain NaN/Inf."
        )

    return grad, lambdas


def orient_to_reference(grad, ref):
    """
    EXACT logic of final 375 individual-gradient script:

    Compare Gi with reference Gi.
    If r < 0, multiply Gi by -1.

    No component permutation.
    """

    grad = np.asarray(
        grad,
        dtype=np.float64
    ).copy()

    ref = np.asarray(
        ref,
        dtype=np.float64
    )

    if grad.shape != ref.shape:
        raise ValueError(
            f"Gradient/reference shape mismatch: "
            f"{grad.shape} vs {ref.shape}"
        )

    raw_C = component_corr_matrix(
        grad,
        ref
    )

    signs = np.ones(
        N_COMPONENTS,
        dtype=int
    )

    raw_diag = np.diag(
        raw_C
    ).copy()

    for k in range(
        N_COMPONENTS
    ):

        r = raw_diag[k]

        if (
            np.isfinite(r)
            and
            r < 0
        ):
            grad[:, k] *= -1.0
            signs[k] = -1

    oriented_C = component_corr_matrix(
        grad,
        ref
    )

    return (
        grad,
        signs,
        raw_C,
        oriented_C
    )


def orthogonal_procrustes_375_style(
    sign_oriented_grad,
    ref
):
    """
    EXACT final 375 Procrustes logic:

    1. sign-oriented individual gradient had previously been saved float32.
       We explicitly reproduce that precision roundtrip here.
    2. column-center individual.
    3. column-center reference.
    4. solve R minimizing ||A R - B||_F with orthogonal R.
    5. no scale.
    6. reflection allowed.
    7. no post-Procrustes sign flipping.
    """

    # Replicate original intermediate .npy float32 storage.
    g_saved_float32 = np.asarray(
        sign_oriented_grad,
        dtype=np.float32
    )

    g = g_saved_float32.astype(
        np.float64
    )

    ref = np.asarray(
        ref,
        dtype=np.float64
    )

    if g.shape != (
        N_VOXELS,
        N_COMPONENTS
    ):
        raise ValueError(
            f"Procrustes individual shape error: "
            f"{g.shape}"
        )

    if ref.shape != (
        N_VOXELS,
        N_COMPONENTS
    ):
        raise ValueError(
            f"Procrustes reference shape error: "
            f"{ref.shape}"
        )

    # Column-center reference.
    ref_c = (
        ref
        -
        ref.mean(
            axis=0,
            keepdims=True
        )
    )

    # Column-center individual.
    g_c = (
        g
        -
        g.mean(
            axis=0,
            keepdims=True
        )
    )

    # Orthogonal Procrustes.
    M = (
        g_c.T
        @
        ref_c
    )

    U, singular_values, Vt = np.linalg.svd(
        M,
        full_matrices=False
    )

    R = (
        U
        @
        Vt
    )

    g_proc = (
        g_c
        @
        R
    )

    return (
        g_proc,
        R,
        singular_values,
        g_saved_float32
    )


def save_vector_to_nifti(
    vec,
    mask_bool,
    mask_img,
    out_file
):
    vec = np.asarray(
        vec,
        dtype=np.float32
    )

    if vec.shape != (
        int(mask_bool.sum()),
    ):
        raise ValueError(
            f"NIfTI vector length mismatch: "
            f"{vec.shape} vs {int(mask_bool.sum())}"
        )

    vol = np.zeros(
        mask_bool.shape,
        dtype=np.float32
    )

    vol[
        mask_bool
    ] = vec

    hdr = mask_img.header.copy()
    hdr.set_data_dtype(
        np.float32
    )

    img = nib.Nifti1Image(
        vol,
        mask_img.affine,
        hdr
    )

    nib.save(
        img,
        str(out_file)
    )


def save_corr_matrix_tsv(
    C,
    out_file
):
    df = pd.DataFrame(
        C,
        index=[
            f"individual_G{k}"
            for k in range(
                1,
                N_COMPONENTS + 1
            )
        ],
        columns=[
            f"reference_G{k}"
            for k in range(
                1,
                N_COMPONENTS + 1
            )
        ],
    )

    df.to_csv(
        out_file,
        sep="\t"
    )


# =============================================================================
# START
# =============================================================================

print("=" * 110)
print("CURRENT64 EXACT 375-REFERENCE INDIVIDUAL GRADIENTS + PROCRUSTES")
print("=" * 110)

print()
print("Gradient parameters:")
print(f"  n_components = {N_COMPONENTS}")
print(f"  approach      = {APPROACH}")
print(f"  kernel        = {KERNEL}")
print(f"  sparsity      = {SPARSITY}")
print(f"  random_state  = {RANDOM_STATE}")
print("  diffusion_time explicitly passed = False")

print()
print("Procrustes:")
print("  components = G1-G10")
print("  individual column-centered = True")
print("  reference column-centered  = True")
print("  scaling = False")
print("  reflection allowed = True")
print("  post-Procrustes sign flip = False")


# =============================================================================
# STEP 1. REQUIRED INPUTS
# =============================================================================

print()
print("=" * 110)
print("STEP 1. INPUT EXISTENCE")
print("=" * 110)

required = [
    STACK_FILE,
    CURRENT_GROUP_MEAN_FILE,
    MANIFEST_FILE,
    REF375_GRADIENT_FILE,
    REF375_LAMBDA_FILE,
    ROW_MAPPING_FILE,
    COORD375_FILE,
    CURRENT2608_IJK_FILE,
    CURRENT2608_MASK_FILE,
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
# STEP 2. VERIFY FROZEN 375 REFERENCE
# =============================================================================

print()
print("=" * 110)
print("STEP 2. VERIFY FROZEN 375 REFERENCE")
print("=" * 110)

grad_sha = sha256(
    REF375_GRADIENT_FILE
)

lambda_sha = sha256(
    REF375_LAMBDA_FILE
)

print(
    "group_gradients SHA256 =",
    grad_sha
)

print(
    "group_lambdas SHA256   =",
    lambda_sha
)

if (
    grad_sha
    !=
    EXPECTED_REF_GRADIENT_SHA256
):
    raise RuntimeError(
        "Frozen 375 group_gradients.npy SHA256 changed."
    )

if (
    lambda_sha
    !=
    EXPECTED_REF_LAMBDA_SHA256
):
    raise RuntimeError(
        "Frozen 375 group_lambdas.npy SHA256 changed."
    )

ref375 = np.load(
    REF375_GRADIENT_FILE,
    allow_pickle=False
).astype(
    np.float64
)

ref375_lambdas = np.load(
    REF375_LAMBDA_FILE,
    allow_pickle=False
).astype(
    np.float64
).reshape(-1)

print(
    "375 gradients shape =",
    ref375.shape
)

print(
    "375 lambdas shape   =",
    ref375_lambdas.shape
)

if ref375.shape != (
    N_VOXELS_OLD,
    N_COMPONENTS
):
    raise RuntimeError(
        f"Unexpected frozen reference shape: "
        f"{ref375.shape}"
    )

if ref375_lambdas.shape != (
    N_COMPONENTS,
):
    raise RuntimeError(
        f"Unexpected frozen lambda shape: "
        f"{ref375_lambdas.shape}"
    )


# =============================================================================
# STEP 3. VERIFY EXACT 2847 -> 2608 MAPPING
# =============================================================================

print()
print("=" * 110)
print("STEP 3. VERIFY EXACT 2847 -> 2608 VOXEL MAPPING")
print("=" * 110)

row_map = np.load(
    ROW_MAPPING_FILE,
    allow_pickle=False
).astype(
    np.int64
).reshape(-1)

current_ijk = np.load(
    CURRENT2608_IJK_FILE,
    allow_pickle=False
).astype(
    np.int64
)

coords375_df = pd.read_csv(
    COORD375_FILE,
    sep="\t"
)

for col in [
    "i",
    "j",
    "k",
]:
    if col not in coords375_df.columns:
        raise RuntimeError(
            f"375 coordinate table missing column: "
            f"{col}"
        )

coords375_ijk = (
    coords375_df[
        [
            "i",
            "j",
            "k",
        ]
    ]
    .to_numpy(
        dtype=np.int64
    )
)

print(
    "row mapping shape =",
    row_map.shape
)

print(
    "row mapping min/max =",
    int(row_map.min()),
    int(row_map.max())
)

print(
    "unique mapped rows =",
    len(
        np.unique(
            row_map
        )
    )
)

print(
    "375 IJK shape =",
    coords375_ijk.shape
)

print(
    "current IJK shape =",
    current_ijk.shape
)

if row_map.shape != (
    N_VOXELS,
):
    raise RuntimeError(
        f"row_mapping shape error: "
        f"{row_map.shape}"
    )

if len(
    np.unique(
        row_map
    )
) != N_VOXELS:
    raise RuntimeError(
        "row_mapping contains duplicate rows."
    )

if (
    row_map.min() < 0
    or
    row_map.max() >= N_VOXELS_OLD
):
    raise RuntimeError(
        "row_mapping index out of range."
    )

if coords375_ijk.shape != (
    N_VOXELS_OLD,
    3
):
    raise RuntimeError(
        f"375 coordinate shape error: "
        f"{coords375_ijk.shape}"
    )

if current_ijk.shape != (
    N_VOXELS,
    3
):
    raise RuntimeError(
        f"Current IJK shape error: "
        f"{current_ijk.shape}"
    )

mapped_375_ijk = (
    coords375_ijk[
        row_map
    ]
)

ijk_exact = np.array_equal(
    mapped_375_ijk,
    current_ijk
)

print(
    "375[row_mapping] IJK == current2608 IJK:",
    ijk_exact
)

if not ijk_exact:
    raise RuntimeError(
        "EXACT 2847 -> 2608 IJK mapping failed."
    )

# Verify the NIfTI mask iteration order is also exactly current2608 order.
mask_img = nib.load(
    str(
        CURRENT2608_MASK_FILE
    )
)

mask_bool = (
    mask_img.get_fdata()
    >
    0
)

n_mask = int(
    mask_bool.sum()
)

mask_ijk = np.column_stack(
    np.where(
        mask_bool
    )
).astype(
    np.int64
)

print(
    "2608 NIfTI mask voxels =",
    n_mask
)

print(
    "NIfTI mask IJK exact order match =",
    np.array_equal(
        mask_ijk,
        current_ijk
    )
)

if n_mask != N_VOXELS:
    raise RuntimeError(
        f"Current analysis mask has "
        f"{n_mask} voxels, expected {N_VOXELS}."
    )

if not np.array_equal(
    mask_ijk,
    current_ijk
):
    raise RuntimeError(
        "NIfTI mask voxel order does not match "
        "current2608 fingerprint order."
    )

# Map frozen 375 reference into exact common 2608 rows.
ref2608 = (
    ref375[
        row_map,
        :
    ]
)

if ref2608.shape != (
    N_VOXELS,
    N_COMPONENTS
):
    raise RuntimeError(
        f"Mapped reference shape error: "
        f"{ref2608.shape}"
    )

np.save(
    INPUT_OUT /
    "375_group_gradients_mapped_2608x10.npy",
    ref2608
)

np.save(
    INPUT_OUT /
    "row_mapping_2847_to_2608.npy",
    row_map
)

np.save(
    INPUT_OUT /
    "current2608_voxel_ijk.npy",
    current_ijk
)

# Save mapped reference NIfTI maps G1-G10.
for k in range(
    N_COMPONENTS
):
    save_vector_to_nifti(
        ref2608[:, k],
        mask_bool,
        mask_img,
        INPUT_OUT /
        f"375_reference_G{k+1}_mapped2608.nii.gz"
    )

print(
    "Mapped 375 reference shape =",
    ref2608.shape
)


# =============================================================================
# STEP 4. CURRENT64 STACK + SUBJECT ORDER EXACT VALIDATION
# =============================================================================

print()
print("=" * 110)
print("STEP 4. CURRENT64 STACK + SUBJECT ORDER EXACT VALIDATION")
print("=" * 110)

stack = np.load(
    STACK_FILE,
    mmap_mode="r",
    allow_pickle=False
)

print(
    "Stack shape =",
    stack.shape
)

if stack.shape != (
    N_SUBJECTS,
    N_VOXELS,
    N_ROIS
):
    raise RuntimeError(
        f"Current64 stack shape error: "
        f"{stack.shape}"
    )

if not np.all(
    np.isfinite(
        stack
    )
):
    raise RuntimeError(
        "Current64 stack contains NaN/Inf."
    )

manifest = pd.read_csv(
    MANIFEST_FILE,
    sep="\t",
    dtype=str
)

subject_col = None

for c in [
    "subject",
    "subject_id",
    "participant_id",
    "sub",
]:

    if c in manifest.columns:
        subject_col = c
        break

if subject_col is None:

    for c in manifest.columns:

        values = (
            manifest[c]
            .fillna("")
            .astype(str)
        )

        if (
            values
            .str.contains(
                r"sub-[A-Za-z0-9]+",
                regex=True
            )
            .sum()
            >=
            N_SUBJECTS
        ):
            subject_col = c
            break

if subject_col is None:
    raise RuntimeError(
        "Could not identify subject column in manifest."
    )

subjects = [
    get_subject_id(x)
    for x in manifest[
        subject_col
    ].tolist()
]

print(
    "Manifest subject column =",
    subject_col
)

print(
    "Manifest N =",
    len(subjects)
)

print(
    "Manifest unique N =",
    len(
        set(
            subjects
        )
    )
)

if len(subjects) != N_SUBJECTS:
    raise RuntimeError(
        f"Manifest has {len(subjects)} subjects."
    )

if len(set(subjects)) != N_SUBJECTS:
    raise RuntimeError(
        "Manifest subject IDs are not unique."
    )

# -------------------------------------------------------------------------
# Locate subject-specific current64 fingerprints robustly.
#
# IMPORTANT:
# Do NOT assume the legacy filename suffix *_fingerprint_z_Nx65.npy.
# The current exact-375-matched workspace may use a different filename.
#
# A valid individual fingerprint must:
#   1. be an .npy file;
#   2. contain the subject ID in its full path;
#   3. have exact shape (2608, 65);
#   4. not be a group stack / group mean / mapping / gradient output.
# -------------------------------------------------------------------------

print()
print("Searching subject-specific 2608 x 65 fingerprint arrays ...")

all_npy_candidates = list(
    CURRENT_ROOT.rglob("*.npy")
)

print(
    "Total .npy files under current fingerprint workspace =",
    len(all_npy_candidates)
)

excluded_name_tokens = [
    "current64_fingerprints_z_2608x65",
    "group_mean",
    "group_stack",
    "row_mapping",
    "voxel_ijk",
    "gradient",
    "lambda",
]

fp_by_subject = {
    sub: []
    for sub in subjects
}

for p in all_npy_candidates:

    p_lower = p.name.lower()

    if any(
        token in p_lower
        for token in excluded_name_tokens
    ):
        continue

    # Identify which current subject this file belongs to.
    matched_sub = None

    for sub in subjects:
        if sub in str(p):
            matched_sub = sub
            break

    if matched_sub is None:
        continue

    # Shape test is the decisive criterion.
    try:
        a = np.load(
            p,
            mmap_mode="r",
            allow_pickle=False
        )
    except Exception:
        continue

    if a.shape == (
        N_VOXELS,
        N_ROIS
    ):
        fp_by_subject[
            matched_sub
        ].append(
            p
        )

print()
print("Candidate fingerprint counts by subject:")
print("-" * 90)

for sub in subjects:
    print(
        f"{sub}: {len(fp_by_subject[sub])}"
    )

order_check_rows = []

for i, sub in enumerate(
    subjects
):

    hits = fp_by_subject.get(
        sub,
        []
    )

    if len(hits) == 0:

        print()
        print("=" * 90)
        print(
            f"NO 2608x65 INDIVIDUAL FILE FOUND FOR {sub}"
        )
        print("=" * 90)

        print(
            "Files containing this subject ID:"
        )

        subject_related = [
            p
            for p in all_npy_candidates
            if sub in str(p)
        ]

        for p in subject_related:

            try:
                a = np.load(
                    p,
                    mmap_mode="r",
                    allow_pickle=False
                )

                print(
                    f"{p} | shape={a.shape}"
                )

            except Exception as e:

                print(
                    f"{p} | unreadable: {e}"
                )

        raise RuntimeError(
            f"{sub}: no subject-specific "
            f"{N_VOXELS}x{N_ROIS} fingerprint "
            f"was found under {CURRENT_ROOT}."
        )

    if len(hits) > 1:

        print()
        print("=" * 90)
        print(
            f"MULTIPLE 2608x65 CANDIDATES FOR {sub}"
        )
        print("=" * 90)

        for p in hits:
            print(p)

        # Resolve using exact equality to the authoritative stack row.
        exact_hits = []

        stack_row = np.asarray(
            stack[i]
        )

        for p in hits:

            X_candidate = np.load(
                p,
                allow_pickle=False
            )

            if np.array_equal(
                stack_row,
                X_candidate
            ):
                exact_hits.append(
                    p
                )

        if len(exact_hits) != 1:

            raise RuntimeError(
                f"{sub}: found {len(hits)} shape-matched "
                f"candidates, but {len(exact_hits)} exactly "
                f"match stack row {i}."
            )

        individual_file = exact_hits[0]

    else:

        individual_file = hits[0]

    X_file = np.load(
        individual_file,
        allow_pickle=False
    )

    if X_file.shape != (
        N_VOXELS,
        N_ROIS
    ):
        raise RuntimeError(
            f"{sub}: individual fingerprint shape "
            f"{X_file.shape}"
        )

    stack_row = np.asarray(
        stack[i]
    )

    exact_equal = np.array_equal(
        stack_row,
        X_file
    )

    max_diff = float(
        np.max(
            np.abs(
                stack_row.astype(
                    np.float64
                )
                -
                X_file.astype(
                    np.float64
                )
            )
        )
    )

    order_check_rows.append({
        "stack_row":
            i,
        "subject":
            sub,
        "fingerprint_file":
            str(
                individual_file
            ),
        "exact_array_equal":
            exact_equal,
        "max_abs_diff":
            max_diff,
    })

    if not exact_equal:
        raise RuntimeError(
            f"{sub}: stack row {i} does not exactly "
            f"equal its individual fingerprint. "
            f"max diff={max_diff}"
        )

order_check_df = pd.DataFrame(
    order_check_rows
)

order_check_df.to_csv(
    QC_OUT /
    "current64_stack_subject_order_exact_qc.tsv",
    sep="\t",
    index=False
)

print()
print(
    "Exact stack-row vs individual-file matches =",
    int(
        order_check_df[
            "exact_array_equal"
        ].sum()
    ),
    "/",
    N_SUBJECTS
)

print()
print("Resolved subject fingerprint files:")
print("-" * 90)

for _, r in order_check_df.iterrows():
    print(
        f"{int(r['stack_row']):02d}  "
        f"{r['subject']}  "
        f"{r['fingerprint_file']}"
    )


# =============================================================================
# STEP 5. GROUP-MEAN FC CHECK
# =============================================================================

print()
print("=" * 110)
print("STEP 5. CURRENT64 GROUP-MEAN FINGERPRINT CHECK")
print("=" * 110)

saved_group_mean = np.load(
    CURRENT_GROUP_MEAN_FILE,
    allow_pickle=False
)

if saved_group_mean.shape != (
    N_VOXELS,
    N_ROIS
):
    raise RuntimeError(
        f"Saved group mean shape error: "
        f"{saved_group_mean.shape}"
    )

# Reproduce NumPy float32 mean behavior from the current stack.
recomputed_group_mean = np.mean(
    np.asarray(
        stack
    ),
    axis=0
)

group_mean_max_diff = float(
    np.max(
        np.abs(
            recomputed_group_mean.astype(
                np.float64
            )
            -
            saved_group_mean.astype(
                np.float64
            )
        )
    )
)

print(
    "Saved group mean shape =",
    saved_group_mean.shape
)

print(
    "Recomputed-vs-saved max abs diff =",
    group_mean_max_diff
)

if group_mean_max_diff > 1e-6:
    raise RuntimeError(
        "Current64 saved group mean does not match "
        "the current64 stack."
    )


# =============================================================================
# STEP 6. CURRENT64 GROUP GRADIENT — QC ONLY
# =============================================================================

print()
print("=" * 110)
print("STEP 6. CURRENT64 GROUP GRADIENT — QC ONLY")
print("=" * 110)

print(
    "NOTE: this group gradient is NOT used as the "
    "individual-gradient reference."
)

group_raw, group_lambdas = fit_gradient(
    saved_group_mean
)

(
    group_sign,
    group_signs,
    group_raw_C,
    group_sign_C,
) = orient_to_reference(
    group_raw,
    ref2608
)

(
    group_proc,
    group_R,
    group_singular_values,
    group_sign_float32,
) = orthogonal_procrustes_375_style(
    group_sign,
    ref2608
)

group_proc_C = component_corr_matrix(
    group_proc,
    ref2608
)

np.save(
    GROUP_QC_OUT /
    "current64_group_gradients_raw.npy",
    group_raw.astype(
        np.float32
    )
)

np.save(
    GROUP_QC_OUT /
    "current64_group_gradients_sign_oriented.npy",
    group_sign_float32
)

np.save(
    GROUP_QC_OUT /
    "current64_group_gradients_procrustes.npy",
    group_proc.astype(
        np.float32
    )
)

np.save(
    GROUP_QC_OUT /
    "current64_group_lambdas.npy",
    group_lambdas.astype(
        np.float32
    )
)

np.save(
    GROUP_QC_OUT /
    "current64_group_procrustes_rotation.npy",
    group_R.astype(
        np.float64
    )
)

save_corr_matrix_tsv(
    group_raw_C,
    GROUP_QC_OUT /
    "current64_group_raw_vs_375ref_10x10.tsv"
)

save_corr_matrix_tsv(
    group_sign_C,
    GROUP_QC_OUT /
    "current64_group_sign_vs_375ref_10x10.tsv"
)

save_corr_matrix_tsv(
    group_proc_C,
    GROUP_QC_OUT /
    "current64_group_procrustes_vs_375ref_10x10.tsv"
)

for k in range(
    3
):
    save_vector_to_nifti(
        group_proc[:, k],
        mask_bool,
        mask_img,
        GROUP_QC_OUT /
        f"current64_group_G{k+1}_procrustes.nii.gz"
    )

print(
    "Group G1 raw same-index r       =",
    float(
        group_raw_C[0, 0]
    )
)

print(
    "Group G1 sign-oriented r        =",
    float(
        group_sign_C[0, 0]
    )
)

print(
    "Group G1 Procrustes r           =",
    float(
        group_proc_C[0, 0]
    )
)


# =============================================================================
# STEP 7. INDIVIDUAL GRADIENTS
# =============================================================================

print()
print("=" * 110)
print("STEP 7. CURRENT64 INDIVIDUAL GRADIENTS")
print("=" * 110)

records = []
failed = []
corr_long_rows = []

for i, sub in enumerate(
    subjects,
    start=1
):

    print()
    print(
        f"[{i:02d}/{N_SUBJECTS:02d}] "
        f"{sub}"
    )

    try:

        X = np.asarray(
            stack[
                i - 1
            ],
            dtype=np.float64
        )

        if X.shape != (
            N_VOXELS,
            N_ROIS
        ):
            raise ValueError(
                f"Fingerprint shape error: "
                f"{X.shape}"
            )

        if not np.all(
            np.isfinite(X)
        ):
            raise ValueError(
                "Fingerprint contains NaN/Inf."
            )

        # ---------------------------------------------------------------------
        # A. RAW INDIVIDUAL GRADIENT
        # ---------------------------------------------------------------------

        raw_grad, lambdas = fit_gradient(
            X
        )

        # ---------------------------------------------------------------------
        # B. FINAL-375 STYLE COMPONENT-WISE SIGN ORIENTATION
        # ---------------------------------------------------------------------

        (
            sign_grad,
            signs,
            raw_C,
            sign_C,
        ) = orient_to_reference(
            raw_grad,
            ref2608
        )

        raw_diag = np.diag(
            raw_C
        )

        sign_diag = np.diag(
            sign_C
        )

        # ---------------------------------------------------------------------
        # C. FINAL-375 STYLE PROCRUSTES
        # ---------------------------------------------------------------------

        (
            proc_grad,
            R,
            singular_values,
            sign_grad_float32,
        ) = orthogonal_procrustes_375_style(
            sign_grad,
            ref2608
        )

        proc_C = component_corr_matrix(
            proc_grad,
            ref2608
        )

        proc_diag = np.diag(
            proc_C
        )

        orthogonality_error = float(
            np.max(
                np.abs(
                    R.T
                    @
                    R
                    -
                    np.eye(
                        N_COMPONENTS
                    )
                )
            )
        )

        det_R = float(
            np.linalg.det(
                R
            )
        )

        # Raw component matching QC.
        raw_best_ref = (
            np.nanargmax(
                np.abs(
                    raw_C
                ),
                axis=1
            )
        )

        n_identity_raw = int(
            np.sum(
                raw_best_ref
                ==
                np.arange(
                    N_COMPONENTS
                )
            )
        )

        n_identity_raw_first3 = int(
            np.sum(
                raw_best_ref[
                    :3
                ]
                ==
                np.arange(
                    3
                )
            )
        )

        # ---------------------------------------------------------------------
        # D. SAVE SUBJECT OUTPUTS
        # ---------------------------------------------------------------------

        sub_dir = (
            SUBJECT_OUT /
            sub
        )

        sub_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            sub_dir /
            f"{sub}_gradients_raw.npy",
            raw_grad.astype(
                np.float32
            )
        )

        # This is the exact precision level that enters Procrustes.
        np.save(
            sub_dir /
            f"{sub}_gradients_sign_oriented.npy",
            sign_grad_float32
        )

        np.save(
            sub_dir /
            f"{sub}_gradients_procrustes.npy",
            proc_grad.astype(
                np.float32
            )
        )

        np.save(
            sub_dir /
            f"{sub}_lambdas.npy",
            lambdas.astype(
                np.float32
            )
        )

        np.save(
            sub_dir /
            f"{sub}_procrustes_rotation.npy",
            R.astype(
                np.float64
            )
        )

        np.save(
            sub_dir /
            f"{sub}_procrustes_singular_values.npy",
            singular_values.astype(
                np.float64
            )
        )

        save_corr_matrix_tsv(
            raw_C,
            sub_dir /
            f"{sub}_raw_vs_375ref_10x10.tsv"
        )

        save_corr_matrix_tsv(
            sign_C,
            sub_dir /
            f"{sub}_sign_vs_375ref_10x10.tsv"
        )

        save_corr_matrix_tsv(
            proc_C,
            sub_dir /
            f"{sub}_procrustes_vs_375ref_10x10.tsv"
        )

        pd.DataFrame(
            proc_grad[
                :,
                :3
            ],
            columns=[
                "G1",
                "G2",
                "G3",
            ]
        ).to_csv(
            sub_dir /
            f"{sub}_G1G2G3_procrustes.tsv",
            sep="\t",
            index=False
        )

        # Convenience NIfTI outputs from final Procrustes result.
        for k in range(
            3
        ):
            save_vector_to_nifti(
                proc_grad[:, k],
                mask_bool,
                mask_img,
                sub_dir /
                f"{sub}_G{k+1}_procrustes_current_ref.nii.gz"
            )

        # ---------------------------------------------------------------------
        # E. LONG 10x10 COMPONENT QC
        # ---------------------------------------------------------------------

        for stage, C in [
            (
                "raw",
                raw_C
            ),
            (
                "sign_oriented",
                sign_C
            ),
            (
                "procrustes",
                proc_C
            ),
        ]:

            for gi in range(
                N_COMPONENTS
            ):

                for gj in range(
                    N_COMPONENTS
                ):

                    corr_long_rows.append({
                        "subject":
                            sub,
                        "stage":
                            stage,
                        "individual_component":
                            gi + 1,
                        "reference_component":
                            gj + 1,
                        "r":
                            float(
                                C[
                                    gi,
                                    gj
                                ]
                            ),
                        "abs_r":
                            float(
                                abs(
                                    C[
                                        gi,
                                        gj
                                    ]
                                )
                            ),
                        "same_index":
                            int(
                                gi == gj
                            ),
                    })

        # ---------------------------------------------------------------------
        # F. SUBJECT SUMMARY
        # ---------------------------------------------------------------------

        rec = {
            "subject":
                sub,

            "stack_row":
                i - 1,

            "status":
                "PASS",

            "n_voxels":
                N_VOXELS,

            "n_rois":
                N_ROIS,

            "n_components":
                N_COMPONENTS,

            "raw_G1_ref_r":
                float(
                    raw_diag[0]
                ),

            "raw_G2_ref_r":
                float(
                    raw_diag[1]
                ),

            "raw_G3_ref_r":
                float(
                    raw_diag[2]
                ),

            "sign_G1_ref_r":
                float(
                    sign_diag[0]
                ),

            "sign_G2_ref_r":
                float(
                    sign_diag[1]
                ),

            "sign_G3_ref_r":
                float(
                    sign_diag[2]
                ),

            "procrustes_G1_ref_r":
                float(
                    proc_diag[0]
                ),

            "procrustes_G2_ref_r":
                float(
                    proc_diag[1]
                ),

            "procrustes_G3_ref_r":
                float(
                    proc_diag[2]
                ),

            "raw_G1_best_reference":
                int(
                    raw_best_ref[0]
                    +
                    1
                ),

            "raw_G2_best_reference":
                int(
                    raw_best_ref[1]
                    +
                    1
                ),

            "raw_G3_best_reference":
                int(
                    raw_best_ref[2]
                    +
                    1
                ),

            "raw_first3_identity_matches":
                n_identity_raw_first3,

            "raw_all10_identity_matches":
                n_identity_raw,

            "procrustes_det_R":
                det_R,

            "procrustes_orthogonality_error":
                orthogonality_error,
        }

        for k in range(
            N_COMPONENTS
        ):

            rec[
                f"lambda_{k+1}"
            ] = float(
                lambdas[k]
            )

            rec[
                f"sign_G{k+1}"
            ] = int(
                signs[k]
            )

            rec[
                f"raw_sameindex_r_G{k+1}"
            ] = float(
                raw_diag[k]
            )

            rec[
                f"sign_sameindex_r_G{k+1}"
            ] = float(
                sign_diag[k]
            )

            rec[
                f"procrustes_sameindex_r_G{k+1}"
            ] = float(
                proc_diag[k]
            )

            rec[
                f"raw_best_reference_G{k+1}"
            ] = int(
                raw_best_ref[k]
                +
                1
            )

        records.append(
            rec
        )

        print(
            "PASS | "
            f"G1 raw={raw_diag[0]:.3f} | "
            f"sign={sign_diag[0]:.3f} | "
            f"proc={proc_diag[0]:.3f} | "
            f"raw_best_G1=G{raw_best_ref[0]+1} | "
            f"identity10={n_identity_raw}/10"
        )

    except Exception as e:

        print(
            f"FAIL | {e}"
        )

        failed.append({
            "subject":
                sub,
            "stack_row":
                i - 1,
            "status":
                "FAIL",
            "error":
                str(
                    e
                ),
        })


# =============================================================================
# STEP 8. SAVE QC TABLES
# =============================================================================

print()
print("=" * 110)
print("STEP 8. SAVE QC TABLES")
print("=" * 110)

summary_df = pd.DataFrame(
    records
)

failed_df = pd.DataFrame(
    failed
)

corr_long_df = pd.DataFrame(
    corr_long_rows
)

summary_file = (
    QC_OUT /
    "current64_375reference_gradient_summary.tsv"
)

failed_file = (
    QC_OUT /
    "current64_375reference_gradient_failed.tsv"
)

corr_long_file = (
    QC_OUT /
    "current64_component_correspondence_10x10_long.tsv"
)

summary_df.to_csv(
    summary_file,
    sep="\t",
    index=False
)

failed_df.to_csv(
    failed_file,
    sep="\t",
    index=False
)

corr_long_df.to_csv(
    corr_long_file,
    sep="\t",
    index=False
)

print(
    "Summary:",
    summary_file
)

print(
    "Failed:",
    failed_file
)

print(
    "10x10 QC:",
    corr_long_file
)


# =============================================================================
# STEP 9. FINAL VALIDATION
# =============================================================================

print()
print("=" * 110)
print("STEP 9. FINAL VALIDATION")
print("=" * 110)

n_success = len(
    records
)

n_failed = len(
    failed
)

if n_success > 0:

    g1_raw = summary_df[
        "raw_G1_ref_r"
    ].to_numpy(
        dtype=float
    )

    g1_sign = summary_df[
        "sign_G1_ref_r"
    ].to_numpy(
        dtype=float
    )

    g1_proc = summary_df[
        "procrustes_G1_ref_r"
    ].to_numpy(
        dtype=float
    )

    identity_first3 = summary_df[
        "raw_first3_identity_matches"
    ].to_numpy(
        dtype=float
    )

    identity_all10 = summary_df[
        "raw_all10_identity_matches"
    ].to_numpy(
        dtype=float
    )

    ortho_error = summary_df[
        "procrustes_orthogonality_error"
    ].to_numpy(
        dtype=float
    )

else:

    g1_raw = np.array([])
    g1_sign = np.array([])
    g1_proc = np.array([])
    identity_first3 = np.array([])
    identity_all10 = np.array([])
    ortho_error = np.array([])

print()
print("SUBJECT STATUS")
print("-" * 90)

print(
    "Input subjects                =",
    N_SUBJECTS
)

print(
    "PASS                          =",
    n_success
)

print(
    "FAIL                          =",
    n_failed
)

print()
print("REFERENCE")
print("-" * 90)

print(
    "375 reference original        =",
    ref375.shape
)

print(
    "375 reference mapped          =",
    ref2608.shape
)

print(
    "Exact mapped IJK match        =",
    ijk_exact
)

print(
    "NIfTI vector-order match      =",
    np.array_equal(
        mask_ijk,
        current_ijk
    )
)

print()
print("CURRENT64 FC")
print("-" * 90)

print(
    "Stack shape                   =",
    stack.shape
)

print(
    "Exact subject-order matches   =",
    int(
        order_check_df[
            "exact_array_equal"
        ].sum()
    ),
    "/",
    N_SUBJECTS
)

print(
    "Group mean max abs diff       =",
    group_mean_max_diff
)

if n_success > 0:

    print()
    print("INDIVIDUAL G1 CORRESPONDENCE TO FIXED 375 REFERENCE")
    print("-" * 90)

    print(
        "Raw mean r                    =",
        float(
            np.nanmean(
                g1_raw
            )
        )
    )

    print(
        "Sign-oriented mean r          =",
        float(
            np.nanmean(
                g1_sign
            )
        )
    )

    print(
        "Procrustes mean r              =",
        float(
            np.nanmean(
                g1_proc
            )
        )
    )

    print(
        "Procrustes median r            =",
        float(
            np.nanmedian(
                g1_proc
            )
        )
    )

    print(
        "Procrustes min/max r           =",
        float(
            np.nanmin(
                g1_proc
            )
        ),
        "/",
        float(
            np.nanmax(
                g1_proc
            )
        )
    )

    print()
    print("COMPONENT-MATCHING QC")
    print("-" * 90)

    print(
        "Mean raw identity matches G1-3 =",
        float(
            np.mean(
                identity_first3
            )
        ),
        "/ 3"
    )

    print(
        "Mean raw identity matches G1-10=",
        float(
            np.mean(
                identity_all10
            )
        ),
        "/ 10"
    )

    print(
        "Max Procrustes orthogonality error =",
        float(
            np.max(
                ortho_error
            )
        )
    )

print()
print("CURRENT64 GROUP GRADIENT QC — NOT USED AS REFERENCE")
print("-" * 90)

print(
    "Group G1 raw r                =",
    float(
        group_raw_C[
            0,
            0
        ]
    )
)

print(
    "Group G1 sign-oriented r      =",
    float(
        group_sign_C[
            0,
            0
        ]
    )
)

print(
    "Group G1 Procrustes r         =",
    float(
        group_proc_C[
            0,
            0
        ]
    )
)

# Structural PASS criteria.
FINAL_PASS = bool(
    n_success == N_SUBJECTS
    and
    n_failed == 0
    and
    ijk_exact
    and
    np.array_equal(
        mask_ijk,
        current_ijk
    )
    and
    int(
        order_check_df[
            "exact_array_equal"
        ].sum()
    )
    ==
    N_SUBJECTS
    and
    group_mean_max_diff
    <=
    1e-6
    and
    np.all(
        np.isfinite(
            summary_df[
                [
                    "sign_G1_ref_r",
                    "procrustes_G1_ref_r",
                    "procrustes_orthogonality_error",
                ]
            ].to_numpy(
                dtype=float
            )
        )
    )
)

print()
print(
    "FINAL_CHECK =",
    "PASS"
    if FINAL_PASS
    else
    "FAIL/REVIEW"
)


# =============================================================================
# STEP 10. PARAMETERS + FINAL REPORT
# =============================================================================

params = {
    "analysis":
        "current64 exact 375-reference individual gradients",

    "main_reference":
        str(
            REF375_GRADIENT_FILE
        ),

    "main_reference_sha256":
        grad_sha,

    "reference_original_shape":
        list(
            ref375.shape
        ),

    "reference_mapping":
        str(
            ROW_MAPPING_FILE
        ),

    "reference_mapped_shape":
        list(
            ref2608.shape
        ),

    "current_stack":
        str(
            STACK_FILE
        ),

    "current_stack_shape":
        list(
            stack.shape
        ),

    "subject_manifest":
        str(
            MANIFEST_FILE
        ),

    "gradient_parameters": {
        "n_components":
            N_COMPONENTS,
        "approach":
            APPROACH,
        "kernel":
            KERNEL,
        "sparsity":
            SPARSITY,
        "random_state":
            RANDOM_STATE,
        "diffusion_time_explicitly_passed":
            False,
        "extra_roi_zscoring":
            False,
    },

    "sign_orientation": {
        "method":
            "component-wise sign orientation to fixed 375 mapped reference",
        "component_permutation":
            False,
    },

    "procrustes": {
        "n_align_components":
            N_COMPONENTS,
        "individual_column_centered":
            True,
        "reference_column_centered":
            True,
        "orthogonal":
            True,
        "scaling":
            False,
        "reflection_allowed":
            True,
        "post_procrustes_sign_flip":
            False,
        "individual_precision_entering_procrustes":
            "float32 saved-intermediate equivalent",
    },

    "current64_group_gradient_role":
        "QC only; never used as individual reference",

    "n_success":
        n_success,

    "n_failed":
        n_failed,

    "final_check":
        (
            "PASS"
            if FINAL_PASS
            else
            "FAIL/REVIEW"
        ),
}

with open(
    QC_OUT /
    "current64_375reference_gradient_parameters.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        params,
        f,
        indent=2,
        ensure_ascii=False
    )

report_lines = [
    "=" * 110,
    "FINAL RESULT",
    "=" * 110,
    "",
    "CURRENT64 EXACT 375-REFERENCE INDIVIDUAL GRADIENTS + PROCRUSTES",
    "-" * 110,
    "",
    f"Input subjects                   = {N_SUBJECTS}",
    f"PASS                             = {n_success}",
    f"FAIL                             = {n_failed}",
    "",
    "REFERENCE",
    "-" * 110,
    f"375 original reference           = {ref375.shape}",
    f"Mapped reference                 = {ref2608.shape}",
    f"Exact IJK mapping                = {ijk_exact}",
    f"NIfTI voxel-order match          = {np.array_equal(mask_ijk, current_ijk)}",
    "",
    "CURRENT64 INPUT",
    "-" * 110,
    f"FC stack                         = {stack.shape}",
    f"Exact stack subject order        = {int(order_check_df['exact_array_equal'].sum())}/{N_SUBJECTS}",
    f"Group mean max abs diff          = {group_mean_max_diff}",
]

if n_success > 0:

    report_lines.extend([
        "",
        "INDIVIDUAL G1 vs FIXED 375 REFERENCE",
        "-" * 110,
        f"Raw mean r                       = {float(np.nanmean(g1_raw))}",
        f"Sign-oriented mean r             = {float(np.nanmean(g1_sign))}",
        f"Procrustes mean r                 = {float(np.nanmean(g1_proc))}",
        f"Procrustes median r               = {float(np.nanmedian(g1_proc))}",
        f"Procrustes min r                  = {float(np.nanmin(g1_proc))}",
        f"Procrustes max r                  = {float(np.nanmax(g1_proc))}",
        "",
        "COMPONENT QC",
        "-" * 110,
        f"Mean identity matches G1-G3       = {float(np.mean(identity_first3))}/3",
        f"Mean identity matches G1-G10      = {float(np.mean(identity_all10))}/10",
        f"Max orthogonality error           = {float(np.max(ortho_error))}",
    ])

report_lines.extend([
    "",
    "CURRENT64 GROUP-GRADIENT QC ONLY",
    "-" * 110,
    f"Group G1 raw r                   = {float(group_raw_C[0,0])}",
    f"Group G1 sign-oriented r         = {float(group_sign_C[0,0])}",
    f"Group G1 Procrustes r            = {float(group_proc_C[0,0])}",
    "",
    f"FINAL_CHECK = {'PASS' if FINAL_PASS else 'FAIL/REVIEW'}",
    "",
    "QC summary:",
    str(summary_file),
    "",
    "Full 10x10 component QC:",
    str(corr_long_file),
    "",
    "Mapped 375 reference:",
    str(
        INPUT_OUT /
        "375_group_gradients_mapped_2608x10.npy"
    ),
    "",
    "Individual outputs:",
    str(
        SUBJECT_OUT
    ),
    "",
])

report = "\n".join(
    report_lines
)

with open(
    QC_OUT /
    "FINAL_RESULT.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        report
        +
        "\n"
    )

print()
print(report)
