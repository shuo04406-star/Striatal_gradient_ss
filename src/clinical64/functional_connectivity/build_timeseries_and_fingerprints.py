#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import hashlib
import json
import re
import shutil
import traceback

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker


# =============================================================================
# PATHS
# =============================================================================

ROOT_VALUE = os.environ.get("STRIATAL_ANALYSIS_ROOT")
if not ROOT_VALUE:
    raise RuntimeError("Set STRIATAL_ANALYSIS_ROOT to the private analysis root")
ROOT = Path(ROOT_VALUE).resolve()

BASE = (
    ROOT /
    "17_current64_375matched_fingerprints"
)

INPUT_DIR = BASE / "01_inputs"
TS_DIR = BASE / "02_timeseries"
FP_DIR = BASE / "03_fingerprints"
QC_DIR = BASE / "04_qc"
GROUP_DIR = BASE / "05_group"

for d in [
    INPUT_DIR,
    TS_DIR,
    FP_DIR,
    QC_DIR,
    GROUP_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True
    )


# -----------------------------------------------------------------------------
# New 375-matched denoised BOLD
# -----------------------------------------------------------------------------

BOLD_MANIFEST = (
    ROOT /
    "16_current64_375matched_denoising" /
    "04_qc" /
    "375matched_denoised_bold_manifest.tsv"
)


# -----------------------------------------------------------------------------
# ROI manifest used by the 375 pipeline
# -----------------------------------------------------------------------------

ROI_MANIFEST = (
    ROOT /
    "01_inputs" /
    "prepared_to_current_ref" /
    "roi65_manifest_current_ref.tsv"
)


# -----------------------------------------------------------------------------
# 375 whole-striatum seed space = 2847 voxels
# -----------------------------------------------------------------------------

WHOLE_STRIATUM_MASK = (
    ROOT /
    "01_inputs" /
    "prepared_to_current_ref" /
    "whole_striatum_seed_mask_current_ref.nii.gz"
)


# -----------------------------------------------------------------------------
# Exact old-375 -> current 2608 row mapping
# -----------------------------------------------------------------------------

ROW_MAPPING = (
    ROOT /
    "09_375HC_fingerprint_LOO_benchmark" /
    "01_inputs" /
    "row_mapping_2847_to_2608.npy"
)


# -----------------------------------------------------------------------------
# Previously frozen current 2608 ordering
# -----------------------------------------------------------------------------

EXPECTED_2608_IJK = (
    ROOT /
    "07_fingerprints_pass64" /
    "final_analysis_mask_2608_voxel_ijk.npy"
)

EXPECTED_2608_FLAT = (
    ROOT /
    "07_fingerprints_pass64" /
    "final_analysis_mask_2608_voxel_indices.npy"
)


# -----------------------------------------------------------------------------
# Previous ROI-name sequence, used only for independent QC
# -----------------------------------------------------------------------------

OLD_ROI_NAMES = (
    ROOT /
    "06_roi_timeseries_pass64" /
    "roi_names_65.txt"
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

QC_TSV = (
    QC_DIR /
    "fingerprint_qc_375matched.tsv"
)

MANIFEST_OUT = (
    QC_DIR /
    "375matched_fingerprint_manifest.tsv"
)

SUMMARY_JSON = (
    QC_DIR /
    "375matched_fingerprint_summary.json"
)

SUMMARY_TXT = (
    QC_DIR /
    "FINAL_VALIDATION.txt"
)

ROI_NAMES_OUT = (
    INPUT_DIR /
    "roi_names_65_exact_order.txt"
)

IJK2847_OUT = (
    INPUT_DIR /
    "whole_striatum_2847_voxel_ijk.npy"
)

FLAT2847_OUT = (
    INPUT_DIR /
    "whole_striatum_2847_voxel_indices.npy"
)

IJK2608_OUT = (
    INPUT_DIR /
    "mapped_2608_voxel_ijk.npy"
)

FLAT2608_OUT = (
    INPUT_DIR /
    "mapped_2608_voxel_indices.npy"
)

ROWMAP_COPY = (
    INPUT_DIR /
    "row_mapping_2847_to_2608.npy"
)


# =============================================================================
# HELPERS
# =============================================================================

def subject_sort_key(x):

    m = re.search(
        r"sub-(\d+)",
        str(x)
    )

    if m:
        return int(
            m.group(1)
        )

    return 10**9


def sha256(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def zscore_time_axis(x):
    """
    EXACT logic used by the old 375 fingerprint pipeline.

    x = time x features
    """

    x = np.asarray(
        x,
        dtype=np.float64
    )

    mean = np.nanmean(
        x,
        axis=0,
        keepdims=True
    )

    std = np.nanstd(
        x,
        axis=0,
        ddof=1,
        keepdims=True
    )

    std[
        std == 0
    ] = np.nan

    z = (
        x - mean
    ) / std

    z = np.nan_to_num(
        z,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return z


def make_fingerprint(
    seed_ts,
    roi_ts
):
    """
    EXACT 375 logic:
        z-score time axis
        Pearson correlation
        Fisher-z
    """

    if seed_ts.shape[0] != roi_ts.shape[0]:
        raise ValueError(
            "Timepoint mismatch: "
            f"seed={seed_ts.shape}, "
            f"ROI={roi_ts.shape}"
        )

    seed_z = zscore_time_axis(
        seed_ts
    )

    roi_z = zscore_time_axis(
        roi_ts
    )

    T = seed_z.shape[0]

    if T < 2:
        raise ValueError(
            f"Too few timepoints: {T}"
        )

    r = (
        seed_z.T @ roi_z
    ) / float(
        T - 1
    )

    # Old 375 pipeline clipping
    r = np.clip(
        r,
        -0.999999,
        0.999999
    )

    z = np.arctanh(
        r
    )

    return (
        r.astype(np.float32),
        z.astype(np.float32)
    )


def same_affine(
    a,
    b
):

    return bool(
        np.allclose(
            a,
            b,
            atol=1e-4
        )
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

print("=" * 110)
print("CURRENT64 EXACT 375-MATCHED TIME SERIES + FINGERPRINT PIPELINE")
print("=" * 110)

required = [
    BOLD_MANIFEST,
    ROI_MANIFEST,
    WHOLE_STRIATUM_MASK,
    ROW_MAPPING,
    EXPECTED_2608_IJK,
    EXPECTED_2608_FLAT,
]

for p in required:

    print(
        f"{p} | exists={p.exists()}"
    )

    if not p.exists():

        raise FileNotFoundError(
            p
        )


# =============================================================================
# SUBJECTS
# =============================================================================

bold_df = pd.read_csv(
    BOLD_MANIFEST,
    sep="\t"
)

required_cols = [
    "subject",
    "status",
    "output_file",
]

for c in required_cols:

    if c not in bold_df.columns:

        raise RuntimeError(
            f"BOLD manifest missing column: {c}"
        )


bold_df = bold_df[
    bold_df["status"].astype(str)
    == "OK"
].copy()

bold_df["_sort"] = (
    bold_df["subject"]
    .map(
        subject_sort_key
    )
)

bold_df = (
    bold_df
    .sort_values("_sort")
    .drop(
        columns="_sort"
    )
    .reset_index(
        drop=True
    )
)

if len(
    bold_df
) != 64:

    raise RuntimeError(
        "Expected 64 OK subjects, "
        f"found {len(bold_df)}"
    )


for p in bold_df[
    "output_file"
].astype(str):

    if not Path(p).exists():

        raise FileNotFoundError(
            p
        )


subjects = (
    bold_df[
        "subject"
    ]
    .astype(str)
    .tolist()
)

print()
print("Subjects =", len(subjects))
print("First    =", subjects[0])
print("Last     =", subjects[-1])


# =============================================================================
# ROI ORDER
# =============================================================================

roi = pd.read_csv(
    ROI_MANIFEST,
    sep="\t"
)

needed = [
    "roi_id",
    "roi_name",
    "prepared_mask",
]

for c in needed:

    if c not in roi.columns:

        raise RuntimeError(
            f"ROI manifest missing column: {c}"
        )


roi = (
    roi
    .sort_values(
        "roi_id",
        kind="stable"
    )
    .reset_index(
        drop=True
    )
)


if len(
    roi
) != 65:

    raise RuntimeError(
        f"Expected 65 ROIs, found {len(roi)}"
    )


if roi[
    "roi_id"
].duplicated().any():

    raise RuntimeError(
        "Duplicate roi_id detected"
    )


roi_names = (
    roi[
        "roi_name"
    ]
    .astype(str)
    .tolist()
)


ROI_NAMES_OUT.write_text(
    "\n".join(
        roi_names
    ) + "\n",
    encoding="utf-8"
)


roi_order_match_old_file = None

if OLD_ROI_NAMES.exists():

    old_names = [
        x.strip()
        for x in OLD_ROI_NAMES.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]

    roi_order_match_old_file = (
        old_names == roi_names
    )

    if not roi_order_match_old_file:

        raise RuntimeError(
            "ROI ORDER MISMATCH with "
            "06_roi_timeseries_pass64/roi_names_65.txt"
        )


print()
print("=" * 110)
print("ROI ORDER")
print("=" * 110)

print("ROI count =", len(roi_names))
print(
    "ROI order vs previous roi_names_65.txt =",
    roi_order_match_old_file
)


# =============================================================================
# REFERENCE GRID
# =============================================================================

first_bold = Path(
    bold_df.loc[
        0,
        "output_file"
    ]
)

first_img = nib.load(
    str(first_bold)
)

if len(
    first_img.shape
) != 4:

    raise RuntimeError(
        f"First BOLD is not 4D: {first_img.shape}"
    )


ref_shape = tuple(
    first_img.shape[:3]
)

ref_affine = (
    first_img.affine.copy()
)


print()
print("=" * 110)
print("REFERENCE GRID")
print("=" * 110)

print("Reference BOLD =", first_bold)
print("Shape          =", ref_shape)
print(
    "Voxel size     =",
    first_img.header.get_zooms()[:3]
)


# =============================================================================
# ROI MASK GEOMETRY
# =============================================================================

roi_maskers = []

for i, row in roi.iterrows():

    name = str(
        row["roi_name"]
    )

    p = Path(
        str(
            row["prepared_mask"]
        )
    )

    if not p.exists():

        raise FileNotFoundError(
            p
        )

    img = nib.load(
        str(p)
    )

    if tuple(
        img.shape[:3]
    ) != ref_shape:

        raise RuntimeError(
            f"ROI shape mismatch: "
            f"{name} | "
            f"{img.shape[:3]} vs {ref_shape}"
        )

    if not same_affine(
        img.affine,
        ref_affine
    ):

        raise RuntimeError(
            f"ROI affine mismatch: {name}"
        )

    nvox = int(
        np.count_nonzero(
            img.get_fdata() > 0
        )
    )

    if nvox == 0:

        raise RuntimeError(
            f"Empty ROI: {name}"
        )

    # Same extraction settings as old 375 pipeline
    masker = NiftiMasker(
        mask_img=str(p),
        standardize=False,
        detrend=False,
        smoothing_fwhm=None,
        verbose=0
    )

    roi_maskers.append(
        masker
    )


# =============================================================================
# 2847 WHOLE STRIATUM
# =============================================================================

whole_img = nib.load(
    str(
        WHOLE_STRIATUM_MASK
    )
)

if tuple(
    whole_img.shape[:3]
) != ref_shape:

    raise RuntimeError(
        "Whole-striatum shape mismatch: "
        f"{whole_img.shape[:3]} vs {ref_shape}"
    )

if not same_affine(
    whole_img.affine,
    ref_affine
):

    raise RuntimeError(
        "Whole-striatum affine mismatch"
    )


whole_mask = (
    whole_img.get_fdata()
    > 0
)

n_2847 = int(
    whole_mask.sum()
)

if n_2847 != 2847:

    raise RuntimeError(
        "Old whole-striatum mask "
        f"must contain 2847 voxels, got {n_2847}"
    )


old_idx = np.where(
    whole_mask
)

old_ijk = np.column_stack(
    old_idx
).astype(
    np.int16
)

old_flat = np.ravel_multi_index(
    old_idx,
    dims=ref_shape
).astype(
    np.int64
)


np.save(
    IJK2847_OUT,
    old_ijk
)

np.save(
    FLAT2847_OUT,
    old_flat
)


seed_masker = NiftiMasker(
    mask_img=str(
        WHOLE_STRIATUM_MASK
    ),
    standardize=False,
    detrend=False,
    smoothing_fwhm=None,
    verbose=0
)


# =============================================================================
# EXACT 2847 -> 2608 MAPPING
# =============================================================================

row_map = np.load(
    ROW_MAPPING,
    allow_pickle=False
).astype(
    np.int64
).reshape(-1)


if row_map.shape != (
    2608,
):

    raise RuntimeError(
        f"row_mapping shape wrong: {row_map.shape}"
    )


if row_map.min() < 0:

    raise RuntimeError(
        "row_mapping has negative indices"
    )


if row_map.max() >= 2847:

    raise RuntimeError(
        "row_mapping index exceeds 2847 space"
    )


if len(
    np.unique(
        row_map
    )
) != 2608:

    raise RuntimeError(
        "row_mapping contains duplicate rows"
    )


mapped_ijk = (
    old_ijk[
        row_map
    ]
)

mapped_flat = (
    old_flat[
        row_map
    ]
)


expected_ijk = np.load(
    EXPECTED_2608_IJK,
    allow_pickle=False
).astype(
    int
)

expected_flat = np.load(
    EXPECTED_2608_FLAT,
    allow_pickle=False
).astype(
    np.int64
).reshape(-1)


if expected_ijk.shape != (
    2608,
    3
):

    raise RuntimeError(
        "Expected IJK file has wrong shape: "
        f"{expected_ijk.shape}"
    )


if expected_flat.shape != (
    2608,
):

    raise RuntimeError(
        "Expected flat-index file has wrong shape: "
        f"{expected_flat.shape}"
    )


ijk_order_match = bool(
    np.array_equal(
        mapped_ijk.astype(int),
        expected_ijk
    )
)

flat_order_match = bool(
    np.array_equal(
        mapped_flat,
        expected_flat
    )
)


if not ijk_order_match:

    raise RuntimeError(
        "2847 -> 2608 IJK ORDER MISMATCH"
    )


if not flat_order_match:

    raise RuntimeError(
        "2847 -> 2608 FLAT ORDER MISMATCH"
    )


np.save(
    IJK2608_OUT,
    mapped_ijk
)

np.save(
    FLAT2608_OUT,
    mapped_flat
)

shutil.copy2(
    ROW_MAPPING,
    ROWMAP_COPY
)


print()
print("=" * 110)
print("STRIATAL VOXEL ORDER")
print("=" * 110)

print("Old whole striatum =", n_2847)
print("Mapped voxels       =", len(row_map))
print("row_map min         =", int(row_map.min()))
print("row_map max         =", int(row_map.max()))
print("row_map unique      =", len(np.unique(row_map)))
print("EXACT IJK MATCH     =", ijk_order_match)
print("EXACT FLAT MATCH    =", flat_order_match)


# =============================================================================
# SUBJECT LOOP
# =============================================================================

qc_rows = []
manifest_rows = []

stack_z_2608 = []
stack_r_2608 = []

success = []
failed = []


print()
print("=" * 110)
print("EXTRACTING TIME SERIES AND BUILDING FINGERPRINTS")
print("=" * 110)


for number, row in bold_df.iterrows():

    sub = str(
        row["subject"]
    )

    bold_path = Path(
        str(
            row["output_file"]
        )
    )

    print()
    print(
        f"[{number + 1:02d}/64] {sub}"
    )

    sub_ts_dir = (
        TS_DIR /
        sub
    )

    sub_fp_dir = (
        FP_DIR /
        sub
    )

    sub_ts_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sub_fp_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    qc = {
        "subject":
            sub,

        "status":
            "FAIL",

        "T":
            "",

        "n_roi":
            65,

        "n_striatum_2847":
            2847,

        "n_striatum_2608":
            2608,

        "zero_variance_roi":
            "",

        "zero_variance_striatum_2847":
            "",

        "zero_variance_striatum_2608":
            "",

        "nonfinite_roi_values":
            "",

        "nonfinite_striatum_2847_values":
            "",

        "nonfinite_striatum_2608_values":
            "",

        "max_abs_direct_vs_mapped_r_difference":
            "",

        "fingerprint_2847_shape":
            "",

        "fingerprint_2608_shape":
            "",

        "error":
            "",
    }


    try:

        # ---------------------------------------------------------------------
        # BOLD validation
        # ---------------------------------------------------------------------

        img = nib.load(
            str(
                bold_path
            )
        )

        if len(
            img.shape
        ) != 4:

            raise RuntimeError(
                f"BOLD is not 4D: {img.shape}"
            )


        if tuple(
            img.shape[:3]
        ) != ref_shape:

            raise RuntimeError(
                f"BOLD shape mismatch: "
                f"{img.shape[:3]} vs {ref_shape}"
            )


        if not same_affine(
            img.affine,
            ref_affine
        ):

            raise RuntimeError(
                "BOLD affine mismatch"
            )


        T = int(
            img.shape[3]
        )

        qc["T"] = T


        # ---------------------------------------------------------------------
        # 2847 striatal voxel time series
        #
        # Same NiftiMasker extraction used by old 375 pipeline:
        # time x 2847
        # ---------------------------------------------------------------------

        seed_ts_2847 = (
            seed_masker
            .fit_transform(
                str(
                    bold_path
                )
            )
            .astype(
                np.float32
            )
        )


        if seed_ts_2847.shape != (
            T,
            2847
        ):

            raise RuntimeError(
                "2847 striatal TS wrong shape: "
                f"{seed_ts_2847.shape}"
            )


        # ---------------------------------------------------------------------
        # Map the actual TIME SERIES into exact current 2608 order
        # ---------------------------------------------------------------------

        seed_ts_2608 = (
            seed_ts_2847[
                :,
                row_map
            ]
            .astype(
                np.float32
            )
        )


        if seed_ts_2608.shape != (
            T,
            2608
        ):

            raise RuntimeError(
                "2608 striatal TS wrong shape: "
                f"{seed_ts_2608.shape}"
            )


        # ---------------------------------------------------------------------
        # Extract 65 ROI mean time series
        #
        # Exact ROI-manifest order.
        # ---------------------------------------------------------------------

        roi_cols = []

        for roi_name, masker in zip(
            roi_names,
            roi_maskers
        ):

            vox_ts = (
                masker
                .fit_transform(
                    str(
                        bold_path
                    )
                )
            )

            if vox_ts.ndim != 2:

                raise RuntimeError(
                    f"{roi_name}: "
                    f"unexpected TS shape "
                    f"{vox_ts.shape}"
                )


            if vox_ts.shape[0] != T:

                raise RuntimeError(
                    f"{roi_name}: "
                    "timepoint mismatch"
                )


            if vox_ts.shape[1] == 0:

                raise RuntimeError(
                    f"{roi_name}: "
                    "zero voxels after masking"
                )


            mean_ts = (
                vox_ts.mean(
                    axis=1
                )
            )

            roi_cols.append(
                mean_ts
            )


        roi_ts = (
            np.column_stack(
                roi_cols
            )
            .astype(
                np.float32
            )
        )


        if roi_ts.shape != (
            T,
            65
        ):

            raise RuntimeError(
                "ROI TS wrong shape: "
                f"{roi_ts.shape}"
            )


        # ---------------------------------------------------------------------
        # QC time series
        # ---------------------------------------------------------------------

        n_nonfinite_roi = int(
            np.sum(
                ~np.isfinite(
                    roi_ts
                )
            )
        )

        n_nonfinite_2847 = int(
            np.sum(
                ~np.isfinite(
                    seed_ts_2847
                )
            )
        )

        n_nonfinite_2608 = int(
            np.sum(
                ~np.isfinite(
                    seed_ts_2608
                )
            )
        )


        qc[
            "nonfinite_roi_values"
        ] = n_nonfinite_roi

        qc[
            "nonfinite_striatum_2847_values"
        ] = n_nonfinite_2847

        qc[
            "nonfinite_striatum_2608_values"
        ] = n_nonfinite_2608


        if (
            n_nonfinite_roi > 0
            or
            n_nonfinite_2847 > 0
            or
            n_nonfinite_2608 > 0
        ):

            raise RuntimeError(
                "Nonfinite values detected "
                "in extracted time series"
            )


        roi_sd = np.std(
            roi_ts,
            axis=0,
            ddof=1
        )

        seed_sd_2847 = np.std(
            seed_ts_2847,
            axis=0,
            ddof=1
        )

        seed_sd_2608 = np.std(
            seed_ts_2608,
            axis=0,
            ddof=1
        )


        zero_roi = int(
            np.sum(
                roi_sd < 1e-8
            )
        )

        zero_2847 = int(
            np.sum(
                seed_sd_2847 < 1e-8
            )
        )

        zero_2608 = int(
            np.sum(
                seed_sd_2608 < 1e-8
            )
        )


        qc[
            "zero_variance_roi"
        ] = zero_roi

        qc[
            "zero_variance_striatum_2847"
        ] = zero_2847

        qc[
            "zero_variance_striatum_2608"
        ] = zero_2608


        # Final analysis space should not contain dead ROI / voxel series.
        if zero_roi != 0:

            raise RuntimeError(
                f"Zero-variance ROI columns = {zero_roi}"
            )


        if zero_2608 != 0:

            raise RuntimeError(
                "Zero-variance voxels in final "
                f"2608 space = {zero_2608}"
            )


        # ---------------------------------------------------------------------
        # Save time series
        # ---------------------------------------------------------------------

        roi_ts_file = (
            sub_ts_dir /
            f"{sub}_roi_timeseries_Tx65.npy"
        )

        seed2847_file = (
            sub_ts_dir /
            f"{sub}_striatum_timeseries_Tx2847.npy"
        )

        seed2608_file = (
            sub_ts_dir /
            f"{sub}_striatum_timeseries_Tx2608.npy"
        )


        np.save(
            roi_ts_file,
            roi_ts
        )

        np.save(
            seed2847_file,
            seed_ts_2847
        )

        np.save(
            seed2608_file,
            seed_ts_2608
        )


        # ---------------------------------------------------------------------
        # 375-space fingerprint: 2847 x 65
        # ---------------------------------------------------------------------

        r_2847, z_2847 = (
            make_fingerprint(
                seed_ts_2847,
                roi_ts
            )
        )


        if r_2847.shape != (
            2847,
            65
        ):

            raise RuntimeError(
                f"2847 r shape wrong: {r_2847.shape}"
            )


        if z_2847.shape != (
            2847,
            65
        ):

            raise RuntimeError(
                f"2847 z shape wrong: {z_2847.shape}"
            )


        # ---------------------------------------------------------------------
        # Map old-375 fingerprint rows -> exact current 2608 order
        # ---------------------------------------------------------------------

        r_2608 = (
            r_2847[
                row_map,
                :
            ]
            .astype(
                np.float32
            )
        )

        z_2608 = (
            z_2847[
                row_map,
                :
            ]
            .astype(
                np.float32
            )
        )


        if r_2608.shape != (
            2608,
            65
        ):

            raise RuntimeError(
                f"Mapped r shape wrong: {r_2608.shape}"
            )


        if z_2608.shape != (
            2608,
            65
        ):

            raise RuntimeError(
                f"Mapped z shape wrong: {z_2608.shape}"
            )


        # ---------------------------------------------------------------------
        # Independent mathematical validation:
        #
        # Compute 2608 FC directly from mapped 2608 time series and verify
        # it equals mapping the original 2847 FC rows.
        # ---------------------------------------------------------------------

        r_direct_2608, z_direct_2608 = (
            make_fingerprint(
                seed_ts_2608,
                roi_ts
            )
        )


        maxdiff_r = float(
            np.max(
                np.abs(
                    r_direct_2608.astype(
                        np.float64
                    )
                    -
                    r_2608.astype(
                        np.float64
                    )
                )
            )
        )


        maxdiff_z = float(
            np.max(
                np.abs(
                    z_direct_2608.astype(
                        np.float64
                    )
                    -
                    z_2608.astype(
                        np.float64
                    )
                )
            )
        )


        qc[
            "max_abs_direct_vs_mapped_r_difference"
        ] = maxdiff_r


        if maxdiff_r > 1e-5:

            raise RuntimeError(
                "Direct 2608 r != mapped 2847 r: "
                f"maxdiff={maxdiff_r}"
            )


        if maxdiff_z > 1e-5:

            raise RuntimeError(
                "Direct 2608 z != mapped 2847 z: "
                f"maxdiff={maxdiff_z}"
            )


        # ---------------------------------------------------------------------
        # Final finite checks
        # ---------------------------------------------------------------------

        for name, matrix in [
            ("r_2847", r_2847),
            ("z_2847", z_2847),
            ("r_2608", r_2608),
            ("z_2608", z_2608),
        ]:

            if not np.all(
                np.isfinite(
                    matrix
                )
            ):

                raise RuntimeError(
                    f"{name} contains NaN/Inf"
                )


        # ---------------------------------------------------------------------
        # Save fingerprints
        # ---------------------------------------------------------------------

        r2847_file = (
            sub_fp_dir /
            f"{sub}_fingerprint_r_2847x65.npy"
        )

        z2847_file = (
            sub_fp_dir /
            f"{sub}_fingerprint_z_2847x65.npy"
        )

        r2608_file = (
            sub_fp_dir /
            f"{sub}_fingerprint_r_2608x65.npy"
        )

        z2608_file = (
            sub_fp_dir /
            f"{sub}_fingerprint_z_2608x65.npy"
        )


        np.save(
            r2847_file,
            r_2847
        )

        np.save(
            z2847_file,
            z_2847
        )

        np.save(
            r2608_file,
            r_2608
        )

        np.save(
            z2608_file,
            z_2608
        )


        qc[
            "fingerprint_2847_shape"
        ] = str(
            r_2847.shape
        )

        qc[
            "fingerprint_2608_shape"
        ] = str(
            r_2608.shape
        )

        qc[
            "status"
        ] = "PASS"


        success.append(
            sub
        )

        stack_r_2608.append(
            r_2608
        )

        stack_z_2608.append(
            z_2608
        )


        manifest_rows.append({
            "subject":
                sub,

            "T":
                T,

            "bold_file":
                str(
                    bold_path
                ),

            "roi_timeseries_Tx65":
                str(
                    roi_ts_file
                ),

            "striatum_timeseries_Tx2847":
                str(
                    seed2847_file
                ),

            "striatum_timeseries_Tx2608":
                str(
                    seed2608_file
                ),

            "fingerprint_r_2847x65":
                str(
                    r2847_file
                ),

            "fingerprint_z_2847x65":
                str(
                    z2847_file
                ),

            "fingerprint_r_2608x65":
                str(
                    r2608_file
                ),

            "fingerprint_z_2608x65":
                str(
                    z2608_file
                ),
        })


        print(
            "PASS | "
            f"T={T} | "
            f"ROI={roi_ts.shape} | "
            f"seed2847={seed_ts_2847.shape} | "
            f"seed2608={seed_ts_2608.shape} | "
            f"FP={z_2608.shape} | "
            f"maxdiff={maxdiff_r:.3e}"
        )


    except Exception as e:

        qc[
            "error"
        ] = str(e)

        failed.append({
            "subject":
                sub,

            "error":
                str(e),

            "traceback":
                traceback.format_exc(),
        })

        print(
            "FAIL |",
            e
        )


    qc_rows.append(
        qc
    )


# =============================================================================
# SAVE SUBJECT QC
# =============================================================================

qc_df = pd.DataFrame(
    qc_rows
)

qc_df.to_csv(
    QC_TSV,
    sep="\t",
    index=False
)


manifest_df = pd.DataFrame(
    manifest_rows
)

manifest_df.to_csv(
    MANIFEST_OUT,
    sep="\t",
    index=False
)


# =============================================================================
# FINAL GROUP ARRAYS
# =============================================================================

if len(
    success
) == 64:

    R = np.stack(
        stack_r_2608,
        axis=0
    ).astype(
        np.float32
    )

    Z = np.stack(
        stack_z_2608,
        axis=0
    ).astype(
        np.float32
    )


    if R.shape != (
        64,
        2608,
        65
    ):

        raise RuntimeError(
            f"R stack wrong shape: {R.shape}"
        )


    if Z.shape != (
        64,
        2608,
        65
    ):

        raise RuntimeError(
            f"Z stack wrong shape: {Z.shape}"
        )


    np.save(
        GROUP_DIR /
        "current64_fingerprints_r_2608x65.npy",
        R
    )

    np.save(
        GROUP_DIR /
        "current64_fingerprints_z_2608x65.npy",
        Z
    )


    group_mean_z = np.mean(
        Z,
        axis=0,
        dtype=np.float64
    ).astype(
        np.float32
    )


    # Same interpretation used previously:
    # average Fisher-z, then tanh for r-space representation
    group_mean_r = np.tanh(
        group_mean_z
    ).astype(
        np.float32
    )


    np.save(
        GROUP_DIR /
        "group_mean_fingerprint_z_2608x65.npy",
        group_mean_z
    )

    np.save(
        GROUP_DIR /
        "group_mean_fingerprint_r_2608x65.npy",
        group_mean_r
    )


# =============================================================================
# FINAL VALIDATION
# =============================================================================

pass_count = int(
    (
        qc_df[
            "status"
        ] == "PASS"
    ).sum()
)

fail_count = int(
    (
        qc_df[
            "status"
        ] != "PASS"
    ).sum()
)


all_final_shapes_ok = bool(
    pass_count == 64
    and
    all(
        str(x) == "(2608, 65)"
        for x in qc_df.loc[
            qc_df["status"] == "PASS",
            "fingerprint_2608_shape"
        ]
    )
)


all_zero_roi_ok = bool(
    pass_count == 64
    and
    (
        pd.to_numeric(
            qc_df[
                "zero_variance_roi"
            ],
            errors="coerce"
        ) == 0
    ).all()
)


all_zero_2608_ok = bool(
    pass_count == 64
    and
    (
        pd.to_numeric(
            qc_df[
                "zero_variance_striatum_2608"
            ],
            errors="coerce"
        ) == 0
    ).all()
)


all_nonfinite_ok = bool(
    pass_count == 64
    and
    (
        pd.to_numeric(
            qc_df[
                "nonfinite_roi_values"
            ],
            errors="coerce"
        ) == 0
    ).all()
    and
    (
        pd.to_numeric(
            qc_df[
                "nonfinite_striatum_2608_values"
            ],
            errors="coerce"
        ) == 0
    ).all()
)


final_check = bool(
    pass_count == 64
    and fail_count == 0
    and ijk_order_match
    and flat_order_match
    and all_final_shapes_ok
    and all_zero_roi_ok
    and all_zero_2608_ok
    and all_nonfinite_ok
    and (
        roi_order_match_old_file
        is not False
    )
)


summary = {
    "pipeline":
        "CURRENT64 exact 375-matched timeseries and fingerprint construction",

    "n_subjects_expected":
        64,

    "n_subjects_pass":
        pass_count,

    "n_subjects_fail":
        fail_count,

    "roi_count":
        65,

    "old_striatum_voxels":
        2847,

    "mapped_striatum_voxels":
        2608,

    "final_fingerprint_shape":
        [
            2608,
            65
        ],

    "roi_order_match_previous_current":
        roi_order_match_old_file,

    "exact_2847_to_2608_IJK_order_match":
        ijk_order_match,

    "exact_2847_to_2608_flat_order_match":
        flat_order_match,

    "bold_manifest":
        str(
            BOLD_MANIFEST
        ),

    "roi_manifest":
        str(
            ROI_MANIFEST
        ),

    "whole_striatum_mask":
        str(
            WHOLE_STRIATUM_MASK
        ),

    "row_mapping":
        str(
            ROW_MAPPING
        ),

    "sha256": {
        "bold_manifest":
            sha256(
                BOLD_MANIFEST
            ),

        "roi_manifest":
            sha256(
                ROI_MANIFEST
            ),

        "whole_striatum_mask":
            sha256(
                WHOLE_STRIATUM_MASK
            ),

        "row_mapping":
            sha256(
                ROW_MAPPING
            ),
    },

    "outputs": {
        "timeseries_dir":
            str(
                TS_DIR
            ),

        "fingerprint_dir":
            str(
                FP_DIR
            ),

        "group_dir":
            str(
                GROUP_DIR
            ),

        "qc_tsv":
            str(
                QC_TSV
            ),

        "manifest":
            str(
                MANIFEST_OUT
            ),
    },

    "FINAL_CHECK":
        "PASS"
        if final_check
        else "FAIL",

    "failed_subjects":
        failed,
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


T_values = pd.to_numeric(
    qc_df[
        "T"
    ],
    errors="coerce"
)


maxdiff_values = pd.to_numeric(
    qc_df[
        "max_abs_direct_vs_mapped_r_difference"
    ],
    errors="coerce"
)


txt = f"""
==============================================================================================================
STEP 4. FINAL VALIDATION
==============================================================================================================

CURRENT64 EXACT 375-MATCHED TIME SERIES + FC FINGERPRINTS
--------------------------------------------------------------------------------------------------------------

Input denoised subjects = {len(bold_df)}
PASS                    = {pass_count}
FAIL                    = {fail_count}

ROI ORDER
--------------------------------------------------------------------------------------------------------------
ROI count                         = 65
Previous ROI order match          = {roi_order_match_old_file}

STRIATAL ORDER
--------------------------------------------------------------------------------------------------------------
375 whole-striatum voxels         = 2847
Mapped final voxels               = 2608
Exact IJK order match             = {ijk_order_match}
Exact flat-index order match      = {flat_order_match}

TIME SERIES
--------------------------------------------------------------------------------------------------------------
ROI time series                   = T x 65
Old striatal time series          = T x 2847
Mapped striatal time series       = T x 2608

Timepoints min                    = {int(T_values.min()) if T_values.notna().any() else "NA"}
Timepoints median                 = {float(T_values.median()) if T_values.notna().any() else "NA"}
Timepoints max                    = {int(T_values.max()) if T_values.notna().any() else "NA"}

FINAL FC
--------------------------------------------------------------------------------------------------------------
Pearson-r fingerprint             = 2608 x 65
Fisher-z fingerprint              = 2608 x 65

Maximum direct-vs-mapped r diff   = {float(maxdiff_values.max()) if maxdiff_values.notna().any() else "NA"}

Final fingerprint shapes OK       = {all_final_shapes_ok}
Zero-variance ROI check           = {all_zero_roi_ok}
Zero-variance 2608 voxel check    = {all_zero_2608_ok}
Nonfinite check                   = {all_nonfinite_ok}

FINAL_CHECK = {"PASS" if final_check else "FAIL"}

QC:
{QC_TSV}

Fingerprint manifest:
{MANIFEST_OUT}

Group stack:
{GROUP_DIR / "current64_fingerprints_z_2608x65.npy"}

Group mean fingerprint:
{GROUP_DIR / "group_mean_fingerprint_z_2608x65.npy"}
""".strip() + "\n"


SUMMARY_TXT.write_text(
    txt,
    encoding="utf-8"
)

print()
print(txt)


if not final_check:

    raise SystemExit(
        1
    )
