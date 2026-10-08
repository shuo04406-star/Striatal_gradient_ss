#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# 避免BLAS内部多线程
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import importlib.util
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
import scipy


# =============================================================================
# PATHS
# =============================================================================

ROOT_VALUE = os.environ.get("STRIATAL_CLINICAL64_DENOISE_ROOT")
if not ROOT_VALUE:
    raise RuntimeError("Set STRIATAL_CLINICAL64_DENOISE_ROOT to a private output root")
ROOT = Path(ROOT_VALUE).resolve()

INPUT_DIR = ROOT / "01_inputs"
OUT_ROOT = ROOT / "03_denoised_bold"
QC_DIR = ROOT / "04_qc"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 375 actual denoising code
# -------------------------------------------------------------------------

OLD_SCRIPT = Path(
    os.environ.get(
        "STRIATAL_375_DENOISE_SCRIPT",
        Path(__file__).resolve().parents[2]
        / "reference375"
        / "preprocessing"
        / "denoise_selected_runs.py",
    )
).resolve()

# -------------------------------------------------------------------------
# Use the source manifest from the CURRENT pass64_v2 analysis.
#
# Important:
# This points to the ORIGINAL fMRIPrep preprocessed BOLD/confounds,
# not to the already-denoised pass64_v2 BOLD.
# -------------------------------------------------------------------------

CURRENT_V2_MANIFEST = Path(
    os.environ.get("STRIATAL_CURRENT64_MANIFEST", "inputs/current64_input_manifest.tsv")
).resolve()

# Independent 64-subject list check
GROUP_TABLE = Path(
    os.environ.get("STRIATAL_CURRENT64_GROUP_TABLE", "inputs/current64_group_table.tsv")
).resolve()

MATCHED_INPUT = (
    INPUT_DIR /
    "current64_input_for_375matched_denoising.tsv"
)

QC_FILE = (
    QC_DIR /
    "denoising_qc_375matched.tsv"
)

SUMMARY_FILE = (
    QC_DIR /
    "SUMMARY.txt"
)

OUTPUT_MANIFEST = (
    QC_DIR /
    "375matched_denoised_bold_manifest.tsv"
)


# =============================================================================
# HELPERS
# =============================================================================

def subject_key(s):

    try:
        return int(
            str(s)
            .replace("sub-", "")
        )
    except Exception:
        return 999999


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_old_module():

    spec = (
        importlib.util
        .spec_from_file_location(
            "denoise375_original",
            OLD_SCRIPT
        )
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            f"Cannot import {OLD_SCRIPT}"
        )

    mod = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        mod
    )

    return mod


# =============================================================================
# STEP 1. PREFLIGHT
# =============================================================================

print("=" * 110)
print("CURRENT64 -> EXACT 375-MATCHED DENOISING")
print("=" * 110)

for path in [
    OLD_SCRIPT,
    CURRENT_V2_MANIFEST,
    GROUP_TABLE,
]:

    print(
        path,
        "exists =",
        path.exists()
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# =============================================================================
# STEP 2. BUILD EXACT 64-SUBJECT INPUT MANIFEST
# =============================================================================

print("\n" + "=" * 110)
print("STEP 1. BUILD CURRENT64 INPUT MANIFEST")
print("=" * 110)

src = pd.read_csv(
    CURRENT_V2_MANIFEST,
    sep="\t",
    keep_default_na=False
)

required = [
    "subject",
    "run",
    "bold_file",
    "confounds_file",
]

missing_cols = [
    x
    for x in required
    if x not in src.columns
]

if missing_cols:

    raise RuntimeError(
        f"Missing columns in current input manifest: "
        f"{missing_cols}"
    )

src = src[
    required
].copy()

src["subject"] = (
    src["subject"]
    .astype(str)
    .str.strip()
)

src["run"] = (
    src["run"]
    .astype(str)
    .str.strip()
)

src.loc[
    src["run"].isin(
        [
            "",
            "nan",
            "NaN",
            "None",
        ]
    ),
    "run"
] = "NA"

src = (
    src
    .sort_values(
        "subject",
        key=lambda x:
        x.map(
            subject_key
        )
    )
    .reset_index(
        drop=True
    )
)

if len(src) != 64:

    raise RuntimeError(
        f"Expected 64 rows, found {len(src)}"
    )

if src["subject"].nunique() != 64:

    raise RuntimeError(
        "Subject IDs are not unique."
    )

# -------------------------------------------------------------------------
# Compare against independently stored final 64-subject table
# -------------------------------------------------------------------------

groups = pd.read_csv(
    GROUP_TABLE,
    sep="\t",
    keep_default_na=False
)

subject_col = None

for candidate in [
    "subject_id",
    "subject",
    "participant_id",
]:

    if candidate in groups.columns:

        subject_col = candidate
        break

if subject_col is None:

    raise RuntimeError(
        "Cannot identify subject column "
        "in final motion-QC table."
    )

expected_subjects = set(
    groups[
        subject_col
    ]
    .astype(str)
    .str.strip()
)

manifest_subjects = set(
    src[
        "subject"
    ]
)

if manifest_subjects != expected_subjects:

    print(
        "Only in manifest:",
        sorted(
            manifest_subjects
            -
            expected_subjects
        )
    )

    print(
        "Only in final-QC table:",
        sorted(
            expected_subjects
            -
            manifest_subjects
        )
    )

    raise RuntimeError(
        "Current denoising manifest does not "
        "match final 64-subject list."
    )

# -------------------------------------------------------------------------
# Check every original fMRIPrep input
# -------------------------------------------------------------------------

bad = []

for _, row in src.iterrows():

    for col in [
        "bold_file",
        "confounds_file",
    ]:

        p = Path(
            row[col]
        )

        if (
            not p.exists()
            or
            p.stat().st_size == 0
        ):

            bad.append(
                {
                    "subject":
                        row["subject"],

                    "field":
                        col,

                    "file":
                        str(p),
                }
            )

if bad:

    pd.DataFrame(
        bad
    ).to_csv(
        QC_DIR /
        "missing_input_files.tsv",
        sep="\t",
        index=False
    )

    raise RuntimeError(
        f"Missing/empty source files = "
        f"{len(bad)}"
    )

src.to_csv(
    MATCHED_INPUT,
    sep="\t",
    index=False
)

print(
    "N subjects =",
    len(src)
)

print(
    "Input manifest:",
    MATCHED_INPUT
)

print(
    "\nIMPORTANT:"
)

print(
    "BOLD inputs are ORIGINAL fMRIPrep "
    "desc-preproc_bold files."
)


# =============================================================================
# STEP 3. RECORD EXACT CODE / SOFTWARE PROVENANCE
# =============================================================================

old_sha = sha256_file(
    OLD_SCRIPT
)

provenance = {
    "purpose":
        "Current64 rerun using exact 375 denoising source code",

    "frozen_375_script":
        str(
            OLD_SCRIPT
        ),

    "frozen_375_script_sha256":
        old_sha,

    "current_source_manifest":
        str(
            CURRENT_V2_MANIFEST
        ),

    "N_subjects":
        64,

    "parameters": {
        "fd_cutoff_mm":
            0.5,

        "high_pass_Hz":
            0.01,

        "low_pass_Hz":
            0.08,

        "min_DOF_ratio":
            0.50,

        "motion":
            "fMRIPrep-generated available 24P columns",

        "aCompCor":
            "first up to 5 available a_comp_cor_* columns",

        "high_FD":
            "one spike regressor for each FD > 0.5 mm volume",

        "DOF":
            "N - rank(final nuisance matrix)",
    },

    "software": {
        "python":
            sys.version,

        "platform":
            platform.platform(),

        "numpy":
            np.__version__,

        "pandas":
            pd.__version__,

        "nibabel":
            nib.__version__,

        "nilearn":
            nilearn.__version__,

        "scipy":
            scipy.__version__,
    },
}

with open(
    QC_DIR /
    "provenance.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        provenance,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    "\n375 frozen script SHA256 =",
    old_sha
)


# =============================================================================
# STEP 4. LOAD EXACT 375 CODE
# =============================================================================

old = load_old_module()

for function in [
    "process_one",
    "get_motion_24p",
    "get_acompcor_first5",
    "make_fd_spikes",
    "compute_dof_ratio",
]:

    if not hasattr(
        old,
        function
    ):

        raise RuntimeError(
            f"Frozen 375 script missing "
            f"function: {function}"
        )

print(
    "\nExact 375 processing module loaded."
)


# =============================================================================
# STEP 5. RESUME SUPPORT
# =============================================================================

results = []

completed = set()

if QC_FILE.exists():

    try:

        previous = pd.read_csv(
            QC_FILE,
            sep="\t",
            keep_default_na=False
        )

        if (
            "subject"
            in
            previous.columns
            and
            "output_file"
            in
            previous.columns
        ):

            for _, r in previous.iterrows():

                p = Path(
                    str(
                        r[
                            "output_file"
                        ]
                    )
                )

                used = Path(
                    str(
                        r.get(
                            "used_confounds_file",
                            ""
                        )
                    )
                )

                if (
                    p.exists()
                    and
                    p.stat().st_size > 0
                    and
                    used.exists()
                ):

                    completed.add(
                        str(
                            r[
                                "subject"
                            ]
                        )
                    )

                    results.append(
                        r.to_dict()
                    )

    except Exception as e:

        print(
            "Previous QC could not be reused:",
            e
        )


if completed:

    print(
        "\nResume mode: already complete =",
        len(
            completed
        )
    )


# =============================================================================
# STEP 6. RUN 64 SUBJECTS USING EXACT 375 process_one()
# =============================================================================

print("\n" + "=" * 110)
print("STEP 2. RUN EXACT 375 DENOISING ON CURRENT64")
print("=" * 110)

to_run = src[
    ~src[
        "subject"
    ].isin(
        completed
    )
].copy()

print(
    "Subjects to process =",
    len(
        to_run
    )
)

for i, (_, row) in enumerate(
    to_run.iterrows(),
    start=1
):

    sub = row[
        "subject"
    ]

    run = row[
        "run"
    ]

    print(
        f"\n[{i:02d}/{len(to_run):02d}] "
        f"{sub} {run}",
        flush=True
    )

    # ---------------------------------------------------------------------
    # THIS CALL USES THE ORIGINAL 375 IMPLEMENTATION.
    # We do not reimplement its denoising logic here.
    # ---------------------------------------------------------------------

    res = old.process_one(
        row=row,

        out_root=str(
            OUT_ROOT
        ),

        fd_cutoff=0.5,

        low_pass=0.08,

        high_pass=0.01,

        min_dof_ratio=0.50,
    )

    results = [
        x
        for x in results
        if str(
            x.get(
                "subject",
                ""
            )
        )
        !=
        sub
    ]

    results.append(
        res
    )

    qc = pd.DataFrame(
        results
    )

    if (
        len(qc)
        and
        "subject"
        in qc.columns
    ):

        qc["_order"] = (
            qc[
                "subject"
            ]
            .map(
                subject_key
            )
        )

        qc = (
            qc
            .sort_values(
                "_order"
            )
            .drop(
                columns="_order"
            )
        )

    qc.to_csv(
        QC_FILE,
        sep="\t",
        index=False
    )

    print(
        "status =",
        res.get(
            "status",
            "NA"
        ),
        "| reason =",
        res.get(
            "reason",
            "NA"
        ),
        flush=True
    )


# =============================================================================
# STEP 7. FINAL OUTPUT VALIDATION
# =============================================================================

print("\n" + "=" * 110)
print("STEP 3. FINAL VALIDATION")
print("=" * 110)

qc = pd.read_csv(
    QC_FILE,
    sep="\t",
    keep_default_na=False
)

qc["_order"] = (
    qc[
        "subject"
    ]
    .map(
        subject_key
    )
)

qc = (
    qc
    .sort_values(
        "_order"
    )
    .drop(
        columns="_order"
    )
    .drop_duplicates(
        "subject",
        keep="last"
    )
)

qc.to_csv(
    QC_FILE,
    sep="\t",
    index=False
)

manifest_rows = []

for _, row in src.iterrows():

    sub = row[
        "subject"
    ]

    q = qc[
        qc[
            "subject"
        ]
        ==
        sub
    ]

    if len(q) != 1:

        manifest_rows.append(
            {
                "subject":
                    sub,

                "status":
                    "MISSING_QC",

                "output_file":
                    "",
            }
        )

        continue

    q = q.iloc[
        0
    ]

    output_file = str(
        q.get(
            "output_file",
            ""
        )
    )

    exists = (
        bool(
            output_file
        )
        and
        Path(
            output_file
        ).exists()
        and
        Path(
            output_file
        ).stat().st_size
        >
        0
    )

    manifest_rows.append(
        {
            "subject":
                sub,

            "run":
                row[
                    "run"
                ],

            "status":
                q.get(
                    "status",
                    "NA"
                ),

            "reason":
                q.get(
                    "reason",
                    "NA"
                ),

            "output_exists":
                int(
                    exists
                ),

            "output_file":
                output_file,

            "used_confounds_file":
                q.get(
                    "used_confounds_file",
                    ""
                ),

            "n_acompcor_regressors":
                q.get(
                    "n_acompcor_regressors",
                    ""
                ),

            "acompcor_regressors":
                q.get(
                    "acompcor_regressors",
                    ""
                ),

            "n_fd_spike_regressors":
                q.get(
                    "n_fd_spike_regressors",
                    ""
                ),

            "confound_rank":
                q.get(
                    "confound_rank",
                    ""
                ),

            "dof":
                q.get(
                    "dof",
                    ""
                ),

            "dof_ratio":
                q.get(
                    "dof_ratio",
                    ""
                ),
        }
    )

M = pd.DataFrame(
    manifest_rows
)

M.to_csv(
    OUTPUT_MANIFEST,
    sep="\t",
    index=False
)

n_output = int(
    M[
        "output_exists"
    ].sum()
)

# -------------------------------------------------------------------------
# aCompCor distribution
# -------------------------------------------------------------------------

acomp = pd.to_numeric(
    M[
        "n_acompcor_regressors"
    ],
    errors="coerce"
).dropna()

dof = pd.to_numeric(
    M[
        "dof_ratio"
    ],
    errors="coerce"
).dropna()

spikes = pd.to_numeric(
    M[
        "n_fd_spike_regressors"
    ],
    errors="coerce"
).dropna()

status_counts = (
    M[
        "status"
    ]
    .value_counts(
        dropna=False
    )
)

# -------------------------------------------------------------------------
# Verify output NIfTI time dimension
# -------------------------------------------------------------------------

bad_nifti = []

n_time = []

for _, row in M.iterrows():

    if not int(
        row[
            "output_exists"
        ]
    ):
        continue

    try:

        img = nib.load(
            row[
                "output_file"
            ]
        )

        if len(
            img.shape
        ) != 4:

            bad_nifti.append(
                {
                    "subject":
                        row[
                            "subject"
                        ],

                    "reason":
                        f"shape={img.shape}",
                }
            )

        else:

            n_time.append(
                int(
                    img.shape[
                        3
                    ]
                )
            )

    except Exception as e:

        bad_nifti.append(
            {
                "subject":
                    row[
                        "subject"
                    ],

                "reason":
                    str(e),
            }
        )

pd.DataFrame(
    bad_nifti
).to_csv(
    QC_DIR /
    "bad_output_nifti.tsv",
    sep="\t",
    index=False
)

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

summary = []

summary.append(
    "=" * 90
)

summary.append(
    "CURRENT64 EXACT 375-MATCHED DENOISING"
)

summary.append(
    "=" * 90
)

summary.append(
    ""
)

summary.append(
    f"Frozen 375 source:"
)

summary.append(
    str(
        OLD_SCRIPT
    )
)

summary.append(
    ""
)

summary.append(
    f"SHA256:"
)

summary.append(
    old_sha
)

summary.append(
    ""
)

summary.append(
    f"Input subjects = {len(src)}"
)

summary.append(
    f"Output BOLD present = {n_output}/64"
)

summary.append(
    f"Bad output NIfTI = {len(bad_nifti)}"
)

summary.append(
    ""
)

summary.append(
    "STATUS COUNTS"
)

for k, v in status_counts.items():

    summary.append(
        f"  {k}: {v}"
    )

summary.append(
    ""
)

summary.append(
    "ACTUAL aCompCor USAGE"
)

if len(acomp):

    summary.append(
        f"  min    = {int(acomp.min())}"
    )

    summary.append(
        f"  median = {float(acomp.median()):.2f}"
    )

    summary.append(
        f"  mean   = {float(acomp.mean()):.2f}"
    )

    summary.append(
        f"  max    = {int(acomp.max())}"
    )

    summary.append(
        f"  N < 5  = {int((acomp < 5).sum())}"
    )

    summary.append(
        f"  N = 5  = {int((acomp == 5).sum())}"
    )

summary.append(
    ""
)

summary.append(
    "FD SPIKES"
)

if len(spikes):

    summary.append(
        f"  min    = {int(spikes.min())}"
    )

    summary.append(
        f"  median = {float(spikes.median()):.2f}"
    )

    summary.append(
        f"  max    = {int(spikes.max())}"
    )

summary.append(
    ""
)

summary.append(
    "DOF RATIO"
)

if len(dof):

    summary.append(
        f"  min    = {float(dof.min()):.6f}"
    )

    summary.append(
        f"  median = {float(dof.median()):.6f}"
    )

    summary.append(
        f"  max    = {float(dof.max()):.6f}"
    )

summary.append(
    ""
)

summary.append(
    "OUTPUT TIMEPOINTS"
)

if n_time:

    summary.append(
        f"  min    = {min(n_time)}"
    )

    summary.append(
        f"  median = {float(np.median(n_time)):.1f}"
    )

    summary.append(
        f"  max    = {max(n_time)}"
    )

summary.append(
    ""
)

if (
    n_output == 64
    and
    len(
        bad_nifti
    )
    ==
    0
):

    summary.append(
        "FINAL_CHECK = PASS"
    )

else:

    summary.append(
        "FINAL_CHECK = REVIEW"
    )

summary.append(
    ""
)

summary.append(
    "IMPORTANT:"
)

summary.append(
    "These outputs were generated by the frozen original 375 "
    "process_one() implementation."
)

summary.append(
    "The existing rest_motionDOF_pass64_v2 outputs were not modified."
)

summary.append(
    ""
)

summary.append(
    f"QC:"
)

summary.append(
    str(
        QC_FILE
    )
)

summary.append(
    ""
)

summary.append(
    f"Output manifest:"
)

summary.append(
    str(
        OUTPUT_MANIFEST
    )
)

summary_text = "\n".join(
    summary
)

SUMMARY_FILE.write_text(
    summary_text
    +
    "\n"
)

print(
    "\n"
    +
    summary_text
)
