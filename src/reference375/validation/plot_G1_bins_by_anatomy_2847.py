#!/usr/bin/env python3

from pathlib import Path
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.affines import apply_affine
from nibabel.processing import resample_from_to

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(os.environ["STRIATAL_HISTORICAL_ROOT"])

OUT = (
    ROOT
    / "23_gradient_validation_figures"
    / "G1_bins_by_anatomy_2847"
)
OUT.mkdir(parents=True, exist_ok=True)

KNOWN_WHOLE_MASK = (
    ROOT
    / "01_inputs"
    / "03_oxford_atlas"
    / "prebuilt_masks_from_old"
    / "oxford_whole_striatum_mask_ref.nii.gz"
)

EXPECTED_N = 2847

# ------------------------------------------------------------------
# Anatomical colours: FIXED for all ten bins
# ------------------------------------------------------------------

COLOR_PUTAMEN = "#0072B2"
COLOR_CAUDATE = "#D55E00"
COLOR_VST     = "#009E73"

COLOR_BACKGROUND = "#D7D7D7"

LABEL_COLOR = {
    1: COLOR_PUTAMEN,
    2: COLOR_CAUDATE,
    3: COLOR_VST,
}

LABEL_NAME = {
    1: "Putamen",
    2: "Caudate",
    3: "VST",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def nii_files(root):
    if not root.exists():
        return []

    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        s = p.name.lower()

        if s.endswith(".nii") or s.endswith(".nii.gz"):
            out.append(p)

    return out


def load_3d(path):
    img = nib.load(str(path))

    if len(img.shape) != 3:
        raise ValueError(
            f"Expected a 3D NIfTI, got shape {img.shape}: {path}"
        )

    return img, np.asanyarray(img.dataobj)


def same_grid(img1, img2):
    return (
        img1.shape == img2.shape
        and np.allclose(img1.affine, img2.affine, atol=1e-4)
    )


# ============================================================
# STEP 1
# FIND THE VERIFIED 2847-VOXEL WHOLE-STRIATUM MASK
# ============================================================

def find_whole_mask():

    if KNOWN_WHOLE_MASK.exists():
        img, data = load_3d(KNOWN_WHOLE_MASK)

        n = int(np.sum(data > 0))

        if n == EXPECTED_N:
            return KNOWN_WHOLE_MASK

        print(
            f"WARNING: known whole mask exists but has {n} voxels, "
            f"not {EXPECTED_N}."
        )

    candidates = []

    for p in nii_files(ROOT):

        name = p.name.lower()

        if "striat" not in name:
            continue

        if "whole" not in name and "mask" not in name:
            continue

        try:
            img, data = load_3d(p)
        except Exception:
            continue

        n = int(np.sum(data > 0))

        if n == EXPECTED_N:

            score = 0

            if "whole" in name:
                score += 20

            if "oxford" in name:
                score += 20

            if "ref" in name:
                score += 10

            if "prebuilt" in str(p).lower():
                score += 10

            candidates.append((score, p))

    if not candidates:
        raise RuntimeError(
            "\nCould not find a 2847-voxel whole-striatum mask.\n"
            f"Expected first at:\n{KNOWN_WHOLE_MASK}"
        )

    candidates.sort(
        key=lambda x: (x[0], str(x[1])),
        reverse=True
    )

    return candidates[0][1]


WHOLE_MASK = find_whole_mask()

whole_img, whole_data = load_3d(WHOLE_MASK)
whole = whole_data > 0

ijk = np.argwhere(whole)
xyz = apply_affine(whole_img.affine, ijk)

assert len(ijk) == EXPECTED_N

print("=" * 90)
print("WHOLE STRIATUM")
print("=" * 90)
print(WHOLE_MASK)
print("Voxel count:", len(ijk))
print("Shape      :", whole_img.shape)
print()


# ============================================================
# STEP 2
# FIND / RECONSTRUCT THE EXACT 10 G1 BINS
# ============================================================

def normalize_discrete_bin_values(vals):

    vals = np.asarray(vals)

    if not np.all(np.isfinite(vals)):
        return None

    rounded = np.rint(vals)

    if not np.allclose(vals, rounded, atol=1e-5):
        return None

    rounded = rounded.astype(int)

    u = sorted(np.unique(rounded).tolist())

    if u == list(range(1, 11)):
        return rounded

    if u == list(range(0, 10)):
        return rounded + 1

    return None


def score_binmap_path(p):

    s = str(p).lower()

    score = 0

    if "g1" in s:
        score += 20

    if "10bin" in s or "10_bin" in s:
        score += 30

    if "binmap" in s or "bin_map" in s:
        score += 20

    if "gradient_validation" in s:
        score += 15

    if "23_gradient_validation_figures" in s:
        score += 15

    if "group" in s:
        score += 5

    if "sub-" in s:
        score -= 100

    return score


def find_discrete_10bin_map():

    candidates = []

    for p in nii_files(ROOT):

        n = p.name.lower()

        if (
            "g1" not in n
            and "gradient" not in n
            and "10bin" not in n
            and "10_bin" not in n
        ):
            continue

        if "bin" not in n:
            continue

        try:
            img, data = load_3d(p)
        except Exception:
            continue

        if img.shape != whole_img.shape:
            continue

        vals = data[whole]

        labels = normalize_discrete_bin_values(vals)

        if labels is None:
            continue

        counts = np.bincount(labels, minlength=11)[1:11]

        if (
            labels.size == EXPECTED_N
            and counts.sum() == EXPECTED_N
            and np.all(counts >= 280)
            and np.all(counts <= 290)
        ):
            candidates.append(
                (
                    score_binmap_path(p),
                    p,
                    labels
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (x[0], str(x[1])),
        reverse=True
    )

    return candidates[0]


def extract_bin_number(name):

    s = name.lower()

    patterns = [
        r"g1[_\-]?bin[_\-]?0?(10|[1-9])(?!\d)",
        r"bin[_\-]?0?(10|[1-9])(?!\d)",
    ]

    for pat in patterns:

        m = re.search(pat, s)

        if m:
            return int(m.group(1))

    return None


def find_individual_bin_masks():

    grouped = {b: [] for b in range(1, 11)}

    for p in nii_files(ROOT):

        b = extract_bin_number(p.name)

        if b is None:
            continue

        try:
            img, data = load_3d(p)
        except Exception:
            continue

        if img.shape != whole_img.shape:
            continue

        vals = data[whole]

        # binary image
        u = np.unique(vals)

        if not np.all(
            np.isin(
                np.rint(u).astype(int),
                [0, 1]
            )
        ):
            continue

        m = vals > 0

        n = int(m.sum())

        if n not in [284, 285]:
            continue

        score = score_binmap_path(p)

        grouped[b].append(
            (
                score,
                p,
                m
            )
        )

    if any(len(grouped[b]) == 0 for b in range(1, 11)):
        return None

    selected = {}

    for b in range(1, 11):

        grouped[b].sort(
            key=lambda x: (x[0], str(x[1])),
            reverse=True
        )

        selected[b] = grouped[b][0]

    labels = np.zeros(EXPECTED_N, dtype=int)

    sources = []

    for b in range(1, 11):

        _, p, m = selected[b]

        if np.any(labels[m] != 0):
            return None

        labels[m] = b
        sources.append(p)

    if np.any(labels == 0):
        return None

    return labels, sources


def score_continuous_g1(p):

    s = str(p).lower()

    score = 0

    if "g1" in p.name.lower():
        score += 30

    if "gradient1" in p.name.lower():
        score += 25

    if "principal" in p.name.lower():
        score += 10

    if "group" in s:
        score += 10

    if "reference" in s or "ref" in p.name.lower():
        score += 5

    if "consensus" in s:
        score += 10

    if "mean" in s:
        score += 5

    if "bin" in p.name.lower():
        score -= 40

    if "sub-" in s:
        score -= 100

    return score


def find_continuous_g1_and_rebin():

    candidates = []

    for p in nii_files(ROOT):

        name = p.name.lower()

        if "g1" not in name and "gradient1" not in name:
            continue

        if "bin" in name:
            continue

        try:
            img, data = load_3d(p)
        except Exception:
            continue

        if img.shape != whole_img.shape:
            continue

        vals = np.asarray(
            data[whole],
            dtype=float
        )

        if vals.size != EXPECTED_N:
            continue

        if not np.all(np.isfinite(vals)):
            continue

        if np.unique(vals).size < 100:
            continue

        if np.std(vals) <= 0:
            continue

        candidates.append(
            (
                score_continuous_g1(p),
                p,
                vals
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (x[0], str(x[1])),
        reverse=True
    )

    _, p, g1 = candidates[0]

    order = np.argsort(
        g1,
        kind="mergesort"
    )

    labels = np.zeros(
        EXPECTED_N,
        dtype=int
    )

    # np.array_split gives:
    # 7 bins x 285 + 3 bins x 284 = 2847
    pieces = np.array_split(
        order,
        10
    )

    for b, inds in enumerate(
        pieces,
        start=1
    ):
        labels[inds] = b

    return labels, p


resolved = find_discrete_10bin_map()

if resolved is not None:

    _, BIN_SOURCE, bin_labels = resolved

    print("=" * 90)
    print("G1 BINS")
    print("=" * 90)
    print("Resolved from existing discrete 10-bin map:")
    print(BIN_SOURCE)

else:

    indiv = find_individual_bin_masks()

    if indiv is not None:

        bin_labels, BIN_SOURCE = indiv

        print("=" * 90)
        print("G1 BINS")
        print("=" * 90)
        print("Reconstructed from ten existing binary bin masks:")
        for p in BIN_SOURCE:
            print(" ", p)

    else:

        continuous = find_continuous_g1_and_rebin()

        if continuous is None:
            raise RuntimeError(
                "\nCould not locate:\n"
                "  1) an existing 10-bin G1 map,\n"
                "  2) ten G1 binary bin masks, or\n"
                "  3) a continuous G1 NIfTI on the 2847-voxel grid.\n"
            )

        bin_labels, BIN_SOURCE = continuous

        print("=" * 90)
        print("G1 BINS")
        print("=" * 90)
        print("No discrete map found.")
        print("Reconstructed exact equal-count bins from continuous G1:")
        print(BIN_SOURCE)


counts = np.bincount(
    bin_labels,
    minlength=11
)[1:11]

print("Bin counts :", counts.tolist())
print("Total      :", int(counts.sum()))

assert int(counts.sum()) == EXPECTED_N
assert set(np.unique(bin_labels)) == set(range(1, 11))


# save resolved bin map for audit
bin_map = np.zeros(
    whole_img.shape,
    dtype=np.int16
)

bin_map[
    ijk[:, 0],
    ijk[:, 1],
    ijk[:, 2]
] = bin_labels

BIN_OUT = OUT / "G1_10bins_resolved_2847.nii.gz"

nib.save(
    nib.Nifti1Image(
        bin_map,
        whole_img.affine,
        whole_img.header
    ),
    str(BIN_OUT)
)

print("Saved resolved bin map:")
print(BIN_OUT)
print()


# ============================================================
# STEP 3
# AUTOMATICALLY FIND THE OXFORD-GSK-IMANOVA
# STRUCTURAL STRIATAL ATLAS
#
# This is the correct atlas for:
#   Caudate / Putamen / Ventral Striatum
# ============================================================

def candidate_fsl_roots():

    roots = []

    if os.environ.get("FSLDIR"):
        roots.append(
            Path(os.environ["FSLDIR"])
        )

    roots += [
        Path("/usr/local/fsl"),
        Path("/opt/fsl"),
        Path("/usr/share/fsl"),
        Path("/usr/share/fsl/6.0"),
        Path("/usr/local/share/fsl"),
    ]

    result = []

    for r in roots:

        if r.exists() and r not in result:
            result.append(r)

    return result


def find_structural_striatum_atlas():

    candidates = []

    # First: search project itself
    search_roots = [ROOT]

    # Then installed FSL
    search_roots += candidate_fsl_roots()

    seen = set()

    for root in search_roots:

        if not root.exists():
            continue

        for p in root.rglob(
            "striatum-structural*.nii*"
        ):

            if p in seen:
                continue

            seen.add(p)

            try:
                img = nib.load(str(p))
            except Exception:
                continue

            if len(img.shape) != 3:
                continue

            score = 0

            s = str(p).lower()

            if "striatum-structural" in p.name.lower():
                score += 100

            if "2mm" in p.name.lower():
                score += 20

            if img.shape == whole_img.shape:
                score += 50

            if str(ROOT).lower() in s:
                score += 25

            candidates.append(
                (
                    score,
                    p
                )
            )

    if not candidates:
        raise RuntimeError(
            "\nOxford-GSK-Imanova structural striatal atlas "
            "could not be found.\n"
            "Searched the project and common $FSLDIR locations.\n"
            "Check that FSL is installed / loaded."
        )

    candidates.sort(
        key=lambda x: (x[0], str(x[1])),
        reverse=True
    )

    return candidates[0][1]


ATLAS_FILE = find_structural_striatum_atlas()

atlas_img = nib.load(
    str(ATLAS_FILE)
)

print("=" * 90)
print("ANATOMICAL ATLAS")
print("=" * 90)
print(ATLAS_FILE)
print("Atlas shape:", atlas_img.shape)
print()


# ============================================================
# STEP 4
# READ LABEL NAMES FROM FSL XML IF AVAILABLE
# ============================================================

def find_matching_xml():

    xmls = []

    search_roots = candidate_fsl_roots()

    # Search the parent tree of the chosen atlas too
    search_roots.append(
        ATLAS_FILE.parent.parent
    )

    seen = set()

    for root in search_roots:

        if not root.exists():
            continue

        for p in root.rglob("*.xml"):

            if p in seen:
                continue

            seen.add(p)

            try:
                text = p.read_text(
                    errors="ignore"
                ).lower()
            except Exception:
                continue

            if (
                "caudate" in text
                and "putamen" in text
                and (
                    "ventral striatum" in text
                    or "ventral_striatum" in text
                )
            ):
                xmls.append(p)

    if not xmls:
        return None

    xmls.sort(
        key=lambda p: (
            "striat" in p.name.lower(),
            "struct" in p.name.lower(),
            -len(str(p))
        ),
        reverse=True
    )

    return xmls[0]


def read_label_indices_from_xml(xml_file):

    if xml_file is None:
        return None

    try:
        tree = ET.parse(str(xml_file))
    except Exception:
        return None

    raw = {
        "caudate": [],
        "putamen": [],
        "vst": [],
    }

    for elem in tree.iter():

        if not elem.tag.lower().endswith("label"):
            continue

        if elem.text is None:
            continue

        txt = elem.text.strip().lower()

        if "index" not in elem.attrib:
            continue

        try:
            idx = int(elem.attrib["index"])
        except Exception:
            continue

        if "caudate" in txt:
            raw["caudate"].append(idx)

        elif "putamen" in txt:
            raw["putamen"].append(idx)

        elif (
            "ventral striatum" in txt
            or "ventral_striatum" in txt
        ):
            raw["vst"].append(idx)

    if any(
        len(raw[k]) == 0
        for k in raw
    ):
        return None

    return raw


XML_FILE = find_matching_xml()
xml_indices = read_label_indices_from_xml(
    XML_FILE
)

atlas_raw = np.asanyarray(
    atlas_img.dataobj
)

atlas_unique = sorted(
    np.unique(
        np.rint(atlas_raw).astype(int)
    ).tolist()
)

print("Atlas unique values:")
print(atlas_unique)

if XML_FILE is not None:
    print("Atlas XML:")
    print(XML_FILE)

print()


def resolve_atlas_values():

    # Preferred: use XML label names
    if xml_indices is not None:

        present = set(atlas_unique)

        # FSL XML often uses index 0..N-1 while
        # hard-label NIfTI uses 1..N.
        for offset in [1, 0]:

            mapped = {
                k: [
                    x + offset
                    for x in vals
                ]
                for k, vals
                in xml_indices.items()
            }

            flat = [
                x
                for vals in mapped.values()
                for x in vals
            ]

            if (
                all(x > 0 for x in flat)
                and all(x in present for x in flat)
            ):
                return mapped

    # Oxford structural striatum hard atlas fallback:
    # 1 = Caudate
    # 2 = Putamen
    # 3 = Ventral striatum
    present = set(atlas_unique)

    if {1, 2, 3}.issubset(present):

        return {
            "caudate": [1],
            "putamen": [2],
            "vst": [3],
        }

    raise RuntimeError(
        "Could not resolve caudate / putamen / VST "
        "label values from the structural atlas."
    )


atlas_values = resolve_atlas_values()

# >>> FIX_CA_VST_MAPPING_20260904
# Previous automatic interpretation inverted Caudate and VST.
# Anatomical inspection of the Oxford structural striatal atlas
# shows that the large curved dorsomedial component is Caudate,
# while the smaller anteroventral component is VST.
atlas_values["caudate"], atlas_values["vst"] = (
    atlas_values["vst"],
    atlas_values["caudate"],
)

print("Resolved structural atlas values:")
print(" Caudate :", atlas_values["caudate"])
print(" Putamen :", atlas_values["putamen"])
print(" VST     :", atlas_values["vst"])
print()


# ============================================================
# STEP 5
# RESAMPLE ANATOMICAL ATLAS TO THE 2847-VOXEL REFERENCE
# ============================================================

if same_grid(
    atlas_img,
    whole_img
):
    atlas_on_ref = np.rint(
        atlas_raw
    ).astype(int)

else:

    atlas_resampled_img = resample_from_to(
        atlas_img,
        (
            whole_img.shape,
            whole_img.affine
        ),
        order=0
    )

    atlas_on_ref = np.rint(
        np.asanyarray(
            atlas_resampled_img.dataobj
        )
    ).astype(int)


# Anatomy code:
# 0 = unclassified
# 1 = putamen
# 2 = caudate
# 3 = VST

anat = np.zeros(
    whole_img.shape,
    dtype=np.int16
)

anat[
    np.isin(
        atlas_on_ref,
        atlas_values["putamen"]
    )
] = 1

anat[
    np.isin(
        atlas_on_ref,
        atlas_values["caudate"]
    )
] = 2

anat[
    np.isin(
        atlas_on_ref,
        atlas_values["vst"]
    )
] = 3


# ============================================================
# STEP 6
# HANDLE EDGE VOXELS CREATED BY GRID RESAMPLING
#
# If the 2847 mask and FSL atlas differ by a tiny resampling
# boundary, assign those edge voxels to the nearest one of
# Caudate / Putamen / VST.
# ============================================================

missing = (
    whole
    & (anat == 0)
)

n_missing = int(
    missing.sum()
)

print(
    "Initially unclassified voxels "
    f"inside 2847 mask: {n_missing}"
)

if n_missing > 0:

    if np.sum(anat > 0) == 0:
        raise RuntimeError(
            "Structural atlas has no overlap with the "
            "2847-voxel mask after resampling."
        )

    voxel_sizes = nib.affines.voxel_sizes(
        whole_img.affine
    )

    dist, nearest_indices = distance_transform_edt(
        anat == 0,
        sampling=voxel_sizes,
        return_indices=True
    )

    nearest_labels = anat[
        tuple(
            nearest_indices[d][missing]
            for d in range(3)
        )
    ]

    missing_dist = dist[missing]

    anat[missing] = nearest_labels

    print(
        "Boundary-fill distance, median:",
        f"{np.median(missing_dist):.2f} mm"
    )

    print(
        "Boundary-fill distance, max   :",
        f"{np.max(missing_dist):.2f} mm"
    )

    if np.max(missing_dist) > 8:
        print(
            "WARNING: maximum boundary-fill distance "
            "is > 8 mm. Inspect the anatomical reference "
            "image carefully."
        )


remaining = int(
    np.sum(
        whole
        & (anat == 0)
    )
)

if remaining != 0:
    raise RuntimeError(
        f"{remaining} voxels remain unclassified."
    )


anat_vec = anat[
    ijk[:, 0],
    ijk[:, 1],
    ijk[:, 2]
]

print()
print("Full 2847-voxel anatomical composition:")
print(
    " Putamen:",
    int(np.sum(anat_vec == 1))
)
print(
    " Caudate:",
    int(np.sum(anat_vec == 2))
)
print(
    " VST    :",
    int(np.sum(anat_vec == 3))
)
print(
    " Total  :",
    int(len(anat_vec))
)
print()


ANAT_OUT = (
    OUT
    / "Oxford_structural_anatomy_on_2847_reference.nii.gz"
)

nib.save(
    nib.Nifti1Image(
        anat,
        whole_img.affine,
        whole_img.header
    ),
    str(ANAT_OUT)
)


# ============================================================
# STEP 7
# COMPUTE THE EXACT TABLE TO BE WRITTEN INTO THE FIGURES
# ============================================================

rows = []

for b in range(1, 11):

    m = bin_labels == b

    n_total = int(
        np.sum(m)
    )

    n_put = int(
        np.sum(
            m & (anat_vec == 1)
        )
    )

    n_cau = int(
        np.sum(
            m & (anat_vec == 2)
        )
    )

    n_vst = int(
        np.sum(
            m & (anat_vec == 3)
        )
    )

    assert (
        n_put
        + n_cau
        + n_vst
        == n_total
    )

    rows.append(
        {
            "G1_bin": b,
            "N_voxels": n_total,

            "Putamen_n": n_put,
            "Putamen_pct": 100.0 * n_put / n_total,

            "Caudate_n": n_cau,
            "Caudate_pct": 100.0 * n_cau / n_total,

            "VST_n": n_vst,
            "VST_pct": 100.0 * n_vst / n_total,
        }
    )


table = pd.DataFrame(rows)

TABLE_OUT = (
    OUT
    / "G1_bin_anatomical_composition.tsv"
)

table.to_csv(
    TABLE_OUT,
    sep="\t",
    index=False,
    float_format="%.3f"
)

print("=" * 90)
print("BIN-WISE ANATOMICAL COMPOSITION")
print("=" * 90)

print(
    table.to_string(
        index=False,
        formatters={
            "Putamen_pct": lambda x: f"{x:.1f}",
            "Caudate_pct": lambda x: f"{x:.1f}",
            "VST_pct": lambda x: f"{x:.1f}",
        }
    )
)

print()


# ============================================================
# PLOTTING HELPERS
# ============================================================

def set_equal_axes(ax):

    mins = xyz.min(
        axis=0
    )

    maxs = xyz.max(
        axis=0
    )

    centers = (
        mins + maxs
    ) / 2.0

    radius = (
        maxs - mins
    ).max() / 2.0

    ax.set_xlim(
        centers[0] - radius,
        centers[0] + radius
    )

    ax.set_ylim(
        centers[1] - radius,
        centers[1] + radius
    )

    ax.set_zlim(
        centers[2] - radius,
        centers[2] + radius
    )


def clean_axis(ax):

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.grid(False)

    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor(
            (1, 1, 1, 0)
        )

        ax.yaxis.pane.set_edgecolor(
            (1, 1, 1, 0)
        )

        ax.zaxis.pane.set_edgecolor(
            (1, 1, 1, 0)
        )
    except Exception:
        pass

    try:
        ax.xaxis.line.set_color(
            (1, 1, 1, 0)
        )

        ax.yaxis.line.set_color(
            (1, 1, 1, 0)
        )

        ax.zaxis.line.set_color(
            (1, 1, 1, 0)
        )
    except Exception:
        pass


def orientation_cross(
    ax,
    right_view=False
):
    """
    Robust 2D orientation glyph embedded inside each 3D panel.

    This avoids the matplotlib 3D problem where text may appear
    but orientation lines drawn with ax.transAxes may disappear.
    """

    # --------------------------------------------------------
    # Small independent 2D axes in lower-left corner
    # --------------------------------------------------------

    iax = ax.inset_axes(
        [0.035, 0.035, 0.20, 0.20],
        transform=ax.transAxes,
        zorder=20
    )

    iax.set_xlim(-1.25, 1.25)
    iax.set_ylim(-1.25, 1.25)

    iax.set_xticks([])
    iax.set_yticks([])

    for spine in iax.spines.values():
        spine.set_visible(False)

    iax.patch.set_facecolor("white")
    iax.patch.set_alpha(0.82)

    c = "#2878B5"

    # --------------------------------------------------------
    # Main superior-inferior axis
    # --------------------------------------------------------

    iax.plot(
        [0, 0],
        [-0.82, 0.82],
        color=c,
        lw=1.8,
        solid_capstyle="round"
    )

    # small arrow toward Superior
    iax.annotate(
        "",
        xy=(0, 0.90),
        xytext=(0, 0.58),
        arrowprops=dict(
            arrowstyle="-|>",
            color=c,
            lw=1.6,
            mutation_scale=10
        )
    )

    iax.text(
        0,
        1.05,
        "S",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    iax.text(
        0,
        -1.05,
        "I",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Left-right axis
    # --------------------------------------------------------

    iax.plot(
        [-0.82, 0.82],
        [0, 0],
        color=c,
        lw=1.8,
        solid_capstyle="round"
    )

    if right_view:
        left_label = "R"
        right_label = "L"
    else:
        left_label = "L"
        right_label = "R"

    iax.text(
        -1.02,
        0,
        left_label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    iax.text(
        1.02,
        0,
        right_label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Anterior-posterior oblique axis
    # --------------------------------------------------------

    iax.plot(
        [-0.66, 0.66],
        [-0.50, 0.50],
        color=c,
        lw=1.8,
        solid_capstyle="round"
    )

    if right_view:

        # right-oblique view
        iax.text(
            -0.82,
            -0.64,
            "A",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

        iax.text(
            0.82,
            0.64,
            "P",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    else:

        # left-oblique view
        iax.text(
            -0.82,
            -0.64,
            "P",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

        iax.text(
            0.82,
            0.64,
            "A",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    iax.set_aspect(
        "equal",
        adjustable="box"
    )

def draw_view(
    ax,
    current_bin=None,
    anatomy_reference=False,
    right_view=False
):

    if anatomy_reference:

        # Complete anatomy image:
        # every voxel is anatomically coloured
        for code in [1, 2, 3]:

            m = anat_vec == code

            ax.scatter(
                xyz[m, 0],
                xyz[m, 1],
                xyz[m, 2],
                s=14,
                c=LABEL_COLOR[code],
                alpha=0.92,
                edgecolors="none",
                depthshade=False,
                rasterized=True
            )

    else:

        current = (
            bin_labels == current_bin
        )

        background = ~current

        # All other G1 bins in grey
        ax.scatter(
            xyz[background, 0],
            xyz[background, 1],
            xyz[background, 2],
            s=8,
            c=COLOR_BACKGROUND,
            alpha=0.48,
            edgecolors="none",
            depthshade=False,
            rasterized=True
        )

        # Current bin coloured according to anatomy
        for code in [1, 2, 3]:

            m = (
                current
                & (anat_vec == code)
            )

            if np.sum(m) == 0:
                continue

            ax.scatter(
                xyz[m, 0],
                xyz[m, 1],
                xyz[m, 2],
                s=18,
                c=LABEL_COLOR[code],
                alpha=1.0,
                edgecolors="none",
                depthshade=False,
                rasterized=True
            )

    # Same two oblique views as before
    if right_view:

        ax.view_init(
            elev=15,
            azim=115
        )

    else:

        ax.view_init(
            elev=15,
            azim=-115
        )

    set_equal_axes(ax)
    clean_axis(ax)

    orientation_cross(
        ax,
        right_view=right_view
    )


legend_handles = [

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor=COLOR_PUTAMEN,
        markeredgecolor="none",
        markersize=9,
        label="Putamen"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor=COLOR_CAUDATE,
        markeredgecolor="none",
        markersize=9,
        label="Caudate"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markerfacecolor=COLOR_VST,
        markeredgecolor="none",
        markersize=9,
        label="VST"
    ),
]


# ============================================================
# STEP 8
# FIRST DRAW ONE COMPLETE ANATOMICAL REFERENCE
#
# This is useful for identifying:
#   Which long strip = caudate?
#   Which mass = putamen?
#   Where exactly = VST?
# ============================================================

fig = plt.figure(
    figsize=(15, 7.5),
    facecolor="white"
)

ax1 = fig.add_subplot(
    1,
    2,
    1,
    projection="3d"
)

ax2 = fig.add_subplot(
    1,
    2,
    2,
    projection="3d"
)

draw_view(
    ax1,
    anatomy_reference=True,
    right_view=False
)

draw_view(
    ax2,
    anatomy_reference=True,
    right_view=True
)

fig.suptitle(
    "Striatal anatomical reference",
    fontsize=21,
    y=0.965
)

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.035),
    ncol=3,
    frameon=False,
    fontsize=12
)

fig.text(
    0.018,
    0.88,
    (
        f"Whole striatum: N = {EXPECTED_N}\n"
        f"Putamen: {int(np.sum(anat_vec == 1))}\n"
        f"Caudate: {int(np.sum(anat_vec == 2))}\n"
        f"VST: {int(np.sum(anat_vec == 3))}"
    ),
    ha="left",
    va="top",
    fontsize=11,
    linespacing=1.45,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="#BDBDBD",
        alpha=0.95
    )
)

plt.tight_layout(
    rect=[0, 0.08, 1, 0.94]
)

ANAT_FIG = (
    OUT
    / "00_striatal_anatomical_reference_two_views.png"
)

fig.savefig(
    ANAT_FIG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("Saved:")
print(ANAT_FIG)


# ============================================================
# STEP 9
# DRAW THE TEN G1 BIN FIGURES
# ============================================================

png_files = []

for b in range(1, 11):

    row = table.loc[
        table["G1_bin"] == b
    ].iloc[0]

    fig = plt.figure(
        figsize=(15, 7.5),
        facecolor="white"
    )

    ax1 = fig.add_subplot(
        1,
        2,
        1,
        projection="3d"
    )

    ax2 = fig.add_subplot(
        1,
        2,
        2,
        projection="3d"
    )

    draw_view(
        ax1,
        current_bin=b,
        anatomy_reference=False,
        right_view=False
    )

    draw_view(
        ax2,
        current_bin=b,
        anatomy_reference=False,
        right_view=True
    )

    # --------------------------------------------------------
    # Only the bin number in the title
    # --------------------------------------------------------

    fig.suptitle(
        f"G1 Bin {b}",
        fontsize=22,
        y=0.965
    )

    # --------------------------------------------------------
    # The exact TABLE ROW written into the upper-left corner
    # --------------------------------------------------------

    annotation = (
        f"N = {int(row['N_voxels'])}\n"
        f"Putamen: {int(row['Putamen_n'])} "
        f"({row['Putamen_pct']:.1f}%)\n"
        f"Caudate: {int(row['Caudate_n'])} "
        f"({row['Caudate_pct']:.1f}%)\n"
        f"VST: {int(row['VST_n'])} "
        f"({row['VST_pct']:.1f}%)"
    )

    fig.text(
        0.018,
        0.88,
        annotation,
        ha="left",
        va="top",
        fontsize=11.5,
        linespacing=1.45,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#BDBDBD",
            alpha=0.95
        )
    )

    # --------------------------------------------------------
    # Same legend in every image
    # --------------------------------------------------------

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        fontsize=12
    )

    plt.tight_layout(
        rect=[0, 0.08, 1, 0.94]
    )

    out_png = (
        OUT
        / f"G1_bin{b:02d}_anatomy_two_views.png"
    )

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    png_files.append(
        out_png
    )

    print("Saved:", out_png)


# ============================================================
# STEP 10
# CREATE 10-BIN CONTACT SHEET AUTOMATICALLY
# ============================================================

try:

    from PIL import Image

    imgs = [
        Image.open(p).convert("RGB")
        for p in png_files
    ]

    # reduce each panel for the contact sheet
    target_w = 1500

    thumbs = []

    for im in imgs:

        scale = (
            target_w
            / im.width
        )

        target_h = int(
            im.height
            * scale
        )

        thumbs.append(
            im.resize(
                (
                    target_w,
                    target_h
                )
            )
        )

    gap = 30

    cols = 2
    rows_n = 5

    cell_w = max(
        im.width
        for im in thumbs
    )

    cell_h = max(
        im.height
        for im in thumbs
    )

    sheet_w = (
        cols * cell_w
        + (cols + 1) * gap
    )

    sheet_h = (
        rows_n * cell_h
        + (rows_n + 1) * gap
    )

    sheet = Image.new(
        "RGB",
        (
            sheet_w,
            sheet_h
        ),
        "white"
    )

    for i, im in enumerate(
        thumbs
    ):

        row_i = i // cols
        col_i = i % cols

        x = (
            gap
            + col_i
            * (
                cell_w
                + gap
            )
        )

        y = (
            gap
            + row_i
            * (
                cell_h
                + gap
            )
        )

        sheet.paste(
            im,
            (x, y)
        )

    CONTACT = (
        OUT
        / "G1_all_10bins_anatomy_contact_sheet.png"
    )

    sheet.save(
        CONTACT,
        quality=95
    )

    print("Saved:")
    print(CONTACT)

except Exception as e:

    print(
        "Contact sheet skipped:",
        repr(e)
    )


# ============================================================
# FINAL AUDIT
# ============================================================

print()
print("=" * 90)
print("FINAL OUTPUT")
print("=" * 90)
print("Output directory:")
print(OUT)
print()
print("Anatomical table:")
print(TABLE_OUT)
print()
print("Anatomical reference:")
print(ANAT_FIG)
print()
print("Ten G1 figures:")
for p in png_files:
    print(p)
print()
print("DONE.")
