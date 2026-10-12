#!/usr/bin/env python3
"""Prepare the 2847-voxel striatal seed and ordered 65 ROI masks.

All source masks are resampled to a caller-supplied three-dimensional reference
grid (or volume 0 of a supplied four-dimensional image) with nearest-neighbour
interpolation and binarized at >0.  The whole
striatum is the union of VST, Caudate, and Putamen.  ROI column order follows
the sorted ``roi_id`` in the explicit source manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img


ORIGINAL_SOURCE_SHA256 = "f375d61321b0f3a02d29ad6c2a8d376dbbf1055441824eeb095c280768a14773"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def read_roi_manifest(path: Path, expected_rois: int) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    required = {"roi_id", "roi_name", "mask_file"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"ROI source manifest lacks columns: {sorted(missing)}")
    table = table.sort_values("roi_id", kind="stable").reset_index(drop=True).copy()
    if len(table) != expected_rois:
        raise ValueError(f"Expected {expected_rois} ROI masks, found {len(table)}")
    if table["roi_id"].duplicated().any() or table["roi_name"].astype(str).duplicated().any():
        raise ValueError("ROI IDs and names must each be unique")
    table["mask_file"] = [
        str(resolve_path(value, path.parent)) for value in table["mask_file"]
    ]
    absent = [value for value in table["mask_file"] if not Path(value).is_file()]
    if absent:
        raise FileNotFoundError(f"Missing ROI source masks; first entries: {absent[:5]}")
    return table


def resample_binary_mask(source: Path, reference: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    source_image = nib.load(source)
    resampled = resample_to_img(
        source_image,
        reference,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    data = resampled.get_fdata()
    if not np.isfinite(data).all():
        raise ValueError(f"Mask contains non-finite values after resampling: {source}")
    binary = (data > 0).astype(np.uint8)
    if not np.any(binary):
        raise ValueError(f"Mask is empty after resampling: {source}")
    return nib.Nifti1Image(binary, reference.affine, reference.header)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--vst-mask", required=True, type=Path)
    parser.add_argument("--caudate-mask", required=True, type=Path)
    parser.add_argument("--putamen-mask", required=True, type=Path)
    parser.add_argument("--roi-source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-voxels", type=int, default=2847)
    parser.add_argument("--expected-rois", type=int, default=65)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate explicit inputs and output safety without resampling or writing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "reference": args.reference_image.resolve(),
        "VST": args.vst_mask.resolve(),
        "Caudate": args.caudate_mask.resolve(),
        "Putamen": args.putamen_mask.resolve(),
    }
    absent = [str(path) for path in inputs.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing mask-preparation inputs: {absent}")
    roi_source_path = args.roi_source_manifest.resolve()
    rois = read_roi_manifest(roi_source_path, args.expected_rois)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {output_dir}")

    reference_source = nib.load(inputs["reference"])
    if len(reference_source.shape) == 4 and reference_source.shape[3] >= 1:
        reference = nib.Nifti1Image(
            np.asanyarray(reference_source.dataobj[..., 0]),
            reference_source.affine,
            reference_source.header,
        )
    elif len(reference_source.shape) == 3:
        reference = reference_source
    else:
        raise ValueError(f"Reference image must be 3D or 4D, got {reference_source.shape}")
    if args.check_only:
        print(
            f"PASS check-only: reference_shape={reference.shape} rois={len(rois)} "
            f"output={output_dir}"
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    roi_output_dir = output_dir / "roi_masks_65"
    roi_output_dir.mkdir()

    striatal_arrays: dict[str, np.ndarray] = {}
    striatal_files: dict[str, Path] = {}
    for name in ("VST", "Caudate", "Putamen"):
        prepared = resample_binary_mask(inputs[name], reference)
        output = output_dir / f"{name.lower()}_mask_current_ref.nii.gz"
        nib.save(prepared, output)
        striatal_arrays[name] = prepared.get_fdata() > 0
        striatal_files[name] = output

    vst = striatal_arrays["VST"]
    caudate = striatal_arrays["Caudate"]
    putamen = striatal_arrays["Putamen"]
    whole = vst | caudate | putamen
    dorsal = caudate | putamen
    if int(whole.sum()) != args.expected_voxels:
        raise ValueError(
            f"Expected {args.expected_voxels} union voxels, found {int(whole.sum())}"
        )
    labels = np.zeros(reference.shape, dtype=np.uint8)
    labels[vst] = 1
    labels[caudate] = 2
    labels[putamen] = 3

    whole_file = output_dir / "whole_striatum_seed_mask_current_ref.nii.gz"
    dorsal_file = output_dir / "dorsal_striatum_mask_current_ref.nii.gz"
    label_file = output_dir / "striatal_subdivision_label_current_ref.nii.gz"
    reference_file = output_dir / "current_denoised_bold_reference_3d.nii.gz"
    nib.save(reference, reference_file)
    nib.save(nib.Nifti1Image(whole.astype(np.uint8), reference.affine, reference.header), whole_file)
    nib.save(nib.Nifti1Image(dorsal.astype(np.uint8), reference.affine, reference.header), dorsal_file)
    nib.save(nib.Nifti1Image(labels, reference.affine, reference.header), label_file)

    roi_rows: list[dict[str, object]] = []
    for row in rois.itertuples(index=False):
        safe_name = re.sub(r"[^A-Za-z0-9_+-]+", "_", str(row.roi_name))
        output = roi_output_dir / f"roi-{int(row.roi_id):02d}_{safe_name}_current_ref.nii.gz"
        prepared = resample_binary_mask(Path(row.mask_file), reference)
        nib.save(prepared, output)
        roi_rows.append(
            {
                "roi_id": int(row.roi_id),
                "roi_name": str(row.roi_name),
                "source_mask": str(row.mask_file),
                "prepared_mask": str(output),
                "n_voxels": int((prepared.get_fdata() > 0).sum()),
                "prepared_sha256": sha256_file(output),
            }
        )
    manifest_file = output_dir / "roi65_manifest_current_ref.tsv"
    write_tsv(manifest_file, roi_rows)

    ijk = np.array(np.where(whole)).T
    xyz = nib.affines.apply_affine(reference.affine, ijk)
    coordinates = [
        {
            "voxel_index": index,
            "i": int(position[0]),
            "j": int(position[1]),
            "k": int(position[2]),
            "x": float(world[0]),
            "y": float(world[1]),
            "z": float(world[2]),
            "label_id": int(labels[tuple(position)]),
            "label_name": {1: "VST", 2: "Caudate", 3: "Putamen"}.get(
                int(labels[tuple(position)]), "Unknown"
            ),
        }
        for index, (position, world) in enumerate(zip(ijk, xyz, strict=True))
    ]
    write_tsv(output_dir / "seed_voxel_coordinates.tsv", coordinates)

    summary = {
        "classification": "RECOVERED_BACKUP_DERIVED_PARAMETERIZED",
        "original_source_sha256": ORIGINAL_SOURCE_SHA256,
        "reference_sha256": sha256_file(inputs["reference"]),
        "prepared_reference_sha256": sha256_file(reference_file),
        "source_mask_sha256": {
            name: sha256_file(inputs[name]) for name in ("VST", "Caudate", "Putamen")
        },
        "prepared_mask_sha256": {
            name: sha256_file(path) for name, path in striatal_files.items()
        },
        "whole_striatum_sha256": sha256_file(whole_file),
        "roi_source_manifest_sha256": sha256_file(roi_source_path),
        "roi_prepared_manifest_sha256": sha256_file(manifest_file),
        "reference_shape": list(reference.shape),
        "vst_voxels": int(vst.sum()),
        "caudate_voxels": int(caudate.sum()),
        "putamen_voxels": int(putamen.sum()),
        "dorsal_striatum_voxels": int(dorsal.sum()),
        "whole_striatum_voxels": int(whole.sum()),
        "roi_count": len(roi_rows),
        "roi_voxel_min": min(int(row["n_voxels"]) for row in roi_rows),
        "roi_voxel_max": max(int(row["n_voxels"]) for row in roi_rows),
    }
    (output_dir / "prepared_mask_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"PASS: whole_striatum_voxels={summary['whole_striatum_voxels']} "
        f"rois={summary['roi_count']} output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
