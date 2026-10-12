#!/usr/bin/env python3
"""Build the 375-subject voxel-by-ROI reference connectome.

The calculation order is:

1. extract T x 2847 striatal-voxel and T x 65 ROI-mean time series;
2. z-score every time series over time with sample SD (ddof=1);
3. replace non-finite standardized values with zero, as in the recovered code;
4. compute Pearson r as ``seed_z.T @ roi_z / (T - 1)``;
5. clip r to [-0.999999, 0.999999] and apply ``arctanh``;
6. save each 2847 x 65 Fisher-z fingerprint as float32;
7. take the elementwise arithmetic mean of the 375 Fisher-z fingerprints.

No inverse Fisher transformation is performed before gradient embedding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker


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


def zscore_time_axis(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Match the recovered time-axis standardization and report invalid inputs."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"Expected a T x features array with T >= 2, got {values.shape}")

    mean = np.nanmean(values, axis=0, keepdims=True)
    sd = np.nanstd(values, axis=0, ddof=1, keepdims=True)
    invalid_sd = (~np.isfinite(sd)) | (sd == 0)
    sd[invalid_sd] = np.nan
    standardized = (values - mean) / sd
    nonfinite_standardized = int((~np.isfinite(standardized)).sum())
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)

    return standardized, {
        "input_nonfinite_values": int((~np.isfinite(values)).sum()),
        "invalid_or_zero_variance_features": int(invalid_sd.sum()),
        "standardized_values_replaced_with_zero": nonfinite_standardized,
    }


def compute_fingerprint(
    seed_timeseries: np.ndarray,
    roi_timeseries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Compute Pearson-r and Fisher-z fingerprints using the recovered order."""

    seed_timeseries = np.asarray(seed_timeseries)
    roi_timeseries = np.asarray(roi_timeseries)
    if seed_timeseries.ndim != 2 or roi_timeseries.ndim != 2:
        raise ValueError("Seed and ROI time series must both be two-dimensional")
    if seed_timeseries.shape[0] != roi_timeseries.shape[0]:
        raise ValueError(
            f"Time-length mismatch: seed={seed_timeseries.shape}, ROI={roi_timeseries.shape}"
        )

    seed_z, seed_qc = zscore_time_axis(seed_timeseries)
    roi_z, roi_qc = zscore_time_axis(roi_timeseries)
    denominator = max(seed_z.shape[0] - 1, 1)
    correlation = (seed_z.T @ roi_z) / denominator
    correlation = np.clip(correlation, -0.999999, 0.999999)
    fisher_z = np.arctanh(correlation).astype(np.float32)

    qc = {
        "seed_input_nonfinite_values": seed_qc["input_nonfinite_values"],
        "seed_invalid_or_zero_variance_features": seed_qc[
            "invalid_or_zero_variance_features"
        ],
        "seed_standardized_values_replaced_with_zero": seed_qc[
            "standardized_values_replaced_with_zero"
        ],
        "roi_input_nonfinite_values": roi_qc["input_nonfinite_values"],
        "roi_invalid_or_zero_variance_features": roi_qc[
            "invalid_or_zero_variance_features"
        ],
        "roi_standardized_values_replaced_with_zero": roi_qc[
            "standardized_values_replaced_with_zero"
        ],
    }
    return correlation, fisher_z, qc


def read_subject_manifest(path: Path, expected_subjects: int) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    required = {"subject", "output_file"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Subject manifest lacks columns: {sorted(missing)}")
    if len(table) != expected_subjects:
        raise ValueError(f"Expected {expected_subjects} rows, found {len(table)}")
    if table["subject"].astype(str).duplicated().any():
        raise ValueError("Subject manifest contains duplicate subject identifiers")

    table = table.copy()
    table["output_file"] = [
        str(resolve_path(value, path.parent)) for value in table["output_file"]
    ]
    absent = [value for value in table["output_file"] if not Path(value).is_file()]
    if absent:
        raise FileNotFoundError(f"Missing denoised BOLD inputs; first entries: {absent[:5]}")
    if "run" not in table:
        table["run"] = "run-NA"
    return table


def read_roi_manifest(path: Path, expected_rois: int) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    mask_column = "prepared_mask" if "prepared_mask" in table else "mask_file"
    required = {"roi_id", "roi_name", mask_column}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"ROI manifest lacks columns: {sorted(missing)}")

    table = table.sort_values("roi_id", kind="stable").reset_index(drop=True).copy()
    if len(table) != expected_rois:
        raise ValueError(f"Expected {expected_rois} ROIs, found {len(table)}")
    if table["roi_id"].duplicated().any() or table["roi_name"].astype(str).duplicated().any():
        raise ValueError("ROI IDs and names must each be unique")

    table["mask_file"] = [
        str(resolve_path(value, path.parent)) for value in table[mask_column]
    ]
    absent = [value for value in table["mask_file"] if not Path(value).is_file()]
    if absent:
        raise FileNotFoundError(f"Missing ROI masks; first entries: {absent[:5]}")
    return table


def extract_roi_timeseries(bold_file: Path, roi_manifest: pd.DataFrame) -> np.ndarray:
    columns: list[np.ndarray] = []
    for row in roi_manifest.itertuples(index=False):
        masker = NiftiMasker(
            mask_img=row.mask_file,
            standardize=False,
            detrend=False,
            smoothing_fwhm=None,
            verbose=0,
        )
        voxel_timeseries = masker.fit_transform(str(bold_file))
        if voxel_timeseries.shape[1] == 0:
            raise ValueError(f"ROI mask has no voxels after masking: {row.mask_file}")
        columns.append(voxel_timeseries.mean(axis=1))
    return np.vstack(columns).T


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty QC table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-manifest", required=True, type=Path)
    parser.add_argument("--seed-mask", required=True, type=Path)
    parser.add_argument("--roi-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-subjects", type=int, default=375)
    parser.add_argument("--expected-voxels", type=int, default=2847)
    parser.add_argument("--expected-rois", type=int, default=65)
    parser.add_argument(
        "--strict-invalid-timeseries",
        action="store_true",
        help="Fail if non-finite or zero-variance time series are encountered. The recovered code did not fail.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate manifests, masks, counts, and output safety without loading BOLD data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subject_manifest_path = args.subject_manifest.resolve()
    seed_mask_path = args.seed_mask.resolve()
    roi_manifest_path = args.roi_manifest.resolve()
    output_dir = args.output_dir.resolve()

    subjects = read_subject_manifest(subject_manifest_path, args.expected_subjects)
    rois = read_roi_manifest(roi_manifest_path, args.expected_rois)
    if not seed_mask_path.is_file():
        raise FileNotFoundError(seed_mask_path)

    seed_image = nib.load(seed_mask_path)
    seed_data = seed_image.get_fdata()
    if not np.isfinite(seed_data).all():
        raise ValueError("Seed mask contains non-finite values")
    seed_voxels = int((seed_data > 0).sum())
    if seed_voxels != args.expected_voxels:
        raise ValueError(f"Expected {args.expected_voxels} seed voxels, found {seed_voxels}")

    for row in rois.itertuples(index=False):
        roi_image = nib.load(row.mask_file)
        if roi_image.shape != seed_image.shape or not np.allclose(
            roi_image.affine, seed_image.affine, rtol=0, atol=1e-5
        ):
            raise ValueError(f"ROI geometry does not match seed mask: {row.mask_file}")
        roi_data = roi_image.get_fdata()
        if not np.isfinite(roi_data).all() or not np.any(roi_data > 0):
            raise ValueError(f"ROI mask is empty or non-finite: {row.mask_file}")

    group_mean_file = output_dir / "group_mean_voxel_by_roi_fisherz.npy"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {output_dir}")
    if args.check_only:
        print(
            f"PASS check-only: subjects={len(subjects)} voxels={seed_voxels} "
            f"rois={len(rois)} output={output_dir}"
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = output_dir / "individual_fingerprints"
    individual_dir.mkdir()
    seed_masker = NiftiMasker(
        mask_img=str(seed_mask_path),
        standardize=False,
        detrend=False,
        smoothing_fwhm=None,
        verbose=0,
    )

    fingerprints: list[np.ndarray] = []
    qc_rows: list[dict[str, object]] = []
    for position, row in subjects.reset_index(drop=True).iterrows():
        subject = str(row["subject"])
        run = str(row["run"])
        bold_file = Path(row["output_file"])
        bold_image = nib.load(bold_file)
        if bold_image.shape[:3] != seed_image.shape or not np.allclose(
            bold_image.affine, seed_image.affine, rtol=0, atol=1e-5
        ):
            raise ValueError(f"BOLD geometry does not match prepared masks: {bold_file}")
        seed_timeseries = seed_masker.fit_transform(str(bold_file))
        roi_timeseries = extract_roi_timeseries(bold_file, rois)
        _, fisher_z, qc = compute_fingerprint(seed_timeseries, roi_timeseries)
        if fisher_z.shape != (args.expected_voxels, args.expected_rois):
            raise ValueError(f"Unexpected fingerprint shape for {subject}: {fisher_z.shape}")
        if not np.isfinite(fisher_z).all():
            raise ValueError(f"Non-finite fingerprint values for {subject}")

        invalid_features = (
            qc["seed_invalid_or_zero_variance_features"]
            + qc["roi_invalid_or_zero_variance_features"]
        )
        invalid_inputs = qc["seed_input_nonfinite_values"] + qc["roi_input_nonfinite_values"]
        if args.strict_invalid_timeseries and (invalid_features or invalid_inputs):
            raise ValueError(f"Invalid time series for {subject}: {qc}")

        output_file = individual_dir / f"{subject}_{run}_voxel_by_roi_fisherz.npy"
        np.save(output_file, fisher_z, allow_pickle=False)
        fingerprints.append(fisher_z)
        qc_rows.append(
            {
                "position": position,
                "subject": subject,
                "run": run,
                "n_timepoints": seed_timeseries.shape[0],
                "n_seed_voxels": fisher_z.shape[0],
                "n_rois": fisher_z.shape[1],
                **qc,
                "fingerprint_sha256": sha256_file(output_file),
            }
        )

    stack = np.stack(fingerprints, axis=0)
    group_mean = stack.mean(axis=0).astype(np.float32)
    np.save(group_mean_file, group_mean, allow_pickle=False)
    write_tsv(output_dir / "individual_fingerprint_qc.tsv", qc_rows)
    provenance = {
        "classification": "RECOVERED_BACKUP_DERIVED_PARAMETERIZED",
        "original_source_sha256": ORIGINAL_SOURCE_SHA256,
        "calculation_order": "Pearson r -> clip -> Fisher z per subject -> arithmetic mean in z space",
        "inverse_fisher_before_gradient": False,
        "subjects": len(subjects),
        "fingerprint_shape": list(group_mean.shape),
        "fingerprint_dtype": str(group_mean.dtype),
        "group_mean_sha256": sha256_file(group_mean_file),
        "strict_invalid_timeseries": bool(args.strict_invalid_timeseries),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: wrote {len(subjects)} fingerprints and {group_mean_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
