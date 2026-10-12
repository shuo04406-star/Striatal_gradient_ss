#!/usr/bin/env python3
"""Build the 375-person group fingerprint as an arithmetic subject mean.

Input fingerprints remain private and are never bundled with the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.npy")
    parser.add_argument("--expected-subjects", type=int, default=375)
    parser.add_argument("--expected-voxels", type=int, default=2847)
    parser.add_argument("--expected-rois", type=int, default=65)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--output-dtype", choices=("float32", "float64"), default="float32")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(args.fingerprint_dir.glob(args.pattern))
    if len(files) != args.expected_subjects:
        raise RuntimeError(
            f"Expected {args.expected_subjects} fingerprints, found {len(files)}"
        )
    expected_shape = (args.expected_voxels, args.expected_rois)
    total = np.zeros(expected_shape, dtype=np.float64)
    manifest_rows: list[dict[str, str | int]] = []
    for path in files:
        array = np.load(path, mmap_mode="r")
        if array.shape != expected_shape:
            raise RuntimeError(f"Unexpected shape for {path.name}: {array.shape}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite value in {path.name}")
        total += np.asarray(array, dtype=np.float64)
        manifest_rows.append(
            {"file_name": path.name, "sha256": sha256_file(path), "rows": array.shape[0], "columns": array.shape[1]}
        )

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mean = (total / len(files)).astype(args.output_dtype)
    np.save(args.output, mean, allow_pickle=False)

    if args.manifest_out:
        if args.manifest_out.exists():
            raise FileExistsError(f"Refusing to overwrite {args.manifest_out}")
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("file_name", "sha256", "rows", "columns"), delimiter="\t")
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"subjects={len(files)} shape={mean.shape} output_sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
