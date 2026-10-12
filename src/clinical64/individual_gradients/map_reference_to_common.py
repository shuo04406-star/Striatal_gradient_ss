#!/usr/bin/env python3
"""Apply the validated 2847-to-2608 row-index mapping to reference gradients."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


N_REFERENCE_VOXELS = 2847
N_COMMON_VOXELS = 2608
N_COMPONENTS = 10


def map_reference(reference: np.ndarray, row_mapping: np.ndarray) -> np.ndarray:
    gradients = np.asarray(reference)
    mapping = np.asarray(row_mapping)
    if gradients.shape != (N_REFERENCE_VOXELS, N_COMPONENTS):
        raise ValueError(
            f"Reference gradients must be {(N_REFERENCE_VOXELS, N_COMPONENTS)}, "
            f"got {gradients.shape}"
        )
    if mapping.shape != (N_COMMON_VOXELS,):
        raise ValueError(f"Row mapping must have shape {(N_COMMON_VOXELS,)}, got {mapping.shape}")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise TypeError(f"Row mapping must use an integer dtype, got {mapping.dtype}")
    mapping = mapping.astype(np.int64, copy=False)
    if mapping.min() < 0 or mapping.max() >= N_REFERENCE_VOXELS:
        raise IndexError("Row mapping contains an out-of-range 2847-space index")
    if len(np.unique(mapping)) != N_COMMON_VOXELS:
        raise ValueError("Row mapping must contain 2608 unique indices")
    if not np.all(np.diff(mapping) > 0):
        raise ValueError("Row mapping must preserve the validated ascending voxel order")
    if not np.all(np.isfinite(gradients)):
        raise ValueError("Reference gradients contain NaN/Inf")
    return gradients[mapping]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-gradients", required=True, type=Path)
    parser.add_argument("--row-mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    reference = np.load(args.reference_gradients, allow_pickle=False)
    mapping = np.load(args.row_mapping, allow_pickle=False)
    mapped = map_reference(reference, mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, mapped, allow_pickle=False)
    print(f"output={args.output} shape={mapped.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
