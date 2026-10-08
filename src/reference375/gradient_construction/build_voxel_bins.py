#!/usr/bin/env python3
"""Create deterministic equal-rank G1 bins from a 2847 x 10 gradient matrix.

This is a reconstructed, verified-equivalent release implementation, not the
unrecovered historical orientation/bin writer. The G1 sign must be selected
explicitly and documented for every run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def equal_rank_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float), kind="stable")
    labels = np.floor(np.arange(len(order)) * n_bins / len(order)).astype(int) + 1
    labels = np.clip(labels, 1, n_bins)
    bins = np.empty(len(order), dtype=int)
    bins[order] = labels
    return bins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, help="Optional row-aligned TSV")
    parser.add_argument("--g1-sign", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--expected-voxels", type=int, default=2847)
    parser.add_argument("--n-bins", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gradients = np.load(args.gradients, allow_pickle=False)
    if gradients.ndim != 2 or gradients.shape[0] != args.expected_voxels:
        raise RuntimeError(f"Expected ({args.expected_voxels}, components), found {gradients.shape}")
    if gradients.shape[1] < 1 or not np.isfinite(gradients[:, 0]).all():
        raise RuntimeError("G1 is missing or non-finite")

    g1 = np.asarray(gradients[:, 0], dtype=np.float64) * args.g1_sign
    frame = pd.DataFrame(
        {
            "voxel_index0": np.arange(args.expected_voxels, dtype=int),
            "G1": g1,
            "G1_bin": equal_rank_bins(g1, args.n_bins),
        }
    )
    if args.coordinates:
        coords = pd.read_csv(args.coordinates, sep="\t")
        if len(coords) != len(frame):
            raise RuntimeError("Coordinate rows are not aligned with gradient rows")
        frame = pd.concat([frame, coords.reset_index(drop=True)], axis=1)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, sep="\t", index=False, float_format="%.17g")
    counts = frame["G1_bin"].value_counts().sort_index().to_dict()
    print(f"rows={len(frame)} g1_sign={args.g1_sign} bin_counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
