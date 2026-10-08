#!/usr/bin/env python3
"""Run data-free smoke tests for release-native reconstruction utilities."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, capture_output=True, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="striatal-gradient-smoke-") as temp_name:
        temp = Path(temp_name)
        fp_dir = temp / "fingerprints"
        fp_dir.mkdir()
        arrays = []
        for index in range(3):
            array = np.arange(100, dtype=np.float32).reshape(20, 5) + index
            arrays.append(array)
            np.save(fp_dir / f"mock_{index}.npy", array, allow_pickle=False)

        group_mean = temp / "group_mean.npy"
        run(
            str(REPO / "src/reference375/functional_connectivity/build_group_mean.py"),
            "--fingerprint-dir", str(fp_dir),
            "--expected-subjects", "3",
            "--expected-voxels", "20",
            "--expected-rois", "5",
            "--output", str(group_mean),
        )
        np.testing.assert_allclose(np.load(group_mean), np.mean(arrays, axis=0), rtol=0, atol=0)

        gradients = temp / "gradients.npy"
        np.save(gradients, np.column_stack([np.linspace(-1, 1, 20), np.ones(20)]), allow_pickle=False)
        bins = temp / "bins.tsv"
        run(
            str(REPO / "src/reference375/gradient_construction/build_voxel_bins.py"),
            "--gradients", str(gradients),
            "--output", str(bins),
            "--g1-sign", "1",
            "--expected-voxels", "20",
        )
        table = pd.read_csv(bins, sep="\t")
        assert table["G1_bin"].value_counts().sort_index().tolist() == [2] * 10
        assert table["G1"].is_monotonic_increasing

    print("PASS: data-free reconstruction smoke tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
