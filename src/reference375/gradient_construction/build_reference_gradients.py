#!/usr/bin/env python3
"""Fit the frozen 375-subject reference connectivity-gradient model.

The input is the 2847 x 65 group-mean Fisher-z connectivity matrix.  The
embedding parameters match the study analysis: BrainSpace diffusion maps,
normalized-angle affinity, 10 components, no sparsity, and random_state=0.

Diffusion-map component signs are arbitrary.  The recovered reference producer
used a positive-z G1 convention; the final manuscript convention is its exact
G1 sign reversal.  Both arrays are written explicitly.  An optional frozen
2847 x 10 template can additionally stabilize component signs across software
environments before those two G1 conventions are applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


N_VOXELS = 2847
N_ROIS = 65
N_COMPONENTS = 10
APPROACH = "dm"
KERNEL = "normalized_angle"
SPARSITY = 0.0
RANDOM_STATE = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        raise ValueError("Cannot determine a component sign from a constant/invalid vector")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def orient_components(
    gradients: np.ndarray,
    z_coordinate: np.ndarray,
    template: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], float, float]:
    """Return positive-z reference and G1-reversed manuscript orientations."""
    oriented = np.asarray(gradients, dtype=np.float64).copy()
    signs = np.ones(oriented.shape[1], dtype=int)

    if template is not None:
        template = np.asarray(template, dtype=np.float64)
        if template.shape != oriented.shape:
            raise ValueError(
                f"Orientation template shape {template.shape} != gradients {oriented.shape}"
            )
        if not np.all(np.isfinite(template)):
            raise ValueError("Orientation template contains NaN/Inf")
        for component in range(oriented.shape[1]):
            if safe_corr(oriented[:, component], template[:, component]) < 0:
                oriented[:, component] *= -1
                signs[component] *= -1

    g1_z_before_lock = safe_corr(oriented[:, 0], z_coordinate)
    if g1_z_before_lock < 0:
        oriented[:, 0] *= -1
        signs[0] *= -1

    reference_g1_z = safe_corr(oriented[:, 0], z_coordinate)
    if not reference_g1_z > 0:
        raise RuntimeError("Reference G1 orientation lock failed: expected corr(G1, z) > 0")

    manuscript = oriented.copy()
    manuscript[:, 0] *= -1
    manuscript_g1_z = safe_corr(manuscript[:, 0], z_coordinate)
    if not manuscript_g1_z < 0:
        raise RuntimeError("Manuscript G1 orientation lock failed: expected corr(G1, z) < 0")
    return oriented, manuscript, signs.tolist(), reference_g1_z, manuscript_g1_z


def fit_reference_gradients(connectome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the frozen BrainSpace embedding without ROI-wise re-standardization."""
    from brainspace.gradient import GradientMaps

    matrix = np.asarray(connectome, dtype=np.float64)
    if matrix.shape != (N_VOXELS, N_ROIS):
        raise ValueError(
            f"Reference connectome must be {(N_VOXELS, N_ROIS)}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Reference connectome contains NaN/Inf")

    model = GradientMaps(
        n_components=N_COMPONENTS,
        approach=APPROACH,
        kernel=KERNEL,
        random_state=RANDOM_STATE,
    )
    model.fit(matrix, sparsity=SPARSITY)
    gradients = np.asarray(model.gradients_, dtype=np.float64)
    if gradients.shape == (N_COMPONENTS, N_VOXELS):
        gradients = gradients.T
    lambdas = np.asarray(model.lambdas_, dtype=np.float64).reshape(-1)[:N_COMPONENTS]

    if gradients.shape != (N_VOXELS, N_COMPONENTS):
        raise ValueError(f"Unexpected gradient shape: {gradients.shape}")
    if lambdas.shape != (N_COMPONENTS,):
        raise ValueError(f"Unexpected eigenvalue shape: {lambdas.shape}")
    if not np.all(np.isfinite(gradients)) or not np.all(np.isfinite(lambdas)):
        raise ValueError("Gradient output contains NaN/Inf")
    return gradients, lambdas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-connectome", required=True, type=Path)
    parser.add_argument(
        "--coordinates",
        required=True,
        type=Path,
        help="Row-aligned 2847-row TSV containing a numeric z column",
    )
    parser.add_argument(
        "--orientation-template",
        type=Path,
        help="Optional frozen 2847 x 10 array used only to stabilize component signs",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.group_connectome, args.coordinates):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.orientation_template is not None and not args.orientation_template.is_file():
        raise FileNotFoundError(args.orientation_template)

    coordinates = pd.read_csv(args.coordinates, sep="\t")
    if len(coordinates) != N_VOXELS or "z" not in coordinates.columns:
        raise ValueError("Coordinates must contain 2847 rows and a z column")
    z_coordinate = pd.to_numeric(coordinates["z"], errors="raise").to_numpy(float)
    if not np.all(np.isfinite(z_coordinate)):
        raise ValueError("Coordinate z column contains NaN/Inf")

    output_files = {
        "gradients": args.output_dir / "group_gradients.npy",
        "manuscript_gradients": args.output_dir / "group_gradients_manuscript.npy",
        "lambdas": args.output_dir / "group_lambdas.npy",
        "table": args.output_dir / "group_gradient_voxel_table.tsv",
        "manuscript_table": args.output_dir / "group_gradient_manuscript_voxel_table.tsv",
        "parameters": args.output_dir / "group_gradient_parameters.json",
    }
    occupied = [str(path) for path in output_files.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {occupied}")

    connectome = np.load(args.group_connectome, allow_pickle=False)
    raw_gradients, lambdas = fit_reference_gradients(connectome)
    template = (
        np.load(args.orientation_template, allow_pickle=False)
        if args.orientation_template is not None
        else None
    )
    gradients, manuscript_gradients, signs, reference_g1_z, manuscript_g1_z = orient_components(
        raw_gradients, z_coordinate, template
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_files["gradients"], gradients, allow_pickle=False)
    np.save(output_files["manuscript_gradients"], manuscript_gradients, allow_pickle=False)
    np.save(output_files["lambdas"], lambdas, allow_pickle=False)

    table = coordinates.copy()
    for component in range(N_COMPONENTS):
        table[f"G{component + 1}"] = gradients[:, component]
    table.to_csv(output_files["table"], sep="\t", index=False)

    manuscript_table = coordinates.copy()
    for component in range(N_COMPONENTS):
        manuscript_table[f"G{component + 1}"] = manuscript_gradients[:, component]
    manuscript_table.to_csv(output_files["manuscript_table"], sep="\t", index=False)

    parameters = {
        "input": str(args.group_connectome),
        "input_sha256": sha256_file(args.group_connectome),
        "input_shape": list(connectome.shape),
        "approach": APPROACH,
        "kernel": KERNEL,
        "n_components": N_COMPONENTS,
        "sparsity": SPARSITY,
        "random_state": RANDOM_STATE,
        "diffusion_time_explicitly_passed": False,
        "extra_roi_zscoring": False,
        "reference_g1_orientation": "corr(G1, z) > 0",
        "reference_g1_z_correlation": reference_g1_z,
        "manuscript_g1_transform": "G1_manuscript = -G1_reference; G2-G10 unchanged",
        "manuscript_g1_z_correlation": manuscript_g1_z,
        "component_sign_multipliers": signs,
        "orientation_template": (
            str(args.orientation_template) if args.orientation_template is not None else None
        ),
        "orientation_template_sha256": (
            sha256_file(args.orientation_template)
            if args.orientation_template is not None
            else None
        ),
    }
    output_files["parameters"].write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )
    print(f"gradients={output_files['gradients']}")
    print(f"manuscript_gradients={output_files['manuscript_gradients']}")
    print(f"lambdas={output_files['lambdas']}")
    print(
        f"shape={gradients.shape} reference_g1_z={reference_g1_z:.12g} "
        f"manuscript_g1_z={manuscript_g1_z:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
