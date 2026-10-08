#!/usr/bin/env python3

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT_VALUE = os.environ.get("STRIATAL_ANALYSIS_ROOT")
if not ROOT_VALUE:
    raise RuntimeError("Set STRIATAL_ANALYSIS_ROOT to the private analysis root")
ROOT = Path(ROOT_VALUE).resolve()

FP = ROOT / "06_fingerprints" / "group_mean_voxel_by_roi_fisherz.npy"
COORD = ROOT / "06_fingerprints" / "seed_voxel_coordinates.tsv"

REF_G = ROOT / "07_group_gradient" / "group_gradients.npy"
REF_L = ROOT / "07_group_gradient" / "group_lambdas.npy"
PARAM = ROOT / "07_group_gradient" / "group_gradient_parameters.json"

OUT = ROOT / "18_final375_gradient_reproduction_audit"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 110)
print("FINAL 375 GRADIENT EXACT REPRODUCTION AUDIT")
print("=" * 110)

# =============================================================================
# 1. CHECK INPUTS
# =============================================================================

for p in [FP, COORD, REF_G, REF_L, PARAM]:
    print(f"{p}")
    print(f"  exists = {p.exists()}")
    if not p.exists():
        raise FileNotFoundError(p)

print()

# =============================================================================
# 2. LOAD ORIGINAL SAVED PARAMETERS
# =============================================================================

print("=" * 90)
print("A. SAVED 375 PARAMETERS")
print("=" * 90)

with open(PARAM, "r", encoding="utf-8") as f:
    params = json.load(f)

print(json.dumps(params, indent=2, ensure_ascii=False))
print()

# =============================================================================
# 3. LOAD FINAL INPUT / REFERENCE
# =============================================================================

X = np.load(FP).astype(np.float64)
ref_g = np.load(REF_G).astype(np.float64)
ref_l = np.load(REF_L).astype(np.float64)
coords = pd.read_csv(COORD, sep="\t")

print("=" * 90)
print("B. INPUT SHAPES")
print("=" * 90)

print("Group fingerprint :", X.shape)
print("Reference gradient:", ref_g.shape)
print("Reference lambdas :", ref_l.shape)
print("Coordinates       :", coords.shape)

if X.shape != (2847, 65):
    raise RuntimeError(f"Unexpected group fingerprint shape: {X.shape}")

if ref_g.shape != (2847, 10):
    raise RuntimeError(f"Unexpected reference gradient shape: {ref_g.shape}")

if ref_l.shape != (10,):
    raise RuntimeError(f"Unexpected reference lambda shape: {ref_l.shape}")

if len(coords) != 2847:
    raise RuntimeError(f"Coordinate rows != 2847: {len(coords)}")

if "z" not in coords.columns:
    raise RuntimeError("seed_voxel_coordinates.tsv has no z column")

print()

# =============================================================================
# 4. BRAINSPACE VERSION
# =============================================================================

import brainspace
from brainspace.gradient import GradientMaps

print("=" * 90)
print("C. BRAINSPACE")
print("=" * 90)

print("brainspace version =", getattr(brainspace, "__version__", "UNKNOWN"))
print()

# =============================================================================
# 5. REPRODUCE EXACT FINAL 375 PROCEDURE
# =============================================================================

print("=" * 90)
print("D. RE-RUN FINAL 375 GRADIENT")
print("=" * 90)

gm = GradientMaps(
    n_components=10,
    approach="dm",
    kernel="normalized_angle",
    random_state=0
)

# EXACTLY as the final 375 script:
# no ROI-dimension z-scoring
# sparsity = 0.0
gm.fit(X, sparsity=0.0)

g = np.asarray(gm.gradients_, dtype=np.float64)
l = np.asarray(gm.lambdas_, dtype=np.float64)

if g.shape[0] != X.shape[0]:
    g = g.T

print("Reproduced gradients:", g.shape)
print("Reproduced lambdas  :", l.shape)

# =============================================================================
# 6. EXACT ORIGINAL G1 POLARITY RULE
# =============================================================================

rz_before = pearsonr(g[:, 0], coords["z"].to_numpy(dtype=float))[0]

g1_flipped = False

if rz_before < 0:
    g[:, 0] *= -1.0
    g1_flipped = True

rz_after = pearsonr(g[:, 0], coords["z"].to_numpy(dtype=float))[0]

print()
print("G1 r(z) before =", rz_before)
print("G1 flipped     =", g1_flipped)
print("G1 r(z) after  =", rz_after)

# =============================================================================
# 7. COMPONENT-BY-COMPONENT COMPARISON
# =============================================================================

print()
print("=" * 90)
print("E. REPRODUCED vs FROZEN 375 REFERENCE")
print("=" * 90)

rows = []

for k in range(10):

    r = pearsonr(g[:, k], ref_g[:, k])[0]

    sign = 1.0 if r >= 0 else -1.0

    g_sign = g[:, k] * sign

    r_sign = pearsonr(g_sign, ref_g[:, k])[0]

    diff = g_sign - ref_g[:, k]

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))

    rows.append({
        "component": f"G{k+1}",
        "raw_r": r,
        "sign_to_reference": int(sign),
        "sign_corrected_r": r_sign,
        "rmse_after_sign": rmse,
        "max_abs_diff_after_sign": max_abs,
    })

res = pd.DataFrame(rows)

print(
    res.to_string(
        index=False,
        float_format=lambda x: f"{x:.12g}"
    )
)

res.to_csv(
    OUT / "gradient_reproduction_component_qc.tsv",
    sep="\t",
    index=False
)

# =============================================================================
# 8. LAMBDA COMPARISON
# =============================================================================

print()
print("=" * 90)
print("F. EIGENVALUE / LAMBDA COMPARISON")
print("=" * 90)

lambda_rows = []

for k in range(min(len(l), len(ref_l))):

    d = float(l[k] - ref_l[k])

    lambda_rows.append({
        "component": f"G{k+1}",
        "reference_lambda": float(ref_l[k]),
        "reproduced_lambda": float(l[k]),
        "absolute_difference": abs(d),
    })

ldf = pd.DataFrame(lambda_rows)

print(
    ldf.to_string(
        index=False,
        float_format=lambda x: f"{x:.15g}"
    )
)

ldf.to_csv(
    OUT / "gradient_reproduction_lambda_qc.tsv",
    sep="\t",
    index=False
)

# =============================================================================
# 9. GLOBAL QC
# =============================================================================

min_abs_component_r = float(
    np.min(np.abs(res["raw_r"].to_numpy(dtype=float)))
)

min_sign_corrected_r = float(
    np.min(res["sign_corrected_r"].to_numpy(dtype=float))
)

max_rmse = float(
    np.max(res["rmse_after_sign"].to_numpy(dtype=float))
)

max_component_diff = float(
    np.max(res["max_abs_diff_after_sign"].to_numpy(dtype=float))
)

max_lambda_diff = float(
    np.max(ldf["absolute_difference"].to_numpy(dtype=float))
)

print()
print("=" * 90)
print("G. FINAL REPRODUCTION SUMMARY")
print("=" * 90)

print(f"Minimum |raw component r|       = {min_abs_component_r:.12g}")
print(f"Minimum sign-corrected r        = {min_sign_corrected_r:.12g}")
print(f"Maximum component RMSE          = {max_rmse:.12g}")
print(f"Maximum component absolute diff = {max_component_diff:.12g}")
print(f"Maximum lambda absolute diff    = {max_lambda_diff:.12g}")

# Structural reproduction criterion.
# Sign is mathematically arbitrary, hence comparison after sign correction.
gradient_pass = bool(min_sign_corrected_r >= 0.999999)

print()
print(f"GRADIENT_REPRODUCTION = {'PASS' if gradient_pass else 'REVIEW'}")

# =============================================================================
# 10. SAVE REPRODUCED ARRAYS WITHOUT TOUCHING FROZEN REFERENCE
# =============================================================================

np.save(
    OUT / "reproduced_group_gradients.npy",
    g
)

np.save(
    OUT / "reproduced_group_lambdas.npy",
    l
)

summary = {
    "input_group_fingerprint": str(FP),
    "input_shape": list(X.shape),

    "frozen_reference_gradient": str(REF_G),
    "frozen_reference_gradient_shape": list(ref_g.shape),

    "frozen_reference_lambdas": str(REF_L),
    "frozen_reference_lambda_shape": list(ref_l.shape),

    "brainspace_version_current_environment":
        getattr(brainspace, "__version__", "UNKNOWN"),

    "reproduction_parameters": {
        "n_components": 10,
        "approach": "dm",
        "kernel": "normalized_angle",
        "sparsity": 0.0,
        "random_state": 0,
        "zscore_roi_dimension": False,
        "diffusion_time_explicitly_passed": False,
    },

    "g1_polarity_rule": "flip G1 if Pearson r(G1, voxel_z) < 0",
    "g1_r_z_before": float(rz_before),
    "g1_flipped": bool(g1_flipped),
    "g1_r_z_after": float(rz_after),

    "minimum_absolute_raw_component_r":
        min_abs_component_r,

    "minimum_sign_corrected_component_r":
        min_sign_corrected_r,

    "maximum_component_rmse_after_sign":
        max_rmse,

    "maximum_component_absolute_difference_after_sign":
        max_component_diff,

    "maximum_lambda_absolute_difference":
        max_lambda_diff,

    "gradient_reproduction":
        "PASS" if gradient_pass else "REVIEW"
}

with open(
    OUT / "FINAL_375_GRADIENT_REPRODUCTION_SUMMARY.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print()
print("Outputs:")
print(OUT / "gradient_reproduction_component_qc.tsv")
print(OUT / "gradient_reproduction_lambda_qc.tsv")
print(OUT / "FINAL_375_GRADIENT_REPRODUCTION_SUMMARY.json")

print()
print("=" * 110)
print("DONE")
print("=" * 110)
