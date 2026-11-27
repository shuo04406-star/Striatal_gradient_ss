# Striatal Functional Connectivity Gradient Pipeline

This repository contains the executable analysis code for a 375-participant
striatal reference gradient and a 64-participant individual-gradient analysis.
It is a code release: imaging data, subject-level matrices, clinical data,
atlases, masks, and frozen result arrays are not distributed.

## Pipeline at a glance

    375 preprocessed resting-state runs
      -> matched denoising
      -> 2847 striatal-voxel + 65 ROI time series
      -> per-subject Pearson r
      -> per-subject Fisher z
      -> elementwise mean of 375 Fisher-z matrices
      -> BrainSpace normalized-angle diffusion map (10 components)
      -> explicit G1 orientation and 10-bin definitions

    64 preprocessed runs
      -> the same denoising model
      -> 2608 x 65 Fisher-z fingerprints in the common voxel space
      -> individual 10-component diffusion-map embeddings
      -> component-wise sign orientation to the mapped 375 reference
      -> centered 10-dimensional orthogonal Procrustes alignment
      -> branch-specific gradient/bin profiles and statistical analyses

The reference connectome is the mean of subject-level Fisher-z connectivity
matrices. It is not an average of subject-level gradient embeddings. No inverse
Fisher transform is applied before gradient fitting.

## Main entry points

| Stage | Entry point | Principal output |
|---|---|---|
| Reference denoising | src/reference375/preprocessing/denoise_selected_runs.py | Denoised BOLD and run QC |
| 2847-voxel/65-ROI masks | src/reference375/functional_connectivity/prepare_reference_masks.py | Prepared masks, ROI manifest, voxel coordinates |
| 375 subject connectomes | src/reference375/functional_connectivity/build_reference_connectome.py | 375 individual 2847 x 65 Fisher-z matrices and their group mean |
| Group mean from frozen matrices | src/reference375/functional_connectivity/build_group_mean.py | 2847 x 65 group mean |
| Reference gradients | src/reference375/gradient_construction/build_reference_gradients.py | Reference/manuscript 2847 x 10 gradients and eigenvalues |
| Equal-rank reference bins | src/reference375/gradient_construction/build_voxel_bins.py | Row-aligned G1/10-bin table |
| 64-person matched denoising | src/clinical64/preprocessing/run_matched_denoising.py | Denoised clinical-cohort BOLD |
| 64-person connectivity | src/clinical64/functional_connectivity/build_timeseries_and_fingerprints.py | 64 x 2608 x 65 Fisher-z stack |
| Reference common-space map | src/clinical64/individual_gradients/map_reference_to_common.py | Directly indexed 2608 x 10 reference |
| Individual gradients/alignment | src/clinical64/individual_gradients/run_fixed_reference_gradients.py | Raw, sign-oriented, and Procrustes-aligned subject gradients |
| Reward-stage bin profiles | src/clinical64/reward_stage/freeze_and_build_profiles.py | Subject x bin x ROI/stage tables |
| Reward-stage models | src/clinical64/reward_stage/run_stage_analyses.py | Models A/B/C and related tables |

The remaining programs under src/clinical64 are the retained current-bin,
stage-specific, SMA/IFG, and frozen old-bin downstream analyses. They are kept
separate because they consume different, prespecified bin definitions.

## Environment

Python 3.11 is recommended.

    conda env create -f environments/environment.yml
    conda activate striatal-gradient-analysis

or:

    python -m venv .venv
    python -m pip install -r environments/requirements.txt

The core shell entry points require Bash. Copy config/paths.example.yaml
outside version control and fill it with private paths, or use the environment
variables named in that template.

## Reference pipeline

### 1. Denoise the selected 375 runs

The input TSV requires subject, run, bold_file, and confounds_file. Defaults
reproduce the retained preprocessing: 24 motion regressors, first five aCompCor
regressors, FD spike regressors above 0.5 mm, removal of non-steady-state
volumes, detrending, 0.01-0.08 Hz filtering, no smoothing, and no global-signal
regression.

    python src/reference375/preprocessing/denoise_selected_runs.py \
      --input-tsv /private/reference_runs.tsv \
      --out-root /private/reference_denoised \
      --qc-out /private/reference_denoising_qc.tsv \
      --n-proc 1

### 2. Prepare the striatal seed and ordered ROI masks

roi_source_masks.tsv requires roi_id, roi_name, and mask_file. Masks are
nearest-neighbour resampled to the supplied reference grid and binarized at
greater than zero. The 2847-row seed is the union of VST, Caudate, and Putamen.
The expected 65-column order is documented in
config/reference_roi_order_65.tsv.

Run the no-write preflight first:

    python src/reference375/functional_connectivity/prepare_reference_masks.py \
      --reference-image /private/reference_grid_or_denoised_bold.nii.gz \
      --vst-mask /private/VST.nii.gz \
      --caudate-mask /private/Caudate.nii.gz \
      --putamen-mask /private/Putamen.nii.gz \
      --roi-source-manifest /private/roi_source_masks.tsv \
      --output-dir /private/prepared_reference_masks \
      --check-only

After reviewing the inputs, repeat without --check-only.

### 3. Build subject fingerprints and the reference connectome

reference_subjects.tsv requires subject and output_file; run is optional.
reference_roi_manifest.tsv requires roi_id, roi_name, and either prepared_mask
or mask_file.

    python src/reference375/functional_connectivity/build_reference_connectome.py \
      --subject-manifest /private/reference_subjects.tsv \
      --seed-mask /private/prepared_reference_masks/whole_striatum_seed_mask_current_ref.nii.gz \
      --roi-manifest /private/prepared_reference_masks/roi65_manifest_current_ref.tsv \
      --output-dir /private/reference_connectome \
      --check-only

Repeat without --check-only to calculate the matrices. For every subject, seed
and ROI time series are standardized over time with sample SD (ddof=1), Pearson
correlation is computed as the standardized dot product divided by T-1, values
are clipped to [-0.999999, 0.999999], and arctanh is applied. The output group
mean is the arithmetic elementwise mean in Fisher-z space.

If the 375 subject matrices already exist, rebuild only their mean with:

    python src/reference375/functional_connectivity/build_group_mean.py \
      --fingerprint-dir /private/reference_connectome/individual_fingerprints \
      --output /private/reference_connectome/group_mean_voxel_by_roi_fisherz.npy

### 4. Fit and orient the reference gradients

    python src/reference375/gradient_construction/build_reference_gradients.py \
      --group-connectome /private/reference_connectome/group_mean_voxel_by_roi_fisherz.npy \
      --coordinates /private/prepared_reference_masks/seed_voxel_coordinates.tsv \
      --orientation-template /private/frozen_reference/group_gradients.npy \
      --output-dir /private/reference_gradients

The fixed parameters are approach=dm, kernel=normalized_angle,
n_components=10, sparsity=0.0, and random_state=0, with no additional
ROI-dimension z-scoring and no explicit diffusion time. The optional template
is used only to stabilize component signs; it does not rescale or reorder
components.

Because eigenvector signs are arbitrary, two explicit G1 orientations are
written:

- group_gradients.npy: the positive-z reference convention used by the retained
  individual-gradient pipeline;
- group_gradients_manuscript.npy: the final manuscript convention,
  G1_manuscript = -G1_reference, with G2-G10 unchanged.

### 5. Construct the equal-rank 10-bin table

For the manuscript-oriented array, no additional sign reversal is needed:

    python src/reference375/gradient_construction/build_voxel_bins.py \
      --gradients /private/reference_gradients/group_gradients_manuscript.npy \
      --coordinates /private/prepared_reference_masks/seed_voxel_coordinates.tsv \
      --g1-sign 1 \
      --output /private/reference_gradients/voxel_G1_and_10bin.tsv

This entry creates the stable equal-rank 375-stage definition. The separately
frozen current-bin table used by current normative/stage/SMA-IFG analyses is a
distinct input and must not be silently regenerated or substituted.

## 64-participant individual pipeline

The retained 64-person programs use environment variables so that private
paths never enter the repository:

    export STRIATAL_CLINICAL64_DENOISE_ROOT=/private/clinical64_denoising
    export STRIATAL_CURRENT64_MANIFEST=/private/current64_input_manifest.tsv
    export STRIATAL_CURRENT64_GROUP_TABLE=/private/current64_group_table.tsv
    python src/clinical64/preprocessing/run_matched_denoising.py

    export STRIATAL_ANALYSIS_ROOT=/private/analysis_root
    python src/clinical64/functional_connectivity/build_timeseries_and_fingerprints.py

The connectivity program applies the validated, unique, ascending 2847-to-2608
row-index mapping. It does not reconstruct the common space by nearest-coordinate
matching. The mapping can also be inspected independently:

    python src/clinical64/individual_gradients/map_reference_to_common.py \
      --reference-gradients /private/reference_gradients/group_gradients.npy \
      --row-mapping /private/row_mapping_2847_to_2608.npy \
      --output /private/reference_gradients/group_gradients_2608x10.npy

Run the verified individual-gradient implementation with the same analysis
root and the 2608-space mask:

    export STRIATAL_CURRENT2608_MASK=/private/current64_analysis_mask.nii.gz
    python src/clinical64/individual_gradients/run_fixed_reference_gradients.py

Each 2608 x 65 subject fingerprint is embedded with the reference settings.
Components are sign-oriented to the corresponding mapped reference component;
no permutation is applied. The sign-oriented gradients are cast through the
historical float32 intermediate, both subject and reference matrices are
column-centered, and a ten-component orthogonal Procrustes rotation is solved
by SVD. Scaling is disabled, reflection is allowed, and there is no
post-Procrustes sign flip. The 64-person group gradient produced by that script
is QC-only and is never used as the alignment reference.

## Downstream gradient/bin analyses

The 375-participant Stage x G1 mixed model is implemented in
src/reference375/statistical_analysis/run_stage_g1_mixed_model.py.

The canonical reward-stage profile builder should always be preflighted before
writing:

    python src/clinical64/reward_stage/freeze_and_build_profiles.py \
      --project-root /private/project \
      --output-root /private/current64_rerun \
      --check-only

After input review, omit --check-only, set
STRIATAL_CURRENT64_ROOT=/private/current64_rerun, and run
src/clinical64/reward_stage/run_stage_analyses.py. Current-bin normative,
stage-specific, and SMA/IFG analyses are separate shell entry points:

    src/clinical64/normative_deviation/run_current_bin_analysis.sh
    src/clinical64/reward_stage/run_stage_specific_current_bins.sh
    src/clinical64/sma_ifg/run_bin10_subfamily.sh

src/clinical64/normative_deviation/run_normative_old_bins.py is retained only
for reproduction of the frozen old-bin branch. Do not combine its bin labels
with current-bin outputs.

## Supplementary resting-state analyses

| Module | Main entry point |
|---|---|
| Reference validation | [Anatomical bins](src/reference375/validation/plot_G1_bins_by_anatomy_2847.py) and [random slopes](src/reference375/validation/02_stage_g1_random_slope.py) |
| ROI and model sensitivity | [ROI deletion](src/clinical64/sensitivity_analysis/roi_deletion/run_step3a_subfamily_robustness.py), [Insula contributions](src/clinical64/sensitivity_analysis/insula_contributions/run_local_contribution_sensitivity.py) and [MixedLM/GEE](src/clinical64/sensitivity_analysis/mixedlm_vs_gee/01_same_data_method_audit.py) |
| Independent regional FC | [SMA, IFG and Cingulate curves and slopes](src/clinical64/regional_analysis/three_regions/run_three_region_rawFC_validation.R) |
| Final64 database | [Subject-level export](src/clinical64/data_preparation/build_subject_level_exports.py) and [database lock and QC](src/clinical64/data_preparation/build_final64_lock_audit.py) |
| Clinical associations | [Clinical extension](src/clinical64/clinical_associations/clinical_extension/01_audit_clinical_data.R), [stage coupling](src/clinical64/clinical_associations/stage_coupling/run_all_stage_coupling.py) and [measurement precision](src/clinical64/clinical_associations/measurement_precision/run_all_measurement_precision.py) |

The [supplementary analysis guide](docs/SUPPLEMENTARY_ANALYSES.md) documents
the frozen inputs, clinical scoring and matching, model and FC sensitivity
branches, plotting workflows, and resting-state BIDS assembly.

## Input checks and data boundary

The calculation scripts check expected subject, voxel, ROI, component, mapping,
and finite-value constraints before writing. Output directories are protected
against accidental overwrite where supported. See
docs/DATA_EXCLUSION_POLICY.md for the files that must remain outside Git.

The upstream mask/connectome entries are explicit-input parameterizations of
the recovered computation, not a distribution of the private source data. The
exact frozen inputs and approved orientation template are therefore required
for numerical reproduction of the reported arrays.

Software is released under the MIT License; that license does not grant
permission to redistribute participant data or third-party atlas resources.
