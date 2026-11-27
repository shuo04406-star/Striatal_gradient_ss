# Supplementary resting-state analyses

Run these modules with authorized inputs and a separate private output or
working directory. Some historical scripts use fixed subdirectories and can
overwrite a previous result in the same working tree. Keep the frozen inputs
read-only and use a copy of the expected layout for each historical run.

## Reference cohort (375)

| Entry point | Inputs and purpose |
|---|---|
| [Random-slope sensitivity](../src/reference375/validation/02_stage_g1_random_slope.py) | Frozen subject-by-stage-by-bin FC and random-intercept results. Fits the historical random-slope model and writes model comparisons. |
| [Individual slopes and paired bin tests](../src/reference375/validation/03_individual_slope_and_binwise_tests.py) | The same FC table and random-slope summary. Writes participant slopes and paired contrasts. |
| [Anatomical bin validation](../src/reference375/validation/plot_G1_bins_by_anatomy_2847.py) | Frozen 2847-voxel G1 and bins, coordinates and approved region masks. Generates bin composition, localization plots and checks. |
| [Manuscript Figure 2](../src/visualization/reference375/draw_core_figure2_375.py) | Frozen stage FC and private expected slopes for QC. Renders the three panels and their plotting tables. |
| [Historical Figure 3](../src/visualization/reference375/01_finalize_figure3_and_results.py) | Frozen stage-family metadata, stage mixed model, random slopes and paired-bin tables. Builds the legacy multi-panel figure and publication tables. |

The root for the historical validation scripts is **STRIATAL_HISTORICAL_ROOT**. The
random-slope and paired tests read the stage table under
**25_subject_level_stage_g1_model1/tables** and write into sibling subdirectories.
For Figure 2, use **--input** and **--out**. Its three reproduction checks read
**STRIATAL_EXPECTED_ANTICIPATION_SLOPE**, **STRIATAL_EXPECTED_OUTCOME_SLOPE** and
**STRIATAL_EXPECTED_SLOPE_DIFFERENCE** from private configuration.

## Final64 data preparation and upstream utilities

| Entry point | Inputs and purpose |
|---|---|
| [Subject-level export](../src/clinical64/data_preparation/build_subject_level_exports.py) | Extracts participant FC profiles, slopes and similarities from frozen analyses. Joins clinical, questionnaire and previously prepared behavioral records. |
| [Freeze and audit Final64](../src/clinical64/data_preparation/build_final64_lock_audit.py) | Cross-checks exports against frozen clinical and imaging sources. Writes the private analysis-ready master, sample matrix, and manifest. |
| [Build rest-only single-T1w BIDS](../src/clinical64/data_preparation/upstream/build_rest_singleT1_v1.py) | Historical rest-only BIDS assembly using already selected T1 scans and a private cohort configuration. The PowerShell wrapper is in the same directory. |
| [T1 series repair](../src/clinical64/data_preparation/t1_repair/01_verify_subject_mapping.py) | Historical SeriesInstanceUID repair, with steps 01–09 in the same directory. Uses authorized DICOM, BIDS and v1 QC inputs. |
| [fMRIPrep rerun template](../src/clinical64/data_preparation/upstream/rerun_fmriprep_repaired_subjects_v2_TEMPLATE.sh) | Resting-state T1-repair rerun configuration from the local record. Requires an approved private participant list, BIDS tree and FreeSurfer license. |

Set **STRIATAL_CLINICAL_SOURCE_ROOT** to the authorized clinical working tree
and **STRIATAL_PROJECT_ROOT** to the authorized frozen analysis tree. The
export reads the three private workbooks from
**STRIATAL_BEHAVIOR_AUDIT_WORKBOOK**, **STRIATAL_REGISTRY_WORKBOOK** and
**STRIATAL_RAW_SCALE_WORKBOOK**. The rest BIDS builder reads a private JSON file
(**STRIATAL_REST_BUILD_CONFIG**) with the keys **subjects** (ordered list),
**two_run_subject**, **t1_verification_subject**, and
**selected_t1_filenames** (participant-to-filename mapping).
**STRIATAL_T1_REPAIR_CONFIG** points to a private JSON file with
**target_subjects** and **excluded_subjects** arrays for the historical T1
repair stages. The rerun template reads **STRIATAL_BIDS_DIR**,
**STRIATAL_FMRIPREP_OUTPUT**, **STRIATAL_WORK_ROOT**, **STRIATAL_FS_LICENSE** and
**STRIATAL_T1_RERUN_LIST**.

The exact scanner/run selection ledgers and cohort-wide fMRIPrep execution
record remain required for end-to-end preprocessing reproduction. The
2847-to-2608 mapping is consumed as a validated frozen input.

## Final64 sensitivities and regional analyses

| Entry point | Procedure |
|---|---|
| [ROI and subfamily deletion](../src/clinical64/sensitivity_analysis/roi_deletion/run_step3a_subfamily_robustness.py) | Continuous G1 Stage-by-Group model with ROI or subfamily deletion. |
| [Count-matched deletion](../src/clinical64/sensitivity_analysis/count_matched_deletion/run_step3b_analysis.py) | Within-stage, count-matched deletion and comparison with the original model. Reuses the ROI deletion code and outputs. |
| [Equal-subfamily weighting](../src/clinical64/sensitivity_analysis/equal_subfamily_weighting/run_step3d_equal_subfamily_weighting.py) | Compares the original ROI weighting to equal subfamily weighting. |
| [Insula contributions](../src/clinical64/sensitivity_analysis/insula_contributions/run_local_contribution_sensitivity.py) | Recomputes the Bin 10 decomposition of Pearson profile similarity after deleting Insula. |
| [Leave-one-subfamily-out contributions](../src/clinical64/sensitivity_analysis/leave_one_subfamily_out/run_leave_one_subfamily_out.py) | Recomputes normalization, similarity and contributions after each deletion, using the fixed correction family. |
| [Covariate sensitivity](../src/clinical64/sensitivity_analysis/covariate_models/01_final_gradient_sensitive_covariate_model.py) | Historical continuous models with different covariate formulas. |
| [MixedLM versus GEE](../src/clinical64/sensitivity_analysis/mixedlm_vs_gee/01_same_data_method_audit.py) | Compares model-based and cluster-robust inference using identical repeated observations. |
| [Finite-sample GEE](../src/clinical64/sensitivity_analysis/gee_finite_sample/01_run_finite_sample_audit.py) | Uses the previous audit, bias-reduced GEE, cluster jackknife, leave-one-subject-out and wild-cluster bootstrap. |
| [Variance and dispersion](../src/clinical64/supplementary_analysis/independent_statistical_verification.py) | Independent verification of covariance models and FC variability. The companion meeting figure reads the saved STEP2 tables. |
| [Individual anatomical ordering](../src/clinical64/supplementary_analysis/anatomy/run_G1_regional_ordering.py) | Uses frozen 2608-voxel individual G1, exact row mapping and approved anatomy masks. The same directory contains anatomical separation, eta-squared and covariate adjustments. |
| [SMA and IFG FC profile](../src/clinical64/regional_analysis/profile_similarity/run_step4_profile_similarity.py) | Decomposes the FC profiles into offset, slope and shape and compares profile similarity. |
| [Regional specificity](../src/clinical64/regional_analysis/regional_specificity/run_step5_regional_specificity.py) | Compares SMA and IFG with other ROIs in their respective stages, with leave-one-out and bootstrap checks. |
| [Six-ROI IFG](../src/clinical64/regional_analysis/ifg_6roi/run_ifg_6roi_reanalysis.R) / [Three-region FC](../src/clinical64/regional_analysis/three_regions/run_three_region_rawFC_validation.R) | Run the IFG analysis first. The three-region check reads its saved comparison table and produces independent SMA, IFG and Cingulate FC curves and slope tests. |
| [Final64 manuscript figures V2](../src/visualization/clinical64/manuscript_v2/generate_main_figures_v2.py) | Renders the V2 figures from authorized source tables and figure assets. |

The existing [current-bin analysis](../src/clinical64/normative_deviation/run_current_bin_analysis.sh)
already writes the Current-versus-Old bin comparison. Use frozen Current bins
for the normative or Pearson-contribution modules and frozen Definition A for
the retained stage FC slopes. The historical Old bins remain a separate
downstream definition. The follow-up audit reads both relevant definitions.
These are resting-state ROI classifications.

## Final64 clinical associations

1. Run [the clinical extension](../src/clinical64/clinical_associations/clinical_extension/01_audit_clinical_data.R)
   in numeric order through step 07 for the earlier formal-v3 clinical source and
   screening workflow. Use **STRIATAL_CLINICAL_EXTENSION_OUTPUT** for its private
   working tree. Step 03 checks private PANSS reference summaries from
   **STRIATAL_PANSS_REFERENCE_RDS** (a named R list of **n**, **mean** and **sd**
   for each PANSS endpoint). Source conflict decisions come from the saved audit.
2. From the Final64 locked inputs, run [the full 119-test analysis](../src/clinical64/clinical_associations/full_scale/run_full_reanalysis.R),
   [the subgroup and interaction audit](../src/clinical64/clinical_associations/full_scale/audit_all_scale_imaging_groups.R),
   then [the requested manuscript figures](../src/clinical64/clinical_associations/full_scale/make_requested_manuscript_figures.R).
   Set **STRIATAL_FINAL64_LOCK** to the locked dataset root and
   **STRIATAL_FULL_SCALE_OUTPUT** to a private analysis directory.
   The procedure retains the eligible association families, direct HC/SZ
   interactions, residualization and multiplicity adjustments.
3. The [TEPS/ACIPS follow-up](../src/clinical64/clinical_associations/teps_acips_followup/run_followup.py)
   audits item scoring, compares actual and covariate-predicted TEPS values,
   examines similarity direction and original FC profiles, and runs
   stage-specific, cross-stage and clinical covariate checks. Supply
   **--analysis-root**, **--project-root** and **--output-dir**. Use **--rscript**
   if Rscript is outside PATH, or **--tables-only** to omit figures. The
   analysis root uses the authorized frozen database and full-scale tables in
   their recorded directory layout. Set **STRIATAL_DISPUTED_SUBJECT_ID**
   privately from the formal source-conflict audit for the preserved
   single-participant sensitivity.
4. [Stage coupling](../src/clinical64/clinical_associations/stage_coupling/run_all_stage_coupling.py)
   and [measurement precision](../src/clinical64/clinical_associations/measurement_precision/run_all_measurement_precision.py)
   are separate supplementary workflows. They read authorized frozen profiles,
   behavior, item scores and prior clinical results. The precision analysis
   also needs frozen timeseries or fingerprints, as named in its input map.

Use **STRIATAL_OUTPUT_ROOT** for scripts that take a separate output root and
**STRIATAL_SUPPLEMENT_ROOT** for the archived STEP3–STEP5 layout shared by their
analysis and plotting scripts. **STRIATAL_CURRENT_BIN_TABLE** points to the
frozen Current table. **STRIATAL_RELEASE_ROOT** can point to this checkout when
a script hashes a published upstream script for provenance.
**STRIATAL_PRIVATE_AUDIT_ROOT** supplies retained STEP1/STEP2 provenance reports
and **STRIATAL_VARIANCE_OUTPUT** points to the saved variance tables. The
finite-sample audit reads the preceding audit fingerprint from
**STRIATAL_MIXEDLM_GEE_OUTPUT**. The V2 figure script takes
**STRIATAL_MANUSCRIPT_INPUT_ROOT** plus **STRIATAL_MANUSCRIPT_OMNIBUS_TSV**
(columns **metric**, **statistic**, **p**). Historical reproduction constants
referenced by **STRIATAL_EXPECTED_***, **STRIATAL_HISTORICAL_***,
**STRIATAL_TARGET_*** and the JSON environment settings are supplied from the
authorized archived benchmarks. Use the specific names in each entry point.
These values are required for the original comparison gates.

## Runtime

The core Python dependencies are in [requirements.txt](../environments/requirements.txt).
Additional Python packages for the supplementary modules are openpyxl,
Pillow, PyMuPDF, pydicom and tzdata where the OS does not provide the IANA
database. Series repair also requires dcm2niix. Use the effective R package
environment captured for the historical run when matching its rendering or
numerical output. The full-scale screens and follow-up use R with boot, broom,
dplyr, ggplot2, patchwork, pdftools, png, purrr, ragg, readr, sandwich, scales,
stringr, svglite, systemfonts, tibble and tidyr. The November follow-up used
Python 3.14 and R 4.6.1. Use Bash for the fMRIPrep template.

This release preserves the executed methods, archived validation variants and
independent conference/variance verification as identified above. Participant
tables, matrices, masks, benchmarks, plots and local paths stay in the
authorized data trees. The public checkout contains the code and methods needed
to identify the inputs and reproduce the supported downstream stages.
