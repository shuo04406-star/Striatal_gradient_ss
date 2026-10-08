# GitHub reproducibility status

This document distinguishes scientific numerical reproduction from code review and data-free smoke testing. A passing synthetic smoke test does not reproduce a scientific result.

| Module | Branch | Status | What is reproducible | Remaining limitation |
|---|---|---|---|---|
| 375 selected-run denoising | reference | CODE_REVIEW_ONLY | Syntax and recovered denoising implementation | Raw/fMRIPrep inputs and full preprocessing rerun are excluded |
| 375 group-mean FC | reference | VERIFIED_FROM_FROZEN_INPUTS | Arithmetic mean of 375 Fisher-z 2847 x 65 fingerprints; prior mean reconstruction `r=0.9999999999998823` | Frozen subject fingerprints are private; release producer is reconstructed, not historical |
| 375 group G1 | reference | NUMERICALLY_EQUIVALENT_WITH_TRANSFORM | BrainSpace reconstruction from frozen group mean; G1 sign transform and G2-G10 identity audited | Exact historical final orientation writer not recovered |
| 375 G1 ten bins | reference | VERIFIED_ARTIFACT_AND_CONSUMER_LINEAGE | Current-oriented G1, no extra standardization, stable equal-rank bins, Stage x G1 consumer traced | Release writer is reconstructed; historical execution log lacks an input hash |
| 375 Stage x G1 model | reference | VERIFIED_FROM_FROZEN_TABLE | All 28 refitted statistics passed; maximum absolute difference `9.37e-17` | Upstream 7500-row table contains subject-level data and is not distributed |
| 64 matched denoising | clinical | CODE_REVIEW_ONLY | Recovered code and derived-artifact lineage | Raw images/confounds are private |
| 64 FC and 2847-to-2608 mapping | clinical | VERIFIED_METHOD; PRIVATE_INPUT_REQUIRED | Exact index mapping and 2608 x 65 construction logic | Subject time series and FC arrays are private |
| 64 individual gradients | Definition B | CODE_REVIEW_ONLY_WITH_VERIFIED_METHOD | Fixed-375 reference, sign orientation, float32 intermediate, ten-component Procrustes | Subject FC/gradient arrays are private |
| 64 reward-stage profiles | Definition A | VERIFIED_FROM_FROZEN_INPUTS | 64 subjects, 1280-row stage table, 43/19/3 ROI split | Rebuild implementation is later equivalent code, not the historical producer |
| 64 Models A/B/C | Definition A | VERIFIED | Coefficients, simple effects, and likelihood-ratio results numerically reproduced | Frozen input table is private; retained MixedLM boundary warnings must be reported |
| Categorical Group x Stage x Bin | categorical | PRIMARY_INFERENCE_FRAMEWORK_UNRESOLVED | Same 43-column design and nine-dimensional null audited across frameworks | GEE `p=0.001753`, adjusted clustered OLS `p=0.008031`, MixedLM LRT `p=0.930344`; primary framework not historically established |
| Old normative deviation | old bins | VERIFIED_FROM_FROZEN_INPUTS | Subject metrics and model/sensitivity fields independently reconstructed | Historical standalone robust producer remains missing |
| Current-bin normative parent | current bins | VERIFIED_FROM_FROZEN_INPUTS | Current table/mapping, metrics, omnibus and bin contrasts checked | Private FC/covariates required; historical log lacks embedded input hash |
| Current Group x Bin x Stage | current bins | VERIFIED_FROM_FROZEN_INPUTS | Stage-specific GEE and clustered-OLS outputs checked | Does not identify the missing historical categorical checkpoint producer |
| Bin10 SMA/IFG contribution | current bins | VERIFIED_FROM_FROZEN_INPUTS | Exact contributions and 13 assigned-subfamily FDR tests checked | Subject-level contribution tables are private |
| Clinical associations | Definition A slopes | VERIFIED_FROM_FROZEN_INPUTS | Four available-case analyses checked, N=`57/58/60/59` | TEPS/ACIPS and participant-level behavior are private; none survive the stated four-test FDR family |
| End-to-end raw-image reproduction | both | NOT_RUN | No claim | Out of release scope; raw images and preprocessing inputs excluded |

## Branch rules

Definition A reward-stage bins, Definition B individual G1, old normative bins, and current bins remain separate. The 16 Definition A/B boundary differences and five old/current bin differences are not harmonized away. The 2847-to-2608 mapping is a fixed row-index selection, not a coordinate-nearest-neighbor operation.

## Practical interpretation

Modules labeled `VERIFIED_FROM_FROZEN_INPUTS` can be numerically rerun only by an authorized holder of the private frozen inputs. Modules labeled `CODE_REVIEW_ONLY` are publishable as methods code but cannot be independently executed from repository contents alone. The public-facing repository must therefore describe itself as an audited code release with controlled-data reproduction instructions, not as a fully self-contained data-and-code reproduction package.
