# GitHub security scan

Date: 2026-10-08
Result: **PASS — no sensitive or prohibited files detected in the release tree**

## Checks performed

The final release tree was scanned recursively while excluding `.git` internals and ignored Python cache files. Checks covered:

- raw/denoised imaging and array extensions (`.nii`, `.nii.gz`, `.npy`, `.npz`, `.mat`, `.h5`, `.hdf5`, `.pkl`, `.pickle`);
- private Windows/server absolute paths;
- GitHub-token, AWS-key, private-key, and generic secret-assignment patterns;
- literal subject identifiers matching `sub-` followed by three or more digits;
- files larger than 10 MiB;
- configuration files that could hold local secrets;
- repository status and remote configuration.

Final findings: zero blocked data files, zero private absolute paths, zero credential/key matches, zero literal subject-ID matches, and zero files above 10 MiB. The only configured remote is the explicitly requested SSH repository. No unreviewed content is staged.

## Exclusion controls

`.gitignore` blocks common neuroimaging/array formats, local input/output directories, local path configuration, environment files, keys, caches, and editor artifacts. The documentation separately prohibits subject-level FC/time-series/gradient data, diagnosis/demographic/behavioral tables, TEPS/ACIPS values, credentials, and masks/atlases lacking redistribution permission.

The repository contains only code, dependency specifications, path templates, documentation, code-provenance tables, and aggregate QC/reproducibility summaries. The aggregate 375 group-mean reconstruction QC contains four summary values and no participant rows.

## Residual risk and required practice

Pattern scanning cannot establish a third-party atlas/mask license or detect every possible disclosure embedded in a future file. Before any commit, regenerate `GITHUB_FILE_MANIFEST.tsv`, rerun `validation/security_scan.py`, inspect `git status --short`, and stage only explicitly reviewed paths. Do not copy outputs from private analysis roots into this repository.

GitHub CLI authentication was not tested because GitHub CLI is unavailable. Git SSH transport was verified without reading or printing credential material.
