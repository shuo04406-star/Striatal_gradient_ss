# GitHub preparation report

Date: 2026-10-08
Repository candidate: `striatal-gradient-analysis`
Overall status: **LOCAL_RELEASE_READY; REQUESTED SSH REMOTE ACCESS VERIFIED**

## Outcome

An independent release directory was created and initialized as a local Git repository on branch `main`. The original research workspace was not converted into a repository, and no source inputs were overwritten or deleted. The requested SSH remote was inspected before synchronization; its initial `main` contained only a placeholder README and an MIT License. Its history and License are retained, and no force-push is used.

The release tree contains 23 scripts:

- 20 Python scripts
- 3 Bash scripts with embedded Python
- 0 R scripts

The mutually exclusive directory counts are:

- 5 scripts under `src/reference375/`
- 9 scripts under `src/clinical64/`
- 9 scripts under `validation/` (four scientific saved-output audits and five release/QA utilities)

Release classifications are 9 `PRIMARY`, 4 `RECONSTRUCTED`, and 10 `SUPPORTING`. Historical and unverified programs are retained in the audit index but not copied into the runnable `src/` tree.

## Git and GitHub environment

- Git: available, `git version 2.54.0.windows.1`
- GitHub CLI: not installed or not on the known executable paths
- `gh auth status`: not checkable
- `gh auth setup-git`: skipped because `gh` is unavailable
- Git SSH access: verified with a read-only `ls-remote`
- Git remote: `origin` points to the requested SSH repository

No token or credential value was read or printed. GitHub CLI remains optional for later repository administration; Git transport uses the already configured SSH authentication.

## Code selection

Sixteen audited scripts were copied into the release tree and then parameterized without changing the original audit copies:

- 9 of the 18 server-recovered scripts: RC01, RC03, RC08, RC09, RC10, RC12, RC14, RC17, and RC18
- 6 audited local scripts: LC01, LC07, LC08, LC09, LC10, and LC12
- 1 later targeted categorical diagnostic with original SHA-256 `b025298d6e85711dd5e22f0b2ba4d6ba3c3b71041b093b9f566a1251de93b128`

The other nine server-recovered scripts were not placed in the runnable tree: RC02 was superseded by release-specific audit tooling; RC04-RC07 and RC11 are historical; RC13 and RC15 remain unverified/canonical-decision items; RC16 is a historical forensic driver that compares incompatible old/current Bin10 parents.

Seven release-native scripts were added: the reconstructed group-mean and voxel-bin producers plus dependency, smoke, security, embedded-Python, and manifest checks. Reconstructed implementations are explicitly labeled and never represented as historical producers.

Every retained recovered/local script is linked to its original SHA-256 in `provenance/RELEASE_CODE_MAP.tsv`. The final manifest records both that original hash and the SHA-256 of the path-parameterized release copy.

## Path and configuration changes

Private absolute paths were removed from release copies. Scripts that access private scientific inputs now require explicit command-line paths or environment variables. `config/paths.example.yaml` documents the interface without including real locations.

The release does not silently write into the code directory when a private root is missing. Recovered scripts preserve their scientific formulas and branch definitions; path edits are release-only adaptations. The original hashes remain in provenance.

## Verification performed

- Python AST/bytecode syntax: PASS for all 20 Python scripts
- Bash syntax: PASS for all 3 shell scripts using Git for Windows Bash
- Embedded Python heredoc AST parsing: PASS for all 3 shell scripts
- Synthetic smoke tests for group-mean and stable-bin utilities: PASS
- README referenced-path consistency: PASS
- Environment-variable documentation coverage: PASS
- YAML parsing for environment/CITATION/config files: PASS
- Security/path/credential/large-file scan: PASS with zero findings

The host Python 3.14 environment does not contain `statsmodels`, `patsy`, or `brainspace`, so the full dependency import check is not complete in this host interpreter. An older audited vendor directory was not compatible with the host Python ABI and was not copied. The documented Python 3.11 environment must be created before running scientific modules. This is an environment-readiness limitation, not a syntax or smoke-test failure.

## Scientific release boundary

The repository includes the principal 375-person group-G1/bin/Stage x G1 code and the 64-person Models A/B/C, normative-deviation, stage-specific, SMA/IFG, and clinical-association code. It excludes all subject-level inputs and results.

The 375 group mean can be rebuilt from the approved frozen 375 fingerprints, and prior reconstruction agreed with the frozen group mean at `r=0.9999999999998823`. The complete raw-image preprocessing was not rerun. The 64 main statistical outputs were numerically reproduced from frozen private inputs.

The categorical result remains `PRIMARY_INFERENCE_FRAMEWORK_UNRESOLVED`: robust GEE `p=0.001753`, adjusted subject-clustered OLS `p=0.008031`, and MixedLM ML LRT `p=0.930344`. No framework is elevated based on significance.

## Release decision

The contents meet the release and security gates for synchronization to the explicitly requested remote. The remote initialization history should be merged normally, the reviewed paths should be staged explicitly, and the result should be pushed without force. GitHub CLI absence is not a transport blocker because SSH access has been verified.
