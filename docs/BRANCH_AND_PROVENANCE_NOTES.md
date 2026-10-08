# Branch and provenance notes

## Canonical reference orientation

The compared 2847 x 10 server gradient matrices differ only by component sign: Current G1 is the exact negative of Backup/local G1, while G2-G10 are identical. The recovered comparison script uses positive correlation with voxel z coordinate as the G1 orientation rule. The exact historical writer that produced every final orientation/bin artifact was not recovered, so release-native orientation and bin construction are explicitly marked as reconstructed.

The 375 `voxel_G1_and_10bin.tsv` values matched the Current-oriented G1 without additional standardization. Its bins use stable G1 ordering and deterministic equal-count assignment. Consumer-code and run evidence support this table as the Stage x G1 input, although the historical run log does not embed its input hash.

## Mapping and branches

The 2847-to-2608 mapping is a fixed index vector into the 2847 reference order. It has SHA-256 `bb12bfa947c09778d3d353a765a397c3a5bb17880bde98f6c147d7379289a268`. Applying this vector preserves the common-mask order and is not equivalent to discovering rows by nearest coordinates.

Definition A (reward-stage reference bins), Definition B (individual-G1 analysis), old normative bins, and current bins are separate branches. The audited discrepancies are 16 Definition A/B boundary rows and five old/current bin rows. The current-bin table has SHA-256 `26d268283702db209940198e76b8630585e54259cd742016951aaa7d8427292a`.

## Reconstructed implementations

- `build_group_mean.py` is a release reconstruction of the arithmetic mean over subject Fisher-z fingerprints. It is not presented as the historical group-mean producer.
- `build_voxel_bins.py` is a release reconstruction of stable equal-rank binning with an explicit orientation sign. It is not presented as the historical bin writer.
- `freeze_and_build_profiles.py` and `run_stage_analyses.py` are later canonical rebuild implementations whose numerical outputs were verified. They are not represented as the original historical execution scripts.
- `categorical_mixedlm_vs_gee_audit.py` is a targeted later diagnostic that harmonizes the fixed-effect design and nine-dimensional contrast across inferential frameworks.

## Unresolved provenance

The exact original producers for the historical categorical checkpoint and two profile-similarity output tables were not recovered. Consumer references and verified later reconstructions do not convert those files into recovered historical producers. The categorical primary framework remains `PRIMARY_INFERENCE_FRAMEWORK_UNRESOLVED`.
