# Data exclusion policy

Only reviewed code, non-sensitive configuration templates, aggregate QC summaries, and provenance records may enter this repository.

Do not commit:

- raw or denoised BOLD/NIfTI data;
- time series, subject-level FC matrices, or individual gradients;
- subject identifiers, diagnostic labels, demographic or motion tables;
- TEPS, ACIPS, or other participant-level clinical/behavioral values;
- passwords, tokens, private keys, `.env` files, or SSH configuration;
- private workstation/server paths;
- atlases, masks, or parcellations lacking explicit redistribution permission;
- generated outputs unless they have undergone a separate disclosure review.

Before each commit, inspect `git status --short`, regenerate `GITHUB_FILE_MANIFEST.tsv`, rerun the credential/path/large-file scan, and stage files explicitly. Do not use a broad add command in the original research directory.
