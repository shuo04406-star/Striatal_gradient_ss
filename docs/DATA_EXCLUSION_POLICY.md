# Data and privacy boundary

This repository contains analysis code, dependency specifications, and non-sensitive configuration templates only.

Do not commit:

- raw or denoised BOLD/NIfTI data;
- time series, subject-level FC matrices, or individual gradients;
- subject identifiers, diagnostic labels, demographic or motion tables;
- TEPS, ACIPS, or other participant-level clinical/behavioral values;
- passwords, tokens, private keys, `.env` files, or SSH configuration;
- private workstation/server paths;
- atlases, masks, or parcellations lacking explicit redistribution permission;
- generated outputs unless they have undergone a separate disclosure review.

Keep all paths in an untracked local configuration or environment variables. Before each release, inspect the staged file list and scan it for credentials, private absolute paths, blocked data extensions, and unexpectedly large files.
