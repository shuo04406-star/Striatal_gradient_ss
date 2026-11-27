source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]), winslash="/")), "_common.R"))

dat <- read_tsv(file.path(INTERNAL_DIR, "clinical_imaging_analysis_dataset.tsv"))
dat <- dat[dat$group == "SZ", ]
for (v in c("PANSS_negative", "bin10_anticipation_profile_z", "bin10_outcome_profile_z", "SMA_anticipation_rawFC_slope")) dat[[v]] <- as_num(dat[[v]])
if (nrow(dat) != 32L) stop("Primary clinical associations must use SZ cases only.")

tests <- data.frame(
  analysis_id = c("P1", "P2", "P3"),
  clinical_variable = "PANSS_negative",
  phenotype = c("bin10_anticipation_profile_z", "bin10_outcome_profile_z", "SMA_anticipation_rawFC_slope"),
  phenotype_label = c("Bin 10 anticipation HC-reference profile similarity (Fisher z)",
                      "Bin 10 outcome HC-reference profile similarity (Fisher z)",
                      "SMA anticipation individual raw-FC gradient slope"),
  seed = 20261029L + 1:3,
  stringsAsFactors = FALSE
)

rows <- lapply(seq_len(nrow(tests)), function(i) {
  t <- tests[i, ]
  z <- spearman_boot(dat, t$clinical_variable, t$phenotype, seed=t$seed, B=10000L)
  data.frame(
    analysis_id=t$analysis_id, analysis_tier="primary_prespecified", test_family="primary_clinical_associations",
    clinical_variable=t$clinical_variable, phenotype=t$phenotype, phenotype_label=t$phenotype_label,
    effective_n=z$n, model_type="Spearman rank correlation",
    effect_type="Spearman_rho", effect=z$rho, ci_low=z$ci_low, ci_high=z$ci_high,
    ci_method="10,000-resample subject-level percentile bootstrap", p_raw=z$p,
    p_holm=NA_real_, correction_family_size=3L, correction_method="Holm",
    covariates="none", bootstrap_valid_replicates=z$boot_valid,
    missing_data_rule="Pairwise observed SZ cases; no imputation",
    deviation_from_prespecified_plan="none", interpretation_scope="association_not_causation",
    stringsAsFactors=FALSE
  )
})
res <- do.call(rbind, rows)
res$p_holm <- safe_holm(res$p_raw)
write_csv_utf8(res, file.path(ANALYSIS_ROOT, "primary_clinical_associations.csv"))

append_provenance(data.frame(
  output_or_result = res$analysis_id,
  analysis_stage = "B_primary_clinical_associations",
  input_files = "clinical_imaging_analysis_dataset.tsv",
  script = "scripts/04_primary_clinical_associations.R",
  variable_source = paste(res$clinical_variable, res$phenotype, sep=" x "),
  sample_rule = paste0("SZ only; observed pairs; N=", res$effective_n),
  method = paste0("Spearman rho; percentile bootstrap CI; Holm across three prespecified tests (raw p=", signif(res$p_raw,4), ")"),
  version = "clinical_extension_v1_20261029", stringsAsFactors = FALSE
))

session_record("04_primary_clinical_associations")
message("Three prespecified primary associations completed with one Holm family.")
