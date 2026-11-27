source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]), winslash="/")), "_common.R"))

dat <- read_tsv(file.path(INTERNAL_DIR, "clinical_imaging_analysis_dataset.tsv"))
dat <- dat[dat$group == "SZ", ]
all_num <- c("PANSS_positive","PANSS_negative","PANSS_general","PANSS_total_recomputed","illness_duration_years",
             "age_at_onset_years","DUP_years","episode_number","first_episode_candidate",
             "bin10_anticipation_profile_z","bin10_outcome_profile_z","SMA_anticipation_rawFC_slope",
             "IFG_outcome_rawFC_slope","Cingulate_outcome_rawFC_slope","age_scan")
for (v in all_num) dat[[v]] <- as_num(dat[[v]])

phenotypes <- c("bin10_anticipation_profile_z","bin10_outcome_profile_z","SMA_anticipation_rawFC_slope",
                "IFG_outcome_rawFC_slope","Cingulate_outcome_rawFC_slope")
phenotype_labels <- c(
  bin10_anticipation_profile_z="Bin 10 anticipation similarity (Fisher z)",
  bin10_outcome_profile_z="Bin 10 outcome similarity (Fisher z)",
  SMA_anticipation_rawFC_slope="SMA anticipation raw-FC slope",
  IFG_outcome_rawFC_slope="IFG outcome raw-FC slope",
  Cingulate_outcome_rawFC_slope="Cingulate outcome raw-FC slope"
)

symptom_grid <- expand.grid(clinical_variable=c("PANSS_positive","PANSS_general","PANSS_total_recomputed"),
                            phenotype=phenotypes, stringsAsFactors=FALSE)
symptom_grid <- rbind(symptom_grid,
                      data.frame(clinical_variable="PANSS_negative",
                                 phenotype=c("IFG_outcome_rawFC_slope","Cingulate_outcome_rawFC_slope"), stringsAsFactors=FALSE))
symptom_rows <- lapply(seq_len(nrow(symptom_grid)), function(i) {
  g <- symptom_grid[i,]
  z <- spearman_boot(dat, g$clinical_variable, g$phenotype, 20261100L+i, B=10000L)
  data.frame(analysis_id=sprintf("S_SYM_%02d",i), analysis_tier="secondary", test_family="secondary_symptom_17",
             clinical_variable=g$clinical_variable, phenotype=g$phenotype, phenotype_label=phenotype_labels[[g$phenotype]],
             effective_n=z$n, model_type="Spearman rank correlation", effect_type="Spearman_rho", effect=z$rho,
             ci_low=z$ci_low, ci_high=z$ci_high, ci_method="10,000-resample subject-level percentile bootstrap",
             p_raw=z$p, p_holm=NA_real_, correction_family_size=nrow(symptom_grid), correction_method="Holm",
             covariates="none", bootstrap_valid_replicates=z$boot_valid,
             missing_data_rule="Pairwise observed SZ cases; no imputation",
             notes="Secondary symptom family specified before inspection of these results", stringsAsFactors=FALSE)
})
symptom <- do.call(rbind, symptom_rows)
symptom$p_holm <- safe_holm(symptom$p_raw)

history_grid <- expand.grid(clinical_variable=c("illness_duration_years","age_at_onset_years"),
                            phenotype=phenotypes[1:3], stringsAsFactors=FALSE)
history_rows <- lapply(seq_len(nrow(history_grid)), function(i) {
  g <- history_grid[i,]
  z <- spearman_boot(dat, g$clinical_variable, g$phenotype, 20261200L+i, B=10000L)
  data.frame(analysis_id=sprintf("S_HIST_%02d",i), analysis_tier="secondary", test_family="secondary_illness_history_6",
             clinical_variable=g$clinical_variable, phenotype=g$phenotype, phenotype_label=phenotype_labels[[g$phenotype]],
             effective_n=z$n, model_type="Spearman rank correlation", effect_type="Spearman_rho", effect=z$rho,
             ci_low=z$ci_low, ci_high=z$ci_high, ci_method="10,000-resample subject-level percentile bootstrap",
             p_raw=z$p, p_holm=NA_real_, correction_family_size=nrow(history_grid), correction_method="Holm",
             covariates="none", bootstrap_valid_replicates=z$boot_valid,
             missing_data_rule="Pairwise observed SZ cases; no imputation",
             notes="Duration and onset age were analyzed separately and never entered jointly with current age", stringsAsFactors=FALSE)
})
history <- do.call(rbind, history_rows)
history$p_holm <- safe_holm(history$p_raw)

secondary <- rbind(symptom, history)
write_csv_utf8(secondary, file.path(ANALYSIS_ROOT, "secondary_clinical_associations.csv"))

# Exploratory DUP associations: retained despite very small N, with explicit underpowered status.
dup_rows <- lapply(seq_along(phenotypes[1:3]), function(i) {
  p <- phenotypes[i]
  z <- spearman_boot(dat, "DUP_years", p, 20261300L+i, B=10000L)
  data.frame(analysis_id=sprintf("E_DUP_%02d",i), analysis_tier="exploratory", test_family="exploratory_DUP_3",
             clinical_variable="DUP_years", phenotype=p, phenotype_label=phenotype_labels[[p]],
             effective_n=z$n, subgroup_n_1=NA_integer_, subgroup_n_0=NA_integer_, model_type="Spearman rank correlation",
             effect_type="Spearman_rho", effect=z$rho, ci_low=z$ci_low, ci_high=z$ci_high,
             ci_method="10,000-resample subject-level percentile bootstrap", p_raw=z$p, p_holm=NA_real_,
             correction_family_size=3L, correction_method="Holm", covariates="none",
             bootstrap_valid_replicates=z$boot_valid, missing_data_rule="Pairwise observed SZ cases; no imputation",
             notes="Exploratory and severely underpowered; no confirmatory inference", stringsAsFactors=FALSE)
})
dup <- do.call(rbind, dup_rows)
dup$p_holm <- safe_holm(dup$p_raw)

# Episode-number == 1 is only an operational candidate group, not a strict FES diagnosis.
first_rows <- lapply(seq_along(phenotypes[1:3]), function(i) {
  p <- phenotypes[i]
  z <- rank_biserial_boot(dat, "first_episode_candidate", p, 20261400L+i, B=10000L)
  data.frame(analysis_id=sprintf("E_FIRSTCAND_%02d",i), analysis_tier="exploratory", test_family="exploratory_first_episode_candidate_3",
             clinical_variable="first_episode_candidate", phenotype=p, phenotype_label=phenotype_labels[[p]],
             effective_n=z$n, subgroup_n_1=z$n1, subgroup_n_0=z$n0, model_type="Wilcoxon rank-sum",
             effect_type="rank_biserial_candidate_minus_other", effect=z$effect, ci_low=z$ci_low, ci_high=z$ci_high,
             ci_method="10,000-resample stratified percentile bootstrap", p_raw=z$p, p_holm=NA_real_,
             correction_family_size=3L, correction_method="Holm", covariates="none",
             bootstrap_valid_replicates=z$boot_valid, missing_data_rule="Observed episode-number records only; no imputation",
             notes="Operational first-episode candidate only; source header may denote relapse/episode count and treatment-naive status is unverified", stringsAsFactors=FALSE)
})
first <- do.call(rbind, first_rows)
first$p_holm <- safe_holm(first$p_raw)

med_blocked <- do.call(rbind, lapply(seq_along(phenotypes[1:3]), function(i) data.frame(
  analysis_id=sprintf("E_CPZ_NOT_RUN_%02d",i), analysis_tier="exploratory_not_run", test_family="exploratory_medication",
  clinical_variable="CPZ_equivalent_mg_day", phenotype=phenotypes[i], phenotype_label=phenotype_labels[[phenotypes[i]]],
  effective_n=0L, subgroup_n_1=NA_integer_, subgroup_n_0=NA_integer_, model_type="not_run", effect_type="not_estimable",
  effect=NA_real_, ci_low=NA_real_, ci_high=NA_real_, ci_method="not_applicable", p_raw=NA_real_, p_holm=NA_real_,
  correction_family_size=0L, correction_method="not_applicable", covariates="none", bootstrap_valid_replicates=0L,
  missing_data_rule="No CPZ values were created or imputed",
  notes="Blocked: current medication text does not support a uniform, source-documented, manually verified CPZ conversion", stringsAsFactors=FALSE
)))

exploratory <- rbind(dup, first, med_blocked)
write_csv_utf8(exploratory, file.path(ANALYSIS_ROOT, "exploratory_clinical_associations.csv"))

# Collinearity/definition audit for age, onset and duration (descriptive, not a joint model).
hist_vars <- dat[complete.cases(dat[c("age_scan","age_at_onset_years","illness_duration_years")]), c("age_scan","age_at_onset_years","illness_duration_years")]
history_audit <- data.frame(
  valid_n=nrow(hist_vars),
  spearman_current_age_vs_onset=if(nrow(hist_vars)>2) cor(hist_vars$age_scan,hist_vars$age_at_onset_years,method="spearman") else NA,
  spearman_current_age_vs_duration=if(nrow(hist_vars)>2) cor(hist_vars$age_scan,hist_vars$illness_duration_years,method="spearman") else NA,
  spearman_onset_vs_duration=if(nrow(hist_vars)>2) cor(hist_vars$age_at_onset_years,hist_vars$illness_duration_years,method="spearman") else NA,
  mean_identity_residual_age_minus_onset_minus_duration=if(nrow(hist_vars)) mean(hist_vars$age_scan-hist_vars$age_at_onset_years-hist_vars$illness_duration_years) else NA,
  sd_identity_residual=if(nrow(hist_vars)>1) sd(hist_vars$age_scan-hist_vars$age_at_onset_years-hist_vars$illness_duration_years) else NA,
  modeling_decision="Current age, onset age, and duration were not entered together because of conceptual and near-mathematical dependence.",
  stringsAsFactors=FALSE
)
write_tsv_utf8(history_audit, file.path(INTERNAL_DIR, "illness_history_collinearity_audit.tsv"))

append_provenance(data.frame(
  output_or_result = c("secondary_clinical_associations.csv", "exploratory_clinical_associations.csv", "illness_history_collinearity_audit.tsv"),
  analysis_stage = "C_secondary_and_exploratory",
  input_files = "clinical_imaging_analysis_dataset.tsv",
  script = "scripts/05_secondary_clinical_associations.R",
  variable_source = c("PANSS P/G/individual total, illness duration, onset age, five fixed phenotypes",
                      "DUP, operational episode count, medication audit, three primary phenotypes", "Current age, onset age, illness duration"),
  sample_rule = c("SZ only; pairwise observed", "SZ only; observed data; CPZ not fabricated", "SZ complete cases for the three history variables"),
  method = c("Spearman with family-specific Holm correction", "Exploratory Spearman/Wilcoxon; explicit blocked CPZ rows", "Descriptive Spearman and identity residual"),
  version = "clinical_extension_v1_20261029", stringsAsFactors=FALSE
))

session_record("05_secondary_clinical_associations")
message("Secondary and explicitly exploratory analyses completed without expanding to all ROIs or bins.")
