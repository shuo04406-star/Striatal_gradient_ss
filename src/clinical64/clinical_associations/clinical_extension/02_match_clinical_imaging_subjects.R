source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]), winslash="/")), "_common.R"))

covar <- read_tsv(file.path(PROJECT_ROOT, "CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "covariates", "current64_covariates.tsv"))
bin10 <- read_tsv(file.path(PROJECT_ROOT, "journal_figure_scatter_revision_20261028", "source_data", "B05_Bin10_stage_subjects_used.tsv"))
rawfc <- read_tsv(file.path(PROJECT_ROOT, "THREE_REGION_RAW_FC_VALIDATION_20261028", "tables", "three_region_subject_bin_FC.tsv"))
clinical64 <- read_tsv(file.path(CLINICAL_AUDIT_ROOT, "03_clinical_master", "final64_clinical_for_server_v3.tsv"))
clinical35_audited <- read_tsv(file.path(INTERNAL_DIR, "clinical35_audited.tsv"))

covar$subject <- normalize_subject(covar$Subject)
covar$age_scan <- as_num(covar$Age)
covar$sex_scan <- covar$Sex
covar$mean_FD_rest_posttrim <- as_num(covar$MeanFD)
covar$group <- covar$Group
if (nrow(covar) != 64L || anyDuplicated(covar$subject)) stop("Covariate source is not the frozen 64-subject cohort.")

bin10$subject <- normalize_subject(bin10$subject)
bin10$profile_z <- as_num(bin10$profile_z)
if (nrow(bin10) != 128L || anyDuplicated(bin10[c("subject", "stage")])) stop("Expected one Bin-10 profile row per subject and stage.")
ant <- bin10[bin10$stage == "Anticipation", c("subject", "profile_z", "profile_r", "n_roi", "reference")]
names(ant)[-1] <- paste0("bin10_anticipation_", names(ant)[-1])
out <- bin10[bin10$stage == "Outcome", c("subject", "profile_z", "profile_r", "n_roi", "reference")]
names(out)[-1] <- paste0("bin10_outcome_", names(out)[-1])

rawfc$subject <- normalize_subject(rawfc$Subject)
rawfc$Bin <- as_num(rawfc$Bin)
rawfc$FC <- as_num(rawfc$FC)
key <- interaction(rawfc$subject, rawfc$Region, rawfc$Stage, drop = TRUE, lex.order = TRUE)
slope_rows <- lapply(split(rawfc, key), function(d) {
  d <- d[is.finite(d$Bin) & is.finite(d$FC), ]
  if (nrow(d) != 10L || length(unique(d$Bin)) != 10L) stop("A raw-FC subject/region profile does not have all 10 fixed bins.")
  fit <- lm(FC ~ I(Bin - 5.5), data = d)
  data.frame(subject = d$subject[1], region = d$Region[1], stage = d$Stage[1],
             rawFC_slope = unname(coef(fit)[2]), slope_n_bins = nrow(d), stringsAsFactors = FALSE)
})
slopes <- do.call(rbind, slope_rows)
if (nrow(slopes) != 192L) stop("Expected 64 x 3 individual raw-FC slopes.")

extract_slope <- function(region, stage, name) {
  x <- slopes[slopes$region == region & slopes$stage == stage, c("subject", "rawFC_slope")]
  if (nrow(x) != 64L || anyDuplicated(x$subject)) stop("Unexpected slope coverage for ", region, " / ", stage)
  names(x)[2] <- name
  x
}
sma <- extract_slope("SMA", "Anticipation", "SMA_anticipation_rawFC_slope")
ifg <- extract_slope("IFG", "Outcome", "IFG_outcome_rawFC_slope")
cing <- extract_slope("Cingulate", "Outcome", "Cingulate_outcome_rawFC_slope")

clinical64$subject <- normalize_subject(clinical64$subject)
numeric_clinical <- c("PANSS_positive", "PANSS_negative", "PANSS_general", "PANSS_total", "illness_duration",
                      "age_at_onset", "DUP", "episode_number", "first_episode_candidate")
for (v in numeric_clinical) clinical64[[v]] <- as_num(clinical64[[v]])
names(clinical64)[names(clinical64) == "PANSS_total"] <- "PANSS_total_recomputed"
names(clinical64)[names(clinical64) == "illness_duration"] <- "illness_duration_years"
names(clinical64)[names(clinical64) == "age_at_onset"] <- "age_at_onset_years"
names(clinical64)[names(clinical64) == "DUP"] <- "DUP_years"

dat <- covar[c("subject", "group", "age_scan", "sex_scan", "mean_FD_rest_posttrim")]
for (x in list(ant, out, sma, ifg, cing)) dat <- merge(dat, x, by = "subject", all.x = TRUE, sort = FALSE)
clinical_keep <- c("subject", "PANSS_positive", "PANSS_negative", "PANSS_general", "PANSS_total_recomputed",
                   "clinical_conflict_flag", "clinical_conflict_note")
dat <- merge(dat, clinical64[clinical_keep], by = "subject", all.x = TRUE, sort = FALSE)
clinical35_audited$subject <- normalize_subject(clinical35_audited$subject)
history_keep <- c("subject", "illness_duration_years", "age_at_onset_years", "DUP_years", "episode_number",
                  "first_episode_candidate", "medication_text_raw", "DUP_raw", "episode_number_raw", "onset_treatment_raw")
for (v in c("illness_duration_years", "age_at_onset_years", "DUP_years", "episode_number", "first_episode_candidate")) {
  clinical35_audited[[v]] <- as_num(clinical35_audited[[v]])
}
names(clinical35_audited)[names(clinical35_audited) == "medication_text_raw"] <- "medication_text"
history_keep[history_keep == "medication_text_raw"] <- "medication_text"
dat <- merge(dat, clinical35_audited[history_keep], by = "subject", all.x = TRUE, sort = FALSE)
dat <- dat[match(covar$subject, dat$subject), ]
if (nrow(dat) != 64L || anyDuplicated(dat$subject)) stop("Merged imaging-clinical analysis dataset is not 64 unique subjects.")
if (any(!complete.cases(dat[c("bin10_anticipation_profile_z", "bin10_outcome_profile_z", "SMA_anticipation_rawFC_slope",
                              "IFG_outcome_rawFC_slope", "Cingulate_outcome_rawFC_slope")]))) stop("Imaging phenotype coverage is incomplete in the frozen cohort.")
write_tsv_utf8(dat, file.path(INTERNAL_DIR, "clinical_imaging_analysis_dataset.tsv"))
write_tsv_utf8(slopes, file.path(INTERNAL_DIR, "individual_rawFC_slopes.tsv"))

audit_path <- file.path(ANALYSIS_ROOT, "sample_overlap_audit.csv")
audit <- read.csv(audit_path, stringsAsFactors = FALSE, check.names = FALSE, na.strings = c(""), fileEncoding = "UTF-8")
for (v in c("bin10_anticipation_profile_z", "bin10_outcome_profile_z", "SMA_anticipation_rawFC_slope",
            "IFG_outcome_rawFC_slope", "Cingulate_outcome_rawFC_slope")) {
  audit[[paste0("has_", v)]] <- as.integer(!is.na(dat[[v]][match(normalize_subject(audit$subject), dat$subject)]))
}
write_csv_utf8(audit, audit_path)

append_provenance(data.frame(
  output_or_result = c("clinical_imaging_analysis_dataset.tsv", "individual_rawFC_slopes.tsv", "sample_overlap_audit.csv phenotype coverage"),
  analysis_stage = "A_data_audit",
  input_files = c("current64_covariates.tsv; final64_clinical_for_server_v3.tsv; B05_Bin10_stage_subjects_used.tsv; three_region_subject_bin_FC.tsv",
                  "three_region_subject_bin_FC.tsv", "All matched clinical and phenotype sources"),
  script = "scripts/02_match_clinical_imaging_subjects.R",
  variable_source = c("Exact subject ID merge", "Per-subject ten-bin FC profiles", "Exact subject ID merge"),
  sample_rule = c("Frozen current64 cohort", "All 64 subjects with all ten bins", "All 70 clinical-cohort rows"),
  method = c("One-to-one ID merge", "OLS FC ~ (Bin - 5.5) separately by subject/region/stage", "Availability flags"),
  version = "clinical_extension_v1_20261029", stringsAsFactors = FALSE
))

session_record("02_match_clinical_imaging_subjects")
message("Imaging phenotypes matched to the clinical cohort by exact participant ID.")
