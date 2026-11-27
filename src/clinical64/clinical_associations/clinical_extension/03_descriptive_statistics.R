source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]), winslash="/")), "_common.R"))

clinical35 <- read_tsv(file.path(INTERNAL_DIR, "clinical35_audited.tsv"))
dat64 <- read_tsv(file.path(INTERNAL_DIR, "clinical_imaging_analysis_dataset.tsv"))
motion70 <- read_tsv(file.path(INTERNAL_DIR, "task_motion70_deidentified.tsv"))

numeric_vars <- c("age", "PANSS_positive", "PANSS_negative", "PANSS_general", "PANSS_total_recomputed",
                  "illness_duration_years", "age_at_onset_years", "DUP_years", "episode_number", "first_episode_candidate")
for (v in intersect(numeric_vars, names(clinical35))) clinical35[[v]] <- as_num(clinical35[[v]])
for (v in intersect(c(numeric_vars, "age_scan", "mean_FD_rest_posttrim"), names(dat64))) dat64[[v]] <- as_num(dat64[[v]])
sz32 <- dat64[dat64$group == "SZ", ]
hc32 <- dat64[dat64$group == "HC", ]
if (nrow(sz32) != 32L || nrow(hc32) != 32L) stop("Expected 32 SZ and 32 HC in the frozen imaging cohort.")

expected <- readRDS(Sys.getenv("STRIATAL_PANSS_REFERENCE_RDS", unset = NA_character_))
for (v in names(expected)) {
  z <- clinical35[[v]][is.finite(clinical35[[v]])]
  e <- expected[[v]]
  if (length(z) != e[["n"]] || abs(mean(z) - e[["mean"]]) > 1e-5 || abs(sd(z) - e[["sd"]]) > 1e-5) {
    stop("Individual-level verification failed for ", v, "; aggregate values do not match the documented source summary.")
  }
}

clinical_variables <- c("PANSS_positive", "PANSS_negative", "PANSS_general", "PANSS_total_recomputed",
                        "illness_duration_years", "age_at_onset_years", "DUP_years", "episode_number",
                        "first_episode_candidate", "medication_text_raw")
missing_rows <- list()
for (cohort_name in c("clinical_SZ35", "matched_imaging_SZ32")) {
  d <- if (cohort_name == "clinical_SZ35") clinical35 else sz32
  for (v in clinical_variables) {
    source_v <- if (v == "medication_text_raw" && !v %in% names(d)) "medication_text" else v
    x <- d[[source_v]]
    valid <- if (is.numeric(x)) is.finite(x) else nonmissing(x)
    missing_rows[[length(missing_rows) + 1L]] <- data.frame(
      cohort = cohort_name, variable = v, total_n = length(x), valid_n = sum(valid),
      missing_n = sum(!valid), missing_proportion = mean(!valid),
      handling = "Observed-data complete cases only; no imputation", stringsAsFactors = FALSE
    )
  }
}
missingness <- do.call(rbind, missing_rows)
write_csv_utf8(missingness, file.path(ANALYSIS_ROOT, "clinical_missingness.csv"))

desc_rows <- list()
desc_vars <- c("PANSS_positive", "PANSS_negative", "PANSS_general", "PANSS_total_recomputed",
               "illness_duration_years", "age_at_onset_years", "DUP_years", "episode_number")
units <- c(PANSS_positive="points", PANSS_negative="points", PANSS_general="points", PANSS_total_recomputed="points",
           illness_duration_years="years", age_at_onset_years="years", DUP_years="years", episode_number="count")
for (cohort_name in c("clinical_SZ35", "matched_imaging_SZ32")) {
  d <- if (cohort_name == "clinical_SZ35") clinical35 else sz32
  for (v in desc_vars) {
    note <- if (v == "PANSS_total_recomputed") "Individual P+N+G; no original total column was located" else "Observed values only"
    desc_rows[[length(desc_rows) + 1L]] <- summary_numeric(d[[v]], cohort_name, v, units[[v]], note)
  }
}
descriptives <- do.call(rbind, desc_rows)
write_csv_utf8(descriptives, file.path(ANALYSIS_ROOT, "clinical_descriptive_statistics.csv"))

motion70$subject <- normalize_subject(motion70$subject)
motion70$group <- ifelse(motion70$group_code == "3", "SZ", ifelse(motion70$group_code == "4", "HC", NA_character_))
motion70$mean_FD_Power <- as_num(motion70$mean_FD_Power)
task_motion_desc <- do.call(rbind, lapply(split(motion70, list(motion70$group, motion70$motion_block), drop = TRUE), function(d) {
  data.frame(cohort = paste0(d$group[1], "35"), motion_source = "reward_task_block",
             block = d$motion_block[1], valid_n = sum(is.finite(d$mean_FD_Power)),
             mean_FD = mean(d$mean_FD_Power, na.rm = TRUE), sd_FD = sd(d$mean_FD_Power, na.rm = TRUE),
             note = "Task-block FD audit only; not the resting-state MeanFD covariate used by the gradient analyses.", stringsAsFactors = FALSE)
}))
write_tsv_utf8(task_motion_desc, file.path(INTERNAL_DIR, "task_motion70_descriptive.tsv"))

demo <- rbind(
  data.frame(cohort="SZ32", n=nrow(sz32), age_mean=mean(sz32$age_scan), age_sd=sd(sz32$age_scan),
             female_n=sum(sz32$sex_scan == "Female"), female_prop=mean(sz32$sex_scan == "Female"),
             rest_meanFD_mean=mean(sz32$mean_FD_rest_posttrim), rest_meanFD_sd=sd(sz32$mean_FD_rest_posttrim)),
  data.frame(cohort="HC32", n=nrow(hc32), age_mean=mean(hc32$age_scan), age_sd=sd(hc32$age_scan),
             female_n=sum(hc32$sex_scan == "Female"), female_prop=mean(hc32$sex_scan == "Female"),
             rest_meanFD_mean=mean(hc32$mean_FD_rest_posttrim), rest_meanFD_sd=sd(hc32$mean_FD_rest_posttrim))
)
write_tsv_utf8(demo, file.path(INTERNAL_DIR, "actual_imaging64_demographics.tsv"))

append_provenance(data.frame(
  output_or_result = c("clinical_missingness.csv", "clinical_descriptive_statistics.csv", "task_motion70_descriptive.tsv", "actual_imaging64_demographics.tsv"),
  analysis_stage = "A_data_audit",
  input_files = c("clinical35_audited.tsv; clinical_imaging_analysis_dataset.tsv", "same", "task_motion70_deidentified.tsv", "clinical_imaging_analysis_dataset.tsv"),
  script = "scripts/03_descriptive_statistics.R",
  variable_source = c("Individual clinical records", "Individual clinical records", "35+35 consolidated task-motion sheet", "Frozen current64 covariates"),
  sample_rule = c("Reported separately for full SZ35 and matched SZ32", "Observed values only", "All available block rows", "Frozen 32 SZ + 32 HC"),
  method = c("Variable-wise missingness", "Mean/SD/median/IQR/range", "Block-specific descriptive statistics", "Group descriptive statistics"),
  version = "clinical_extension_v1_20261029", stringsAsFactors = FALSE
))

session_record("03_descriptive_statistics")
message("Clinical missingness and descriptive statistics verified from individual-level data.")
