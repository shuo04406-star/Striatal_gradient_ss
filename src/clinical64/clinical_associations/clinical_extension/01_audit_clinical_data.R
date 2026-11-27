source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]), winslash="/")), "_common.R"))

c70 <- read_tsv(file.path(INTERNAL_DIR, "clinical70_deidentified.tsv"))
raw35 <- read_tsv(file.path(INTERNAL_DIR, "raw_clinical35_deidentified.tsv"))
frozen <- read_tsv(file.path(PROJECT_ROOT, "CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "subject_table", "current64_subject_table.tsv"))
canonical64 <- read_tsv(file.path(CLINICAL_AUDIT_ROOT, "03_clinical_master", "final64_clinical_for_server_v3.tsv"))

c70$subject <- normalize_subject(c70$subject)
c70$group <- ifelse(c70$group_code == "3", "SZ", ifelse(c70$group_code == "4", "HC", NA_character_))
c70$sex <- ifelse(c70$sex_raw == "1", "Male", ifelse(c70$sex_raw == "2", "Female", NA_character_))
c70$age <- as_num(c70$age)
c70$PANSS_positive <- as_num(c70$PANSS_positive)
c70$PANSS_negative <- as_num(c70$PANSS_negative)
c70$PANSS_general <- as_num(c70$PANSS_general)
c70$PANSS_total_recomputed <- ifelse(
  complete.cases(c70[c("PANSS_positive", "PANSS_negative", "PANSS_general")]),
  c70$PANSS_positive + c70$PANSS_negative + c70$PANSS_general, NA_real_
)
if (nrow(c70) != 70L || anyDuplicated(c70$subject)) stop("The consolidated cohort must contain 70 unique research IDs.")

raw35$subject <- normalize_subject(raw35$subject)
if (nrow(raw35) != 35L || anyDuplicated(raw35$subject)) stop("The raw clinical mapping must contain 35 unique patients.")
raw35$illness_duration_years <- parse_duration_years(raw35$illness_duration_years_raw)
raw35$age_at_onset_years <- as_num(raw35$age_at_onset_years_raw)
raw35$DUP_years <- parse_dup_years(raw35$DUP_raw)
raw35$episode_number <- parse_first_number(raw35$episode_number_raw)
raw35$first_episode_candidate <- ifelse(is.na(raw35$episode_number), NA_integer_, as.integer(raw35$episode_number == 1))

clinical35 <- merge(c70[c("subject", "group", "sex", "age", "PANSS_positive", "PANSS_negative",
                          "PANSS_general", "PANSS_total_recomputed")], raw35, by = "subject", all.y = TRUE, sort = FALSE)
if (any(clinical35$group != "SZ")) stop("Mapped raw clinical records are not restricted to the 35-patient cohort.")
write_tsv_utf8(clinical35, file.path(INTERNAL_DIR, "clinical35_audited.tsv"))

frozen$subject <- normalize_subject(frozen$Subject)
frozen$imaging_group <- frozen$Group
canonical64$subject <- normalize_subject(canonical64$subject)
if (nrow(frozen) != 64L || anyDuplicated(frozen$subject)) stop("Frozen imaging cohort must contain 64 unique subjects.")

overlap <- c70[c("subject", "group", "sex", "age", "PANSS_positive", "PANSS_negative", "PANSS_general", "PANSS_total_recomputed")]
overlap$in_clinical70 <- 1L
overlap$in_imaging64 <- as.integer(overlap$subject %in% frozen$subject)
overlap$imaging_group <- frozen$imaging_group[match(overlap$subject, frozen$subject)]
overlap$has_raw_clinical_record <- as.integer(overlap$subject %in% raw35$subject)
for (v in c("illness_duration_years", "age_at_onset_years", "DUP_years", "episode_number", "medication_text_raw")) {
  overlap[[paste0("has_", v)]] <- ifelse(overlap$group == "SZ",
    as.integer(nonmissing(clinical35[[v]][match(overlap$subject, clinical35$subject)])), NA_integer_)
}
overlap$has_PANSS_positive <- ifelse(overlap$group == "SZ", as.integer(!is.na(overlap$PANSS_positive)), NA_integer_)
overlap$has_PANSS_negative <- ifelse(overlap$group == "SZ", as.integer(!is.na(overlap$PANSS_negative)), NA_integer_)
overlap$has_PANSS_general <- ifelse(overlap$group == "SZ", as.integer(!is.na(overlap$PANSS_general)), NA_integer_)
overlap$has_PANSS_total_recomputed <- ifelse(overlap$group == "SZ", as.integer(!is.na(overlap$PANSS_total_recomputed)), NA_integer_)
overlap$sample_status <- ifelse(overlap$in_imaging64 == 1, "included_in_frozen_64_imaging_cohort", "not_in_frozen_64_imaging_cohort")
overlap$exclusion_or_missing_reason <- ifelse(overlap$in_imaging64 == 1, "",
  "Not present in the frozen canonical 64-subject list; a subject-specific upstream exclusion reason was not recoverable from current provenance and is not inferred.")

keep <- c("subject", "group", "sex", "age", "in_clinical70", "in_imaging64", "imaging_group",
          "has_raw_clinical_record", "has_PANSS_positive", "has_PANSS_negative", "has_PANSS_general",
          "has_PANSS_total_recomputed", "has_illness_duration_years", "has_age_at_onset_years", "has_DUP_years",
          "has_episode_number", "has_medication_text_raw", "sample_status", "exclusion_or_missing_reason")
write_csv_utf8(overlap[keep], file.path(ANALYSIS_ROOT, "sample_overlap_audit.csv"))

dictionary <- data.frame(
  variable = c("subject", "group", "age", "sex", "mean_FD_rest_posttrim", "PANSS_positive", "PANSS_negative",
               "PANSS_general", "PANSS_total_recomputed", "illness_duration_years", "age_at_onset_years", "DUP_years",
               "DUP_raw", "episode_number", "first_episode_candidate", "medication_text_raw",
               "bin10_anticipation_similarity_z", "bin10_outcome_similarity_z", "SMA_anticipation_rawFC_slope",
               "IFG_outcome_rawFC_slope", "Cingulate_outcome_rawFC_slope"),
  meaning = c("De-identified research participant ID", "Cohort label", "Age at scan", "Recorded sex",
              "Mean framewise displacement from resting-state post-trim motion file",
              "PANSS positive subscale", "PANSS negative subscale", "PANSS general psychopathology subscale",
              "Individual total recomputed as P+N+G only when all three are present", "Recorded illness duration",
              "Recorded age at onset", "Duration of untreated psychosis converted to years for description",
              "Original DUP text", "Raw operational count labelled 'Times of relapses / 第几次发病'",
              "Operational candidate flag: episode_number == 1; not strict first-episode schizophrenia",
              "Medication text at the clinical record", "Fisher-z HC-reference profile similarity at Bin 10, anticipation",
              "Fisher-z HC-reference profile similarity at Bin 10, outcome",
              "Within-subject OLS slope across fixed G1 bins for SMA raw FC, anticipation",
              "Within-subject OLS slope across fixed G1 bins for IFG composite raw FC, outcome",
              "Within-subject OLS slope across fixed G1 bins for cingulate raw FC, outcome"),
  unit = c("ID", "category", "years", "category", "mm", "points", "points", "points", "points", "years", "years",
           "years", "original text", "count", "0/1", "original text", "Fisher z", "Fisher z", "FC per bin", "FC per bin", "FC per bin"),
  data_type = c("character", "categorical", "numeric", "categorical", "numeric", rep("numeric", 7), "character", "numeric", "binary", "character", rep("numeric", 5)),
  source = c(
    "fMRI_35vs35_整理版.xlsx / 01_主信息_70人; exact ID audit", "same", "same", "same",
    "CURRENT64_CANONICAL_RERUN/01_FROZEN_INPUTS/covariates/current64_covariates.tsv",
    rep("fMRI_35vs35_整理版.xlsx / 01_主信息_70人; validated against fMRI_SCZ.sav", 4),
    rep("humor_scales原始录入.xlsm / 临床", 7),
    rep("journal_figure_scatter_revision_20261028/source_data/B05_Bin10_stage_subjects_used.tsv", 2),
    rep("THREE_REGION_RAW_FC_VALIDATION_20261028/tables/three_region_subject_bin_FC.tsv", 3)
  ),
  cleaning_or_caveat = c(
    "Exact normalized ID matching only; no row-order or fuzzy matching", "3=SZ, 4=HC in the consolidated source", "No imputation",
    "1=Male, 2=Female in consolidated source", "Do not substitute task motion block summaries",
    "No imputation", "Primary clinical variable; no imputation", "No imputation",
    "No original total field located; this is an individual-level recomputation, never inferred from summary statistics",
    "Fractions/month annotations converted transparently; raw value retained internally", "No imputation", "Original unit text converted only when explicit",
    "Retained unchanged", "Definition is ambiguous and requires clinical confirmation", "Named first-episode candidate only",
    "Missing does not mean unmedicated", "Analysis uses profile_z; SZ reference is HC32", "Analysis uses profile_z; SZ reference is HC32",
    "Ten fixed bins; slope from FC ~ (Bin-5.5)", "Ten fixed bins; slope from FC ~ (Bin-5.5)", "Ten fixed bins; slope from FC ~ (Bin-5.5)"
  ), stringsAsFactors = FALSE
)
write_csv_utf8(dictionary, file.path(ANALYSIS_ROOT, "clinical_data_dictionary.csv"))

append_provenance(data.frame(
  output_or_result = c("sample_overlap_audit.csv", "clinical_data_dictionary.csv", "clinical35_audited.tsv"),
  analysis_stage = "A_data_audit",
  input_files = c("clinical70_deidentified.tsv; raw_clinical35_deidentified.tsv; current64_subject_table.tsv",
                  "Source workbook headers and verified imaging phenotype sources",
                  "clinical70_deidentified.tsv; raw_clinical35_deidentified.tsv"),
  script = "scripts/01_audit_clinical_data.R", variable_source = "Exact research-ID linkage",
  sample_rule = c("All 70 consolidated participants", "Variable-level metadata", "All 35 exactly mapped SZ clinical records"),
  method = c("Exact ID overlap audit", "Data dictionary construction", "Canonical PANSS plus raw clinical field merge"),
  version = "clinical_extension_v1_20261029", stringsAsFactors = FALSE
))

session_record("01_audit_clinical_data")
message("Stage A core clinical audit created.")
