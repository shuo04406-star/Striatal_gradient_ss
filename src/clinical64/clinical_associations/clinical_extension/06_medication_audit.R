source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]), winslash="/")), "_common.R"))

clinical35 <- read_tsv(file.path(INTERNAL_DIR, "clinical35_audited.tsv"))
dat64 <- read_tsv(file.path(INTERNAL_DIR, "clinical_imaging_analysis_dataset.tsv"))
demo <- read_tsv(file.path(INTERNAL_DIR, "actual_imaging64_demographics.tsv"))
miss <- read.csv(file.path(ANALYSIS_ROOT, "clinical_missingness.csv"), stringsAsFactors = FALSE, fileEncoding = "UTF-8")

standardize_drugs <- function(x) {
  if (is.na(x) || !nzchar(trimws(x))) return(NA_character_)
  rules <- c("paliperidone"="帕立哌酮|善思达", "risperidone"="利培酮", "olanzapine"="奥氮平",
             "amisulpride"="氨磺必利", "aripiprazole"="阿立哌唑", "quetiapine"="喹硫平",
             "trihexyphenidyl_adjunct"="苯海索", "valproate_adjunct"="丙戊酸")
  hits <- names(rules)[vapply(rules, function(p) grepl(p, x, perl = TRUE), logical(1))]
  if (!length(hits)) "unrecognized" else paste(hits, collapse = ";")
}

audit_one <- function(text) {
  missing <- is.na(text) || !nzchar(trimws(text))
  if (missing) return(c(route="unknown", dose_recognizable="no", conversion_status="not_calculable",
                        reason="Medication information missing; this is not interpreted as unmedicated."))
  drugs <- standardize_drugs(text)
  route <- if (grepl("微球|/月|/2周|每月|针", text)) "long_acting_injectable_or_depot" else if (grepl("口服液|ml", text, ignore.case=TRUE)) "oral_liquid" else "oral_or_unspecified"
  has_unit <- grepl("[0-9]+(?:\\.[0-9]+)?\\s*(?:mg|g|ml)", text, ignore.case=TRUE, perl=TRUE)
  dose_rec <- if (!has_unit) "no" else if (grepl("[0-9]+\\s*[-—]\\s*[0-9]+", text, perl=TRUE)) "partial_range" else "yes_textual"
  n_antipsych <- sum(strsplit(drugs, ";", fixed=TRUE)[[1]] %in% c("paliperidone","risperidone","olanzapine","amisulpride","aripiprazole","quetiapine"))
  reason <- if (route == "oral_liquid") {
    "Liquid volume is recorded without a verified concentration; mg/day cannot be established."
  } else if (route == "long_acting_injectable_or_depot") {
    "Long-acting/depot formulation requires formulation-specific interval conversion and manual verification."
  } else if (dose_rec == "partial_range") {
    "Dose is a range rather than a single current dose."
  } else if (!has_unit) {
    "Numeric text lacks an explicit mass unit and must not be assumed to be grams or milligrams."
  } else if (n_antipsych > 1) {
    "Antipsychotic polypharmacy requires component-level conversion and summation after manual verification."
  } else if (drugs == "unrecognized") {
    "Drug name could not be standardized reliably."
  } else {
    "Potentially convertible, but no project-approved published CPZ conversion table or independent manual review is documented."
  }
  status <- if (grepl("Potentially convertible", reason, fixed=TRUE)) "pending_conversion_source_and_manual_review" else "not_calculable_from_current_record"
  c(route=route, dose_recognizable=dose_rec, conversion_status=status, reason=reason)
}

rows <- lapply(seq_len(nrow(clinical35)), function(i) {
  text <- clinical35$medication_text_raw[i]
  a <- audit_one(text)
  data.frame(
    participant_id = clinical35$subject[i],
    in_frozen_imaging64 = as.integer(clinical35$subject[i] %in% dat64$subject),
    medication_name_original = text,
    original_dose_text = text,
    standardized_medication_name = standardize_drugs(text),
    route_or_formulation = unname(a[["route"]]),
    dose_identification_status = unname(a[["dose_recognizable"]]),
    can_reliably_calculate_CPZ_now = "no",
    CPZ_equivalent_mg_day = NA_real_,
    CPZ_conversion_status = unname(a[["conversion_status"]]),
    reason_not_converted = unname(a[["reason"]]),
    manual_review_status = "required_before_any_CPZ_analysis",
    stringsAsFactors = FALSE
  )
})
med <- do.call(rbind, rows)
if (nrow(med) != 35L || sum(nonmissing(med$medication_name_original)) != 33L) stop("Medication audit must preserve 33 observed and 2 missing records.")
write_csv_utf8(med, file.path(ANALYSIS_ROOT, "medication_audit.csv"))

full35 <- read_tsv(file.path(INTERNAL_DIR, "clinical35_audited.tsv"))
for (v in c("PANSS_positive","PANSS_negative","PANSS_general","PANSS_total_recomputed","illness_duration_years","age_at_onset_years","DUP_years","episode_number")) full35[[v]] <- as_num(full35[[v]])
sz32 <- dat64[dat64$group == "SZ", ]
for (v in c("PANSS_positive","PANSS_negative","PANSS_general","PANSS_total_recomputed")) sz32[[v]] <- as_num(sz32[[v]])

fmt <- function(x, d=2) formatC(x, digits=d, format="f")
audit_lines <- c(
  "# Stage A data audit",
  "",
  paste0("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M %Z")),
  "",
  "## Cohort linkage",
  "",
  "- The consolidated clinical/demographic workbook contains 35 SZ and 35 controls with unique MRI research IDs.",
  "- Exact participant-ID matching identified 32 SZ and 32 controls in the frozen canonical imaging cohort.",
  "- The three excluded SZ IDs and three excluded control IDs are absent from the frozen 64-person list. Current provenance does not document a subject-specific upstream reason, so no reason was inferred.",
  "- No row-order matching or fuzzy ID matching was used.",
  "",
  "## Individual clinical data verification",
  "",
  paste0("- PANSS positive: N=", sum(is.finite(full35$PANSS_positive)), ", mean=", fmt(mean(full35$PANSS_positive, na.rm=TRUE)), ", SD=", fmt(sd(full35$PANSS_positive, na.rm=TRUE)), "."),
  paste0("- PANSS negative: N=", sum(is.finite(full35$PANSS_negative)), ", mean=", fmt(mean(full35$PANSS_negative, na.rm=TRUE)), ", SD=", fmt(sd(full35$PANSS_negative, na.rm=TRUE)), "."),
  paste0("- PANSS general: N=", sum(is.finite(full35$PANSS_general)), ", mean=", fmt(mean(full35$PANSS_general, na.rm=TRUE)), ", SD=", fmt(sd(full35$PANSS_general, na.rm=TRUE)), "."),
  paste0("- PANSS total: N=", sum(is.finite(full35$PANSS_total_recomputed)), ", mean=", fmt(mean(full35$PANSS_total_recomputed, na.rm=TRUE)), ", SD=", fmt(sd(full35$PANSS_total_recomputed, na.rm=TRUE)), ". It is a real individual-level P+N+G recomputation; no original total-score column was located."),
  paste0("- In the matched SZ32 imaging cohort, PANSS-negative is available for N=", sum(is.finite(sz32$PANSS_negative)), "."),
  paste0("- Illness duration is observed for ", sum(is.finite(full35$illness_duration_years)), "/35, age at onset for ", sum(is.finite(full35$age_at_onset_years)), "/35, DUP for ", sum(is.finite(full35$DUP_years)), "/35, and episode number for ", sum(is.finite(full35$episode_number)), "/35."),
  paste0("- The operational episode-number variable equals 1 for ", sum(full35$episode_number == 1, na.rm=TRUE), " candidates. Its header is 'Times of relapses / 第几次发病'; it does not establish strict first-episode schizophrenia or treatment-naive status."),
  "",
  "## Medication audit",
  "",
  "- Medication text is present for 33/35 patients; the two missing records are not treated as unmedicated.",
  "- Drug names and formulation clues were standardized conservatively while preserving the original text.",
  "- No CPZ-equivalent dose was calculated: liquid concentrations, depot formulations, ranges, unit-free doses, and polypharmacy occur, and no project-approved conversion table plus independent manual review was available.",
  "",
  "## Imaging phenotype provenance",
  "",
  "- Bin 10 anticipation and outcome similarity comes from B05_Bin10_stage_subjects_used.tsv and is analyzed on the Fisher-z scale.",
  "- SMA anticipation, IFG outcome, and cingulate outcome individual slopes were computed from the validated ten-bin raw-FC table as the subject-specific coefficient of FC ~ (Bin - 5.5).",
  "- All five phenotype fields are complete for the frozen 64-person cohort.",
  "",
  "## Stage A decision",
  "",
  "The exact-ID linkage, individual PANSS records, and three prespecified imaging phenotypes passed the audit. The primary clinical association analyses may proceed using observed SZ cases only. CPZ-adjusted analyses and strict first-episode analyses remain blocked pending external clinical verification."
)
writeLines(audit_lines, file.path(ANALYSIS_ROOT, "STAGE_A_DATA_AUDIT.md"), useBytes = TRUE)

append_provenance(data.frame(
  output_or_result = c("medication_audit.csv", "STAGE_A_DATA_AUDIT.md"),
  analysis_stage = "A_data_audit",
  input_files = c("raw_clinical35_deidentified.tsv; clinical_imaging_analysis_dataset.tsv", "All Stage A audit inputs"),
  script = "scripts/06_medication_audit.R",
  variable_source = c("Original medication text", "Individual clinical, cohort, and phenotype sources"),
  sample_rule = c("All 35 exactly mapped SZ clinical records", "Full clinical70 and frozen imaging64"),
  method = c("Conservative text standardization; no CPZ calculation", "Programmatic audit summary"),
  version = "clinical_extension_v1_20261029", stringsAsFactors = FALSE
))

session_record("06_medication_audit")
message("Medication and Stage A audit complete; CPZ analysis is not authorized by current data quality.")
