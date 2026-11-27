#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(svglite)
  library(ragg)
})

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
if (length(file_arg) != 1L) stop("Run this script with Rscript.", call. = FALSE)
script_path <- normalizePath(sub("^--file=", "", file_arg), winslash = "/", mustWork = TRUE)
scripts_dir <- dirname(script_path)
out_dir <- Sys.getenv("STRIATAL_OUTPUT_ROOT", unset = NA_character_)
project_root <- Sys.getenv("STRIATAL_PROJECT_ROOT", unset = NA_character_)

paths <- list(
  fc_long = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "04_BINWISE_FC_PROFILES", "subject_bin_roi_FC_long.tsv"),
  fc_wide = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "04_BINWISE_FC_PROFILES", "subject_bin_roi_FC_wide.tsv"),
  covariates = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "covariates", "current64_covariates.tsv"),
  roi_order = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "roi_order", "canonical_roi_order_65.tsv"),
  frozen_fingerprints = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "fc_fingerprints_2608x65"),
  locked_B08 = file.path(project_root, "figures", "derived_results", "B08_SMA_IFG_raw_FC_tests.tsv"),
  historical_6 = file.path(project_root, "IFG_6ROI_rawFC_reanalysis_20261028", "tables", "recomputed_2targets_6tests.tsv"),
  contribution = file.path(project_root, "figures", "derived_results", "B06_subfamily_signed_contributions.tsv"),
  g1_reference = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "canonical_G1", "375_reference_G1_manuscript_2608.npy"),
  fixed_bins = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "canonical_bins", "375_reference_bins_manuscript_2608.npy")
)

required_paths <- unlist(paths[c("fc_long", "fc_wide", "covariates", "roi_order", "locked_B08", "historical_6", "contribution", "g1_reference", "fixed_bins")])
if (!all(file.exists(required_paths))) {
  stop(paste("Missing required inputs:", paste(required_paths[!file.exists(required_paths)], collapse = "\n")), call. = FALSE)
}

dir.create(file.path(out_dir, "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "tables"), recursive = TRUE, showWarnings = FALSE)

assert_true <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

write_tsv_stable <- function(x, path) {
  readr::write_tsv(x, path, na = "NA")
}

fc <- readr::read_tsv(paths$fc_long, show_col_types = FALSE, progress = FALSE)
covariates <- readr::read_tsv(paths$covariates, show_col_types = FALSE, progress = FALSE)
roi_order <- readr::read_tsv(paths$roi_order, show_col_types = FALSE, progress = FALSE)
locked_B08 <- readr::read_tsv(paths$locked_B08, show_col_types = FALSE, progress = FALSE)
historical_6 <- readr::read_tsv(paths$historical_6, show_col_types = FALSE, progress = FALSE)
contribution_all <- readr::read_tsv(paths$contribution, show_col_types = FALSE, progress = FALSE)

required_fc <- c("Subject", "Group", "Bin", "ROI", "Subfamily", "RewardStage", "FC")
required_cov <- c("Subject", "Group", "Age", "Sex", "MeanFD")
assert_true(all(required_fc %in% names(fc)), "Canonical FC table lacks required columns.")
assert_true(all(required_cov %in% names(covariates)), "Covariate table lacks required columns.")
assert_true(nrow(fc) == 64L * 10L * 65L, "Canonical FC table is not 64 subjects x 10 bins x 65 ROIs.")
assert_true(nrow(covariates) == 64L, "Covariate table does not contain 64 participants.")
assert_true(identical(as.integer(sort(unique(fc$Bin))), 1:10), "Canonical FC table does not contain fixed bins 1-10.")

specs <- tribble(
  ~Region, ~Stage, ~N_ROIs, ~ROI,
  "SMA", "Anticipation", 1L, "R_SFG_7_5_A6m_SMA__BN10",
  "IFG", "Outcome", 6L, "L_IFG_6_1_A44d__BN29",
  "IFG", "Outcome", 6L, "L_IFG_6_2_IFS__BN31",
  "IFG", "Outcome", 6L, "L_IFG_6_3_A45c__BN33",
  "IFG", "Outcome", 6L, "L_IFG_6_4_A45r__BN35",
  "IFG", "Outcome", 6L, "L_IFG_6_5_A44op__BN37",
  "IFG", "Outcome", 6L, "L_IFG_6_6_A44v__BN39",
  "Cingulate", "Outcome", 1L, "R_CG_7_7_A32sg_ACC__BN188"
)

roi_mapping <- specs |>
  left_join(
    roi_order |>
      transmute(
        ROI = roi_name,
        ROI_array_column_0based = roi_index,
        ROI_array_column_1based = roi_index + 1L,
        mapped_subfamily = subfamily,
        mapped_stage = reward_stage,
        atlas_label = atlas_source
      ),
    by = "ROI"
  ) |>
  mutate(
    Brainnetome = sub("^.*__(BN[0-9]+)$", "\\1", ROI),
    Brainnetome_number = as.integer(sub("^.*__BN", "", ROI)),
    canonical_FC_long_file = file.path("CURRENT64_CANONICAL_RERUN", "04_BINWISE_FC_PROFILES", "subject_bin_roi_FC_long.tsv"),
    canonical_FC_field = "FC",
    canonical_FC_selector = paste0("ROI == ", ROI),
    frozen_Fisher_z_file_pattern = file.path("CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "fc_fingerprints_2608x65", "<Subject>_fingerprint_z_2608x65.npy")
  )

assert_true(!anyNA(roi_mapping$ROI_array_column_0based), "At least one requested ROI is absent from the canonical 65-ROI mapping.")
assert_true(all(roi_mapping$Region == roi_mapping$mapped_subfamily), "Requested region and canonical subfamily mapping disagree.")
assert_true(all(roi_mapping$Stage == roi_mapping$mapped_stage), "Requested stage and canonical reward-stage mapping disagree.")
cing_map <- roi_mapping |> filter(Region == "Cingulate")
assert_true(nrow(cing_map) == 1L, "Cingulate must map to exactly one ROI.")
assert_true(
  cing_map$ROI == "R_CG_7_7_A32sg_ACC__BN188" && cing_map$Brainnetome_number == 188L &&
    cing_map$Stage == "Outcome" && cing_map$ROI_array_column_0based == 42L && cing_map$ROI_array_column_1based == 43L,
  "Cingulate mapping does not match the canonical R_A32sg / BN188 Outcome ROI at array column 43."
)

selected_raw <- fc |>
  inner_join(specs, by = "ROI")
assert_true(nrow(selected_raw) == (1L + 6L + 1L) * 64L * 10L, "Selected raw ROI rows are incomplete.")
assert_true(!anyDuplicated(selected_raw[c("Subject", "Bin", "ROI")]), "Selected Subject x Bin x ROI keys are duplicated.")
assert_true(all(is.finite(selected_raw$FC)), "Selected raw Fisher-z FC contains non-finite values.")
assert_true(all(selected_raw$Stage == selected_raw$RewardStage), "Selected FC rows disagree with the locked reward-stage mapping.")
assert_true(all(selected_raw$Region == selected_raw$Subfamily), "Selected FC rows disagree with the locked subfamily mapping.")

completeness <- selected_raw |>
  group_by(Region, Stage, N_ROIs, ROI, Group) |>
  summarise(
    n_rows = n(),
    n_subjects = n_distinct(Subject),
    n_bins = n_distinct(Bin),
    n_missing = sum(is.na(FC)),
    all_finite = all(is.finite(FC)),
    .groups = "drop"
  )
assert_true(all(completeness$n_rows == 320L), "Each ROI x group must contain 32 subjects x 10 bins.")
assert_true(all(completeness$n_subjects == 32L) && all(completeness$n_bins == 10L), "ROI completeness failed.")

sma_data <- selected_raw |>
  filter(Region == "SMA") |>
  transmute(Subject, Group, Bin, FC, Region, Stage, N_ROIs)

ifg_composite <- selected_raw |>
  filter(Region == "IFG") |>
  group_by(Subject, Group, Bin, Region, Stage, N_ROIs) |>
  summarise(
    FC = mean(FC),
    n_ROIs_observed = n_distinct(ROI),
    min_ROI_FC = min(FC),
    max_ROI_FC = max(FC),
    .groups = "drop"
  ) |>
  arrange(Subject, Bin)
assert_true(nrow(ifg_composite) == 640L, "IFG composite is not 64 subjects x 10 bins.")
assert_true(all(ifg_composite$n_ROIs_observed == 6L), "IFG composite does not use all six ROIs in every Subject x Bin cell.")

cingulate_data <- selected_raw |>
  filter(Region == "Cingulate") |>
  transmute(Subject, Group, Bin, FC, Region, Stage, N_ROIs)
assert_true(nrow(cingulate_data) == 640L, "Cingulate data are not 64 subjects x 10 bins.")
assert_true(n_distinct(cingulate_data$Subject) == 64L, "Cingulate does not contain 64 unique participants.")
assert_true(all(table(cingulate_data$Group, cingulate_data$Bin) == 32L), "Cingulate is not complete at 32 subjects/group/bin.")

analysis_data <- bind_rows(
  sma_data,
  ifg_composite |> select(Subject, Group, Bin, FC, Region, Stage, N_ROIs),
  cingulate_data
) |>
  mutate(
    Region = factor(Region, levels = c("SMA", "IFG", "Cingulate")),
    Group = factor(Group, levels = c("HC", "SZ"))
  ) |>
  arrange(Region, Subject, Bin)
assert_true(nrow(analysis_data) == 3L * 640L, "Three-region analysis table is not 1,920 rows.")
assert_true(!anyDuplicated(analysis_data[c("Region", "Subject", "Bin")]), "Three-region analysis keys are duplicated.")

age_center <- mean(covariates$Age)
fd_center <- mean(covariates$MeanFD)

prepare_model_data <- function(data) {
  out <- data |>
    left_join(covariates, by = c("Subject", "Group")) |>
    mutate(
      Group = factor(Group, levels = c("HC", "SZ")),
      Sex = factor(Sex, levels = c("Female", "Male")),
      Bin_c = as.numeric(Bin) - 5.5,
      Bin_f = factor(Bin, levels = 1:10),
      Age_c = Age - age_center,
      MeanFD_c = MeanFD - fd_center
    ) |>
    arrange(Subject, Bin)
  assert_true(!anyNA(out[c("FC", "Group", "Bin", "Age_c", "Sex", "MeanFD_c")]), "Model data contain missing values.")
  out
}

cluster_vcov_hc1 <- function(fit, cluster) {
  X <- model.matrix(fit)
  residual <- residuals(fit)
  cluster <- factor(cluster)
  G <- nlevels(cluster)
  N <- nrow(X)
  K <- ncol(X)
  assert_true(G > 1L && N > K, "Invalid cluster-robust covariance dimensions.")
  cluster_scores <- rowsum(X * residual, group = cluster, reorder = FALSE)
  bread <- solve(crossprod(X))
  meat <- crossprod(cluster_scores)
  finite_sample <- (G / (G - 1)) * ((N - 1) / (N - K))
  vcov <- finite_sample * bread %*% meat %*% bread
  dimnames(vcov) <- list(colnames(X), colnames(X))
  vcov
}

coef_test_row <- function(fit, vcov, term, test, definition, region, stage, n_rois) {
  estimate <- unname(coef(fit)[term])
  se <- sqrt(vcov[term, term])
  df <- length(unique(model.frame(fit)$Subject)) - 1L
  t_value <- estimate / se
  p <- 2 * pt(abs(t_value), df = df, lower.tail = FALSE)
  critical <- qt(0.975, df = df)
  tibble(
    Region = region,
    Stage = stage,
    N_ROIs = n_rois,
    Test = test,
    Definition = definition,
    Estimate_type = "Beta",
    Beta_or_F = estimate,
    SE_cluster = se,
    CI95_low = estimate - critical * se,
    CI95_high = estimate + critical * se,
    Statistic = t_value,
    Statistic_type = "cluster-robust t",
    df_num = 1,
    df_denom = df,
    p = p
  )
}

fit_target_tests <- function(data, region, stage, n_rois) {
  data <- prepare_model_data(data)
  assert_true(nrow(data) == 640L && n_distinct(data$Subject) == 64L, paste(region, "model data must be 64 x 10."))

  continuous <- lm(FC ~ Group * Bin_c + Age_c + Sex + MeanFD_c, data = data)
  continuous$model$Subject <- data$Subject
  vc_cont <- cluster_vcov_hc1(continuous, data$Subject)
  offset <- coef_test_row(
    continuous, vc_cont, "GroupSZ", "Offset",
    "Adjusted SZ-HC difference at centered Bin=5.5",
    region, stage, n_rois
  )
  slope <- coef_test_row(
    continuous, vc_cont, "GroupSZ:Bin_c", "Linear slope",
    "Adjusted SZ-HC linear slope difference per fixed G1 bin",
    region, stage, n_rois
  )

  categorical <- lm(FC ~ Group * Bin_f + Age_c + Sex + MeanFD_c, data = data)
  categorical$model$Subject <- data$Subject
  vc_cat <- cluster_vcov_hc1(categorical, data$Subject)
  interaction_terms <- paste0("GroupSZ:Bin_f", 2:10)
  assert_true(all(interaction_terms %in% names(coef(categorical))), "Categorical model lacks nine Group x Bin terms.")
  beta <- coef(categorical)[interaction_terms]
  vc_sub <- vc_cat[interaction_terms, interaction_terms, drop = FALSE]
  q <- length(interaction_terms)
  f_value <- as.numeric(t(beta) %*% solve(vc_sub, beta) / q)
  df_denom <- n_distinct(data$Subject) - 1L
  p_f <- pf(f_value, df1 = q, df2 = df_denom, lower.tail = FALSE)
  shape <- tibble(
    Region = region,
    Stage = stage,
    N_ROIs = n_rois,
    Test = "Binned shape",
    Definition = "Adjusted 9-df joint Group x categorical fixed-G1-Bin test",
    Estimate_type = "F",
    Beta_or_F = f_value,
    SE_cluster = NA_real_,
    CI95_low = NA_real_,
    CI95_high = NA_real_,
    Statistic = f_value,
    Statistic_type = "cluster-robust F",
    df_num = q,
    df_denom = df_denom,
    p = p_f
  )

  bind_rows(offset, slope, shape)
}

tests_9 <- bind_rows(
  fit_target_tests(sma_data, "SMA", "Anticipation", 1L),
  fit_target_tests(ifg_composite |> select(Subject, Group, Bin, FC), "IFG", "Outcome", 6L),
  fit_target_tests(cingulate_data, "Cingulate", "Outcome", 1L)
) |>
  mutate(
    Region = factor(Region, levels = c("SMA", "IFG", "Cingulate")),
    Test = factor(Test, levels = c("Offset", "Linear slope", "Binned shape")),
    BH_q_9_tests = p.adjust(p, method = "BH"),
    FDR_significant_9 = BH_q_9_tests < 0.05,
    Multiplicity_family = "Exploratory BH across 3 targets x 3 prespecified Raw-FC tests"
  ) |>
  arrange(Region, Test)

bh_manual <- function(p) {
  order_index <- order(p)
  sorted <- p[order_index]
  adjusted_sorted <- pmin(1, rev(cummin(rev(sorted * length(sorted) / seq_along(sorted)))))
  out <- numeric(length(p))
  out[order_index] <- adjusted_sorted
  out
}
tests_9$BH_q_9_manual <- bh_manual(tests_9$p)
assert_true(max(abs(tests_9$BH_q_9_tests - tests_9$BH_q_9_manual)) < 1e-15, "Independent BH-9 verification failed.")

historical_6_recomputed <- tests_9 |>
  filter(Region %in% c("SMA", "IFG")) |>
  mutate(BH_q_6_recomputed = p.adjust(p, method = "BH"))

historical_6_standard <- historical_6 |>
  transmute(
    Region = target,
    Test = recode(test,
      offset_difference = "Offset",
      slope_difference = "Linear slope",
      categorical_binned_shape = "Binned shape"
    ),
    beta_historical = beta,
    SE_historical = SE_cluster,
    CI95_low_historical = CI95_low,
    CI95_high_historical = CI95_high,
    statistic_historical = statistic,
    p_historical = p_raw,
    BH_q_6_historical = q_BH_6_tests
  )

history_check <- historical_6_recomputed |>
  mutate(Region = as.character(Region), Test = as.character(Test)) |>
  left_join(historical_6_standard, by = c("Region", "Test")) |>
  mutate(
    beta_abs_diff = abs(Beta_or_F - beta_historical),
    SE_abs_diff = abs(SE_cluster - SE_historical),
    CI_low_abs_diff = abs(CI95_low - CI95_low_historical),
    CI_high_abs_diff = abs(CI95_high - CI95_high_historical),
    statistic_abs_diff = abs(Statistic - statistic_historical),
    p_abs_diff = abs(p - p_historical),
    q6_abs_diff = abs(BH_q_6_recomputed - BH_q_6_historical),
    pass = coalesce(beta_abs_diff < 1e-10, TRUE) &
      coalesce(SE_abs_diff < 1e-10, TRUE) &
      coalesce(CI_low_abs_diff < 1e-10, TRUE) &
      coalesce(CI_high_abs_diff < 1e-10, TRUE) &
      statistic_abs_diff < 1e-10 & p_abs_diff < 1e-10 & q6_abs_diff < 1e-10
  )
assert_true(nrow(history_check) == 6L && all(history_check$pass), "SMA/IFG six-test family was not reproduced exactly.")

bn29_data <- selected_raw |>
  filter(ROI == "L_IFG_6_1_A44d__BN29") |>
  select(Subject, Group, Bin, FC)
method_recomputed <- bind_rows(
  fit_target_tests(sma_data, "SMA", "Anticipation", 1L),
  fit_target_tests(bn29_data, "IFG", "Outcome", 1L)
) |>
  mutate(
    locked_test_name = recode(as.character(Test),
      Offset = "offset_difference",
      `Linear slope` = "slope_difference",
      `Binned shape` = "categorical_binned_shape"
    ),
    BH_q_6_recomputed = p.adjust(p, method = "BH")
  )
method_locked <- locked_B08 |>
  transmute(
    Region = subfamily,
    locked_test_name = test,
    beta_locked = beta,
    SE_locked = SE_cluster,
    statistic_locked = statistic,
    p_locked = p_raw,
    q_locked = q_BH_6_tests
  )
method_check <- method_recomputed |>
  mutate(Region = as.character(Region)) |>
  left_join(method_locked, by = c("Region", "locked_test_name")) |>
  mutate(
    beta_abs_diff = abs(Beta_or_F - beta_locked),
    SE_abs_diff = abs(SE_cluster - SE_locked),
    statistic_abs_diff = abs(Statistic - statistic_locked),
    p_abs_diff = abs(p - p_locked),
    q_abs_diff = abs(BH_q_6_recomputed - q_locked),
    pass = coalesce(beta_abs_diff < 1e-10, TRUE) & coalesce(SE_abs_diff < 1e-10, TRUE) &
      statistic_abs_diff < 1e-10 & p_abs_diff < 1e-10 & q_abs_diff < 1e-10
  )
assert_true(nrow(method_check) == 6L && all(method_check$pass), "HC1 implementation does not reproduce locked B08 SMA/BN29 results.")

q_comparison <- tests_9 |>
  mutate(Region_chr = as.character(Region), Test_chr = as.character(Test)) |>
  select(Region = Region_chr, Test = Test_chr, p_recomputed = p, BH_q_9_tests) |>
  left_join(
    historical_6_standard |>
      select(Region, Test, p_historical, BH_q_6_historical),
    by = c("Region", "Test")
  ) |>
  mutate(
    p_unchanged_from_historical_6 = ifelse(is.na(p_historical), NA, abs(p_recomputed - p_historical) < 1e-10),
    q_family_note = ifelse(is.na(BH_q_6_historical),
      "New Cingulate row; no six-test q",
      "Historical q uses SMA+IFG six-test family; new q uses three-region nine-test family"
    )
  )

curve_summary <- analysis_data |>
  group_by(Region, Stage, N_ROIs, Group, Bin) |>
  summarise(n = n_distinct(Subject), mean = mean(FC), sd = sd(FC), .groups = "drop") |>
  mutate(
    se = sd / sqrt(n),
    t_critical = qt(0.975, df = n - 1),
    CI95_low = mean - t_critical * se,
    CI95_high = mean + t_critical * se
  )
assert_true(nrow(curve_summary) == 60L && all(curve_summary$n == 32L), "Three-region curve summary is incomplete.")

contribution_3 <- contribution_all |>
  filter(subfamily %in% c("SMA", "IFG", "Cingulate")) |>
  mutate(Region = factor(subfamily, levels = c("SMA", "IFG", "Cingulate"))) |>
  arrange(Region) |>
  select(
    Region, Stage = stage, N_ROIs, contribution_beta_SZ_minus_HC = beta_SZ_minus_HC,
    contribution_CI95_low = CI95_low, contribution_CI95_high = CI95_high,
    contribution_p = p_raw, contribution_BH_q_13 = q_FDR_13_assigned,
    contribution_FDR_significant = FDR_significant
  )
assert_true(nrow(contribution_3) == 3L, "Contribution source must contain SMA, IFG and Cingulate exactly once.")
cing_contribution <- contribution_3 |> filter(Region == "Cingulate")
assert_true(
  abs(cing_contribution$contribution_beta_SZ_minus_HC - 0.00121211175507039) < 1e-14 &&
    abs(cing_contribution$contribution_BH_q_13 - 0.00192672315739131) < 1e-14,
  "Locked Cingulate Contribution values differ from the expected read-only result."
)

validation_table <- tests_9 |>
  mutate(
    Region = as.character(Region),
    Test = as.character(Test),
    CI95 = ifelse(Estimate_type == "F", NA_character_, sprintf("[%+.9f, %+.9f]", CI95_low, CI95_high)),
    F_df = ifelse(Estimate_type == "F", sprintf("F(%d,%d)", df_num, df_denom), NA_character_)
  ) |>
  left_join(
    q_comparison |> select(Region, Test, BH_q_6_historical),
    by = c("Region", "Test")
  ) |>
  select(
    Region, Stage, N_ROIs, Test, Estimate_type, Beta_or_F, CI95, CI95_low, CI95_high,
    SE_cluster, Statistic, Statistic_type, F_df, df_num, df_denom, p,
    BH_q_9_tests, FDR_significant_9, BH_q_6_historical, Multiplicity_family
  )

source_manifest <- tribble(
  ~role, ~path, ~use,
  "Canonical subject-bin-ROI Fisher-z FC", file.path("CURRENT64_CANONICAL_RERUN", "04_BINWISE_FC_PROFILES", "subject_bin_roi_FC_long.tsv"), "All three Raw FC targets",
  "Canonical ROI order and mapping", file.path("CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "roi_order", "canonical_roi_order_65.tsv"), "ROI identity, stage, subfamily and array column",
  "Frozen covariates", file.path("CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "covariates", "current64_covariates.tsv"), "Age, Sex and MeanFD",
  "Frozen 375-person reference G1", file.path("CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "canonical_G1", "375_reference_G1_manuscript_2608.npy"), "Read-only provenance; not recalculated",
  "Frozen fixed G1 bins", file.path("CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "canonical_bins", "375_reference_bins_manuscript_2608.npy"), "Read-only provenance; canonical Bin is consumed from the FC table",
  "Frozen Fisher-z arrays", file.path("CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "fc_fingerprints_2608x65", "<Subject>_fingerprint_z_2608x65.npy"), "Upstream 2608 x 65 source; read-only provenance",
  "Historical two-target six-test results", file.path("IFG_6ROI_rawFC_reanalysis_20261028", "tables", "recomputed_2targets_6tests.tsv"), "Locked SMA + six-ROI IFG comparison",
  "Locked original B08 tests", file.path("figures", "derived_results", "B08_SMA_IFG_raw_FC_tests.tsv"), "HC1 implementation reproduction using SMA + BN29",
  "Locked Contribution results", file.path("figures", "derived_results", "B06_subfamily_signed_contributions.tsv"), "Read-only values for optional evidence matrix"
)

write_tsv_stable(roi_mapping, file.path(out_dir, "tables", "three_region_ROI_mapping.tsv"))
write_tsv_stable(completeness, file.path(out_dir, "tables", "ROI_data_completeness.tsv"))
write_tsv_stable(analysis_data, file.path(out_dir, "tables", "three_region_subject_bin_FC.tsv"))
write_tsv_stable(ifg_composite, file.path(out_dir, "tables", "IFG_6ROI_subject_bin_mean.tsv"))
write_tsv_stable(curve_summary, file.path(out_dir, "tables", "three_region_curve_summary.tsv"))
write_tsv_stable(validation_table, file.path(out_dir, "tables", "three_region_rawFC_validation.tsv"))
write_tsv_stable(historical_6, file.path(out_dir, "tables", "historical_2targets_6tests.tsv"))
write_tsv_stable(q_comparison, file.path(out_dir, "tables", "BH6_vs_BH9_comparison.tsv"))
write_tsv_stable(history_check, file.path(out_dir, "tables", "SMA_IFG_reproduction_check.tsv"))
write_tsv_stable(method_check, file.path(out_dir, "tables", "HC1_method_reproduction_against_locked_B08.tsv"))
write_tsv_stable(contribution_3, file.path(out_dir, "tables", "Contribution_values_readonly.tsv"))
write_tsv_stable(source_manifest, file.path(out_dir, "tables", "source_manifest.tsv"))
writeLines(capture.output(sessionInfo()), file.path(out_dir, "tables", "R_sessionInfo.txt"), useBytes = TRUE)

COL_HC <- "#2D6F9E"
COL_SZ <- "#C65D3A"
COL_INK <- "#222222"
COL_GREY <- "#656B70"
COL_GRID <- "#D9DEE3"
COL_SUPPORT <- "#3E6176"
COL_NOSUPPORT <- "#EEF1F3"
FONT <- "Arial"

theme_validation <- function(base_size = 7) {
  theme_classic(base_size = base_size, base_family = FONT) +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = COL_INK),
      axis.ticks = element_line(linewidth = 0.35, colour = COL_INK),
      axis.title = element_text(size = base_size, colour = COL_INK),
      axis.text = element_text(size = base_size - 0.3, colour = COL_INK),
      legend.position = "top",
      legend.justification = "left",
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.2),
      legend.key.width = grid::unit(12, "pt"),
      panel.grid.major.y = element_line(colour = COL_GRID, linewidth = 0.28),
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = base_size + 1.0, face = "bold", colour = COL_INK, margin = margin(0, 0, 2, 0)),
      plot.subtitle = element_text(size = base_size - 0.2, colour = COL_GREY, margin = margin(0, 0, 3, 0)),
      plot.caption = element_text(size = base_size - 0.7, colour = COL_GREY, hjust = 0),
      plot.tag = element_text(size = base_size + 1.2, face = "bold", colour = COL_INK),
      plot.margin = margin(6, 7, 6, 7, unit = "pt")
    )
}

fmt_q <- function(x) {
  ifelse(x < 0.001, format(x, scientific = TRUE, digits = 2), sprintf("%.3f", x))
}

group_components <- list(
  scale_colour_manual(values = c(HC = COL_HC, SZ = COL_SZ)),
  scale_fill_manual(values = c(HC = COL_HC, SZ = COL_SZ)),
  scale_shape_manual(values = c(HC = 21, SZ = 24)),
  scale_linetype_manual(values = c(HC = "22", SZ = "solid"))
)

all_y <- range(c(curve_summary$CI95_low, curve_summary$CI95_high), finite = TRUE)
all_y_span <- diff(all_y)
shared_y <- all_y + c(-0.06, 0.08) * all_y_span
shared_y <- range(c(shared_y, 0))

make_curve_panel <- function(region, panel_title, tag = NULL, show_y = TRUE, single = FALSE) {
  dat <- curve_summary |> filter(Region == region)
  stat <- tests_9 |> filter(Region == region)
  q_slope <- stat$BH_q_9_tests[stat$Test == "Linear slope"]
  q_shape <- stat$BH_q_9_tests[stat$Test == "Binned shape"]
  region_map <- roi_mapping |> filter(Region == region)
  roi_label <- if (region == "IFG") {
    "BN29/31/33/35/37/39 mean | Outcome"
  } else if (single) {
    paste0(region_map$ROI[1], " | ", region_map$Stage[1])
  } else {
    paste0(region_map$atlas_label[1], " | ", region_map$Stage[1])
  }
  subtitle <- if (single) roi_label else paste0("Linear q = ", fmt_q(q_slope), " | Binned q = ", fmt_q(q_shape))

  p <- ggplot(
    dat,
    aes(x = Bin, y = mean, colour = Group, fill = Group, shape = Group, linetype = Group, group = Group)
  ) +
    geom_ribbon(aes(ymin = CI95_low, ymax = CI95_high), colour = NA, alpha = 0.11, show.legend = FALSE) +
    geom_line(linewidth = if (single) 0.75 else 0.62) +
    geom_point(size = if (single) 2.0 else 1.55, stroke = 0.5, fill = "white") +
    geom_hline(yintercept = 0, colour = "#B7BEC5", linewidth = 0.32) +
    group_components +
    scale_x_continuous(breaks = if (single) 1:10 else c(1, 5, 10), limits = c(0.8, 10.2)) +
    scale_y_continuous(limits = shared_y) +
    labs(
      title = panel_title,
      subtitle = subtitle,
      tag = tag,
      x = "Fixed G1 bin",
      y = if (show_y) "Mean Fisher-z FC" else NULL
    ) +
    theme_validation(if (single) 7.4 else 7.0)

  if (!show_y) {
    p <- p + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), axis.line.y = element_blank())
  }
  p
}

cing_tests <- tests_9 |> filter(Region == "Cingulate")
cing_slope <- cing_tests |> filter(Test == "Linear slope")
cing_shape <- cing_tests |> filter(Test == "Binned shape")

figure1 <- make_curve_panel(
  "Cingulate",
  "Cingulate raw FC profile",
  show_y = TRUE,
  single = TRUE
) +
  labs(
    caption = paste0(
      "n = 32/group; points are observed bin means and ribbons are t-based 95% CIs.\n",
      "Linear slope q = ", fmt_q(cing_slope$BH_q_9_tests),
      "; binned shape q = ", fmt_q(cing_shape$BH_q_9_tests), "."
    )
  )

figure2 <- (
  make_curve_panel("SMA", "SMA | Anticipation | 1 ROI", "a", TRUE, FALSE) |
    make_curve_panel("IFG", "IFG | Outcome | 6-ROI mean", "b", FALSE, FALSE) |
    make_curve_panel("Cingulate", "Cingulate | Outcome | 1 ROI", "c", FALSE, FALSE)
) +
  plot_layout(guides = "collect") &
  theme(legend.position = "top")

evidence_raw <- tests_9 |>
  mutate(
    Region = factor(Region, levels = c("SMA", "IFG", "Cingulate")),
    Evidence = recode(as.character(Test),
      Offset = "Raw FC\noffset",
      `Linear slope` = "Raw FC\nlinear slope",
      `Binned shape` = "Raw FC\nbinned shape"
    ),
    q = BH_q_9_tests,
    significant = FDR_significant_9,
    family = "Raw FC: BH across 9 tests"
  ) |>
  select(Region, Evidence, q, significant, family)

evidence_contribution <- contribution_3 |>
  transmute(
    Region,
    Evidence = "Contribution",
    q = contribution_BH_q_13,
    significant = contribution_FDR_significant,
    family = "Contribution: locked BH across 13 assigned subfamilies"
  )

evidence <- bind_rows(evidence_contribution, evidence_raw) |>
  mutate(
    Evidence = factor(Evidence, levels = c("Contribution", "Raw FC\noffset", "Raw FC\nlinear slope", "Raw FC\nbinned shape")),
    Region = factor(Region, levels = c("SMA", "IFG", "Cingulate")),
    cell_label = paste0(ifelse(significant, "Supported", "Not supported"), "\nq = ", fmt_q(q)),
    fill_colour = ifelse(significant, COL_SUPPORT, COL_NOSUPPORT),
    text_colour = ifelse(significant, "white", COL_INK)
  )

figure3 <- ggplot(evidence, aes(x = Evidence, y = Region)) +
  geom_tile(aes(fill = fill_colour), width = 0.94, height = 0.76, colour = "white", linewidth = 0.9) +
  geom_text(aes(label = cell_label, colour = text_colour), family = FONT, size = 2.45, lineheight = 0.92) +
  scale_fill_identity() +
  scale_colour_identity() +
  scale_x_discrete(position = "top", expand = expansion(add = 0.04)) +
  scale_y_discrete(limits = rev) +
  labs(
    title = "Contribution and raw FC validation are complementary",
    x = NULL,
    y = NULL,
    caption = "Contribution q values are unchanged locked BH-13 results; Raw FC q values use the new exploratory BH-9 family.\nEffect sizes are not compared across scales."
  ) +
  theme_void(base_family = FONT, base_size = 7) +
  theme(
    axis.text.x = element_text(face = "bold", colour = COL_INK, lineheight = 0.95, margin = margin(0, 0, 4, 0)),
    axis.text.y = element_text(face = "bold", colour = COL_INK, margin = margin(0, 4, 0, 0)),
    plot.title = element_text(size = 8.4, face = "bold", colour = COL_INK, margin = margin(0, 0, 7, 0)),
    plot.caption = element_text(size = 6.2, colour = COL_GREY, hjust = 0, margin = margin(7, 0, 0, 0)),
    plot.margin = margin(7, 8, 7, 8, unit = "pt")
  )

save_plot_bundle <- function(plot, filename, width_mm, height_mm) {
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4
  svg_path <- paste0(filename, ".svg")
  pdf_path <- paste0(filename, ".pdf")
  png_path <- paste0(filename, ".png")

  svglite::svglite(svg_path, width = width_in, height = height_in, bg = "white")
  print(plot)
  dev.off()

  grDevices::cairo_pdf(pdf_path, width = width_in, height = height_in, family = FONT, bg = "white")
  print(plot)
  dev.off()

  ragg::agg_png(png_path, width = width_in, height = height_in, units = "in", res = 300, background = "white")
  print(plot)
  dev.off()

  outputs <- c(svg_path, pdf_path, png_path)
  assert_true(all(file.exists(outputs)) && all(file.info(outputs)$size > 1000), paste("Figure export failed:", basename(filename)))
  outputs
}

figure1_files <- save_plot_bundle(
  figure1,
  file.path(out_dir, "figures", "Figure1_Cingulate_rawFC_profile"),
  width_mm = 135,
  height_mm = 92
)
figure2_files <- save_plot_bundle(
  figure2,
  file.path(out_dir, "figures", "Figure2_three_region_rawFC_validation"),
  width_mm = 183,
  height_mm = 82
)
figure3_files <- save_plot_bundle(
  figure3,
  file.path(out_dir, "figures", "Figure3_Contribution_vs_rawFC_support"),
  width_mm = 165,
  height_mm = 78
)

cing_offset <- cing_tests |> filter(Test == "Offset")
raw_supported <- any(cing_tests$FDR_significant_9)
direction_note <- if (raw_supported) {
  "Cingulate obtains Raw FC support in at least one prespecified test."
} else {
  "Cingulate does not obtain Raw FC support after BH correction across nine Raw FC tests."
}

qa_lines <- c(
  "# QA report: three-region Raw FC validation",
  "",
  "## Scope protection",
  "",
  "- No Contribution result was recalculated or modified; values were read from the locked contribution table.",
  "- The 375-person reference G1 and fixed-bin definitions were not regenerated or modified.",
  "- No prior analysis file or PPT was overwritten.",
  "",
  "## ROI mapping",
  "",
  "- SMA: R_SFG_7_5_A6m_SMA__BN10, Anticipation, one ROI.",
  "- IFG: BN29/31/33/35/37/39, Outcome, six ROIs averaged within Subject x Bin.",
  "- Cingulate: R_CG_7_7_A32sg_ACC__BN188, Outcome, zero-based array column 42 / one-based column 43.",
  "",
  "## Data completeness",
  "",
  "- Every selected ROI contains 64 subjects x 10 bins, with 32 HC and 32 SZ per bin.",
  "- All selected Fisher-z FC values are finite; Subject x Bin x ROI keys are unique.",
  "- IFG composite has 640 rows and n_ROIs_observed = 6 for every Subject x Bin.",
  "- Cingulate has 640 rows and 64 unique participants.",
  "",
  "## Statistical implementation",
  "",
  "- Continuous model: FC ~ Group * (Bin - 5.5) + Age_c + Sex + MeanFD_c.",
  "- Categorical model: FC ~ Group * factor(Bin) + Age_c + Sex + MeanFD_c.",
  "- Clustered HC1 covariance: (G/(G-1))*((N-1)/(N-K)) times the cluster sandwich; G=64.",
  "- Coefficient tests use t(63); the joint nine-term shape test uses F(9,63).",
  "- R implementation reproduces locked original B08 SMA/BN29 beta, SE, statistic, p and q within 1e-10.",
  "- Recomputed SMA and six-ROI IFG beta, SE, CI, statistic, p and historical BH-6 q match the prior reanalysis within 1e-10.",
  "- All six historical SMA/IFG raw p values are unchanged after adding Cingulate.",
  "- BH-9 q values from p.adjust(method='BH') match an independent manual BH implementation within 1e-15.",
  "",
  "## Figure integrity",
  "",
  "- All formal panels use the real canonical Fisher-z FC values; no smoothing, synthetic data or coefficient-to-pseudodata conversion was used.",
  "- Curves show observed bin means with t-based 95% CIs across participants.",
  "- All figures were generated in R and exported as editable SVG, PDF and 300-dpi PNG.",
  "- Figure 3 uses support tiles rather than placing Contribution and Raw FC beta values on a shared numeric axis.",
  "",
  paste0("## Overall Raw FC conclusion\n\n", direction_note)
)
writeLines(qa_lines, file.path(out_dir, "QA_REPORT.md"), useBytes = TRUE)

summary_lines <- c(
  "# Results summary: SMA, IFG and Cingulate Raw FC validation",
  "",
  "## Cingulate Raw FC",
  "",
  sprintf("- Offset at Bin 5.5: beta=%+.9f, 95%% CI [%+.9f, %+.9f], p=%.9g, BH-9 q=%.9g.", cing_offset$Beta_or_F, cing_offset$CI95_low, cing_offset$CI95_high, cing_offset$p, cing_offset$BH_q_9_tests),
  sprintf("- Linear slope: beta=%+.9f, 95%% CI [%+.9f, %+.9f], p=%.9g, BH-9 q=%.9g.", cing_slope$Beta_or_F, cing_slope$CI95_low, cing_slope$CI95_high, cing_slope$p, cing_slope$BH_q_9_tests),
  sprintf("- Binned shape: F(%d,%d)=%.9f, p=%.9g, BH-9 q=%.9g.", cing_shape$df_num, cing_shape$df_denom, cing_shape$Beta_or_F, cing_shape$p, cing_shape$BH_q_9_tests),
  "",
  "## Interpretation",
  "",
  paste0("- ", direction_note),
  "- The positive locked Contribution effect is therefore treated as complementary profile-similarity evidence; its direction is not assumed to imply a positive Raw FC offset or slope.",
  "- Historical SMA/IFG beta, SE and p values are unchanged. Their q values are reported separately for the historical BH-6 and expanded exploratory BH-9 families.",
  "",
  "## Main files",
  "",
  "- tables/three_region_rawFC_validation.tsv",
  "- tables/BH6_vs_BH9_comparison.tsv",
  "- figures/Figure1_Cingulate_rawFC_profile.*",
  "- figures/Figure2_three_region_rawFC_validation.*",
  "- figures/Figure3_Contribution_vs_rawFC_support.*",
  "- scripts/run_three_region_rawFC_validation.R"
)
writeLines(summary_lines, file.path(out_dir, "RESULTS_SUMMARY.md"), useBytes = TRUE)

qa_table <- tibble(
  check = c(
    "Cingulate canonical mapping", "Each ROI complete 64x10", "IFG six ROIs per subject-bin",
    "Cingulate 640 rows", "All selected FC finite", "Locked B08 HC1 reproduction",
    "Prior SMA/IFG six-test reproduction", "Raw p unchanged for prior six", "Independent BH-9 check",
    "Figure 1 exports", "Figure 2 exports", "Figure 3 exports"
  ),
  observed = c(
    paste(cing_map$ROI, cing_map$Stage, paste0("column1=", cing_map$ROI_array_column_1based)),
    all(completeness$n_rows == 320L), all(ifg_composite$n_ROIs_observed == 6L), nrow(cingulate_data),
    all(is.finite(selected_raw$FC)), all(method_check$pass), all(history_check$pass),
    all(q_comparison$p_unchanged_from_historical_6[!is.na(q_comparison$p_unchanged_from_historical_6)]),
    max(abs(tests_9$BH_q_9_tests - tests_9$BH_q_9_manual)),
    all(file.exists(figure1_files)), all(file.exists(figure2_files)), all(file.exists(figure3_files))
  ),
  pass = c(
    TRUE, all(completeness$n_rows == 320L), all(ifg_composite$n_ROIs_observed == 6L), nrow(cingulate_data) == 640L,
    all(is.finite(selected_raw$FC)), all(method_check$pass), all(history_check$pass),
    all(q_comparison$p_unchanged_from_historical_6[!is.na(q_comparison$p_unchanged_from_historical_6)]),
    max(abs(tests_9$BH_q_9_tests - tests_9$BH_q_9_manual)) < 1e-15,
    all(file.exists(figure1_files)), all(file.exists(figure2_files)), all(file.exists(figure3_files))
  )
)
write_tsv_stable(qa_table, file.path(out_dir, "tables", "QA_checks.tsv"))
assert_true(all(qa_table$pass), "At least one final QA check failed.")

artifact_files <- list.files(out_dir, recursive = TRUE, full.names = TRUE)
artifact_files <- artifact_files[file.info(artifact_files)$isdir %in% FALSE]
file_manifest <- tibble(
  relative_path = substring(normalizePath(artifact_files, winslash = "/"), nchar(out_dir) + 2L),
  bytes = file.info(artifact_files)$size
) |>
  arrange(relative_path)
write_tsv_stable(file_manifest, file.path(out_dir, "tables", "output_file_manifest.tsv"))

cat("\n=== CINGULATE RAW FC: THREE TESTS ===\n")
cat(sprintf("Offset: beta=%+.9f; 95%% CI [%+.9f, %+.9f]; p=%.9g; BH9 q=%.9g\n", cing_offset$Beta_or_F, cing_offset$CI95_low, cing_offset$CI95_high, cing_offset$p, cing_offset$BH_q_9_tests))
cat(sprintf("Linear slope: beta=%+.9f; 95%% CI [%+.9f, %+.9f]; p=%.9g; BH9 q=%.9g\n", cing_slope$Beta_or_F, cing_slope$CI95_low, cing_slope$CI95_high, cing_slope$p, cing_slope$BH_q_9_tests))
cat(sprintf("Binned shape: F(%d,%d)=%.9f; p=%.9g; BH9 q=%.9g\n", cing_shape$df_num, cing_shape$df_denom, cing_shape$Beta_or_F, cing_shape$p, cing_shape$BH_q_9_tests))

cat("\n=== NEW BH-9 Q VALUES FOR SMA AND IFG ===\n")
for (region in c("SMA", "IFG")) {
  rows <- tests_9 |> filter(Region == region)
  cat(region, "\n")
  for (i in seq_len(nrow(rows))) {
    cat(sprintf("  %s: p=%.9g; BH9 q=%.9g\n", as.character(rows$Test[i]), rows$p[i], rows$BH_q_9_tests[i]))
  }
}

cat("\n=== INTERPRETATION AND CONSISTENCY ===\n")
cat("Cingulate Raw FC support: ", ifelse(raw_supported, "YES", "NO"), "\n", sep = "")
cat(direction_note, "\n")
cat("Historical six vs new nine: beta/SE/p are unchanged for SMA and IFG; q values differ only because the multiplicity family expanded from 6 to 9 tests.\n")
cat("Contribution values were read-only and the 375-person reference G1 was not regenerated.\n")

cat("\n=== OUTPUT PATHS ===\n")
cat("Root: ", out_dir, "\n", sep = "")
cat("Three-region table: ", file.path(out_dir, "tables", "three_region_rawFC_validation.tsv"), "\n", sep = "")
cat("Figure 1: ", file.path(out_dir, "figures", "Figure1_Cingulate_rawFC_profile.[png|pdf|svg]"), "\n", sep = "")
cat("Figure 2: ", file.path(out_dir, "figures", "Figure2_three_region_rawFC_validation.[png|pdf|svg]"), "\n", sep = "")
cat("Figure 3: ", file.path(out_dir, "figures", "Figure3_Contribution_vs_rawFC_support.[png|pdf|svg]"), "\n", sep = "")
cat("Script: ", script_path, "\n", sep = "")
cat("THREE_REGION_RAW_FC_VALIDATION=PASS\n")
