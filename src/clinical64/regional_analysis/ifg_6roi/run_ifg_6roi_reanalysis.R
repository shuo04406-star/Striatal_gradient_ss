#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(svglite)
  library(ragg)
})

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
if (length(file_arg) != 1L) stop("Run this script with Rscript.", call. = FALSE)
script_path <- normalizePath(sub("^--file=", "", file_arg), winslash = "/", mustWork = TRUE)
out_dir <- Sys.getenv("STRIATAL_OUTPUT_ROOT", unset = NA_character_)
project_root <- Sys.getenv("STRIATAL_PROJECT_ROOT", unset = NA_character_)

paths <- list(
  fc_long = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "04_BINWISE_FC_PROFILES", "subject_bin_roi_FC_long.tsv"),
  fc_wide = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "04_BINWISE_FC_PROFILES", "subject_bin_roi_FC_wide.tsv"),
  covariates = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "covariates", "current64_covariates.tsv"),
  roi_order = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "roi_order", "canonical_roi_order_65.tsv"),
  locked_tests = file.path(project_root, "figures", "derived_results", "B08_SMA_IFG_raw_FC_tests.tsv"),
  frozen_fingerprint_dir = file.path(project_root, "CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "fc_fingerprints_2608x65")
)
stopifnot(all(file.exists(unlist(paths[c("fc_long", "fc_wide", "covariates", "roi_order", "locked_tests")]))))

dir.create(file.path(out_dir, "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "source_data"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "tables"), recursive = TRUE, showWarnings = FALSE)

write_tsv_stable <- function(x, path) {
  readr::write_tsv(x, path, na = "NA")
}

assert_true <- function(x, message) {
  if (!isTRUE(x)) stop(message, call. = FALSE)
}

fc <- readr::read_tsv(paths$fc_long, show_col_types = FALSE, progress = FALSE)
covariates <- readr::read_tsv(paths$covariates, show_col_types = FALSE, progress = FALSE)
roi_order <- readr::read_tsv(paths$roi_order, show_col_types = FALSE, progress = FALSE)
locked_tests <- readr::read_tsv(paths$locked_tests, show_col_types = FALSE, progress = FALSE)

required_fc <- c("Subject", "Group", "Bin", "ROI", "Subfamily", "RewardStage", "FC")
required_cov <- c("Subject", "Group", "Age", "Sex", "MeanFD")
assert_true(all(required_fc %in% names(fc)), "Canonical FC table is missing required columns.")
assert_true(all(required_cov %in% names(covariates)), "Covariate table is missing required columns.")
assert_true(nrow(fc) == 64L * 10L * 65L, "Canonical FC table is not 64 x 10 x 65.")
assert_true(nrow(covariates) == 64L, "Covariate table is not 64 participants.")

ifg_roi_manifest <- roi_order |>
  filter(subfamily == "IFG", reward_stage == "Outcome") |>
  arrange(roi_index) |>
  transmute(
    ROI_order_1based = roi_index + 1L,
    ROI_order_0based = roi_index,
    ROI = roi_name,
    parcel = sub("^L_IFG_6_[0-9]+_", "", sub("__BN[0-9]+$", "", roi_name)),
    Brainnetome = sub("^.*__(BN[0-9]+)$", "\\1", roi_name),
    Brainnetome_number = as.integer(sub("^.*__BN", "", roi_name)),
    atlas_label = atlas_source,
    canonical_long_file = file.path("CURRENT64_CANONICAL_RERUN", "04_BINWISE_FC_PROFILES", "subject_bin_roi_FC_long.tsv"),
    long_file_selector = paste0("ROI == ", roi_name),
    canonical_wide_file = file.path("CURRENT64_CANONICAL_RERUN", "04_BINWISE_FC_PROFILES", "subject_bin_roi_FC_wide.tsv"),
    wide_column = roi_name,
    frozen_subject_fingerprint_files = file.path("CURRENT64_CANONICAL_RERUN", "01_FROZEN_INPUTS", "fc_fingerprints_2608x65", "<Subject>_fingerprint_z_2608x65.npy"),
    fingerprint_column_1based = roi_index + 1L,
    fingerprint_column_0based = roi_index
  )

expected_ifg <- c(
  "L_IFG_6_1_A44d__BN29",
  "L_IFG_6_2_IFS__BN31",
  "L_IFG_6_3_A45c__BN33",
  "L_IFG_6_4_A45r__BN35",
  "L_IFG_6_5_A44op__BN37",
  "L_IFG_6_6_A44v__BN39"
)
assert_true(identical(ifg_roi_manifest$ROI, expected_ifg), "IFG ROI membership/order is not the locked six-ROI set.")

ifg_raw <- fc |>
  filter(ROI %in% expected_ifg, Subfamily == "IFG", RewardStage == "Outcome") |>
  mutate(
    ROI = factor(ROI, levels = expected_ifg),
    ROI_label = factor(
      sub("^L_IFG_6_[0-9]+_", "", as.character(ROI)) |>
        sub("__", " / ", x = _),
      levels = sub("^L_IFG_6_[0-9]+_", "", expected_ifg) |>
        sub("__", " / ", x = _)
    )
  )

assert_true(nrow(ifg_raw) == 64L * 10L * 6L, "IFG raw table is not 64 x 10 x 6.")
assert_true(!anyDuplicated(ifg_raw[c("Subject", "Bin", "ROI")]), "IFG Subject x Bin x ROI keys are duplicated.")
assert_true(all(is.finite(ifg_raw$FC)), "IFG raw Fisher-z FC contains non-finite values.")
assert_true(all(table(ifg_raw$ROI, ifg_raw$Group) == 320L), "Each IFG ROI must contain 32 subjects x 10 bins in each group.")

ifg_composite <- ifg_raw |>
  group_by(Subject, Group, Bin) |>
  summarise(
    FC = mean(FC),
    n_ROI = n_distinct(ROI),
    min_ROI_FC = min(FC),
    max_ROI_FC = max(FC),
    .groups = "drop"
  ) |>
  arrange(Subject, Bin)

assert_true(nrow(ifg_composite) == 64L * 10L, "IFG composite is not 64 x 10.")
assert_true(all(ifg_composite$n_ROI == 6L), "Not every Subject x Bin composite contains six IFG ROIs.")
assert_true(all(is.finite(ifg_composite$FC)), "IFG composite contains non-finite values.")

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
  assert_true(G > 1L && N > K, "Invalid dimensions for cluster-robust covariance.")
  cluster_scores <- rowsum(X * residual, group = cluster, reorder = FALSE)
  bread <- solve(crossprod(X))
  meat <- crossprod(cluster_scores)
  finite_sample <- (G / (G - 1)) * ((N - 1) / (N - K))
  vcov <- finite_sample * bread %*% meat %*% bread
  dimnames(vcov) <- list(colnames(X), colnames(X))
  vcov
}

coef_test_row <- function(fit, vcov, term, test, definition, target, stage, roi_scope) {
  estimate <- unname(coef(fit)[term])
  se <- sqrt(vcov[term, term])
  df <- length(unique(model.frame(fit)$Subject)) - 1L
  t_value <- estimate / se
  p <- 2 * pt(abs(t_value), df = df, lower.tail = FALSE)
  critical <- qt(0.975, df = df)
  tibble(
    target = target,
    stage = stage,
    ROI_scope = roi_scope,
    test = test,
    definition = definition,
    beta = estimate,
    SE_cluster = se,
    CI95_low = estimate - critical * se,
    CI95_high = estimate + critical * se,
    statistic = t_value,
    statistic_type = "cluster-robust t",
    df_num = 1,
    df_denom = df,
    p_raw = p
  )
}

fit_target_tests <- function(data, target, stage, roi_scope) {
  data <- prepare_model_data(data)
  assert_true(nrow(data) == 640L, paste(target, "must contain 64 x 10 rows."))
  assert_true(n_distinct(data$Subject) == 64L, paste(target, "must contain 64 participants."))

  continuous <- lm(FC ~ Group * Bin_c + Age_c + Sex + MeanFD_c, data = data)
  continuous$model$Subject <- data$Subject
  vc_cont <- cluster_vcov_hc1(continuous, data$Subject)
  offset <- coef_test_row(
    continuous, vc_cont, "GroupSZ", "offset_difference",
    "Adjusted SZ-HC difference at centered Bin=5.5",
    target, stage, roi_scope
  )
  slope <- coef_test_row(
    continuous, vc_cont, "GroupSZ:Bin_c", "slope_difference",
    "Adjusted SZ-HC linear slope difference per fixed G1 bin",
    target, stage, roi_scope
  )

  categorical <- lm(FC ~ Group * Bin_f + Age_c + Sex + MeanFD_c, data = data)
  categorical$model$Subject <- data$Subject
  vc_cat <- cluster_vcov_hc1(categorical, data$Subject)
  interaction_terms <- paste0("GroupSZ:Bin_f", 2:10)
  assert_true(all(interaction_terms %in% names(coef(categorical))), "Categorical model does not expose all nine Group x Bin terms.")
  beta <- coef(categorical)[interaction_terms]
  vc_sub <- vc_cat[interaction_terms, interaction_terms, drop = FALSE]
  q <- length(interaction_terms)
  f_value <- as.numeric(t(beta) %*% solve(vc_sub, beta) / q)
  df_denom <- n_distinct(data$Subject) - 1L
  p_f <- pf(f_value, df1 = q, df2 = df_denom, lower.tail = FALSE)
  shape <- tibble(
    target = target,
    stage = stage,
    ROI_scope = roi_scope,
    test = "categorical_binned_shape",
    definition = "Adjusted 9-df joint Group x categorical fixed-G1-Bin test",
    beta = NA_real_,
    SE_cluster = NA_real_,
    CI95_low = NA_real_,
    CI95_high = NA_real_,
    statistic = f_value,
    statistic_type = "cluster-robust F",
    df_num = q,
    df_denom = df_denom,
    p_raw = p_f
  )

  bind_rows(offset, slope, shape)
}

sma_data <- fc |>
  filter(ROI == "R_SFG_7_5_A6m_SMA__BN10") |>
  select(Subject, Group, Bin, FC)
bn29_data <- fc |>
  filter(ROI == "L_IFG_6_1_A44d__BN29") |>
  select(Subject, Group, Bin, FC)
ifg_composite_model <- ifg_composite |>
  select(Subject, Group, Bin, FC)

old_recomputed <- bind_rows(
  fit_target_tests(sma_data, "SMA", "Anticipation", "R_A6m / BN10 (single ROI)"),
  fit_target_tests(bn29_data, "IFG", "Outcome", "L_A44d / BN29 (single ROI)")
) |>
  mutate(
    q_BH_6_tests = p.adjust(p_raw, method = "BH"),
    FDR_significant = q_BH_6_tests < 0.05
  )

locked_for_check <- locked_tests |>
  transmute(
    target = subfamily,
    test,
    beta_locked = beta,
    SE_locked = SE_cluster,
    statistic_locked = statistic,
    p_locked = p_raw,
    q_locked = q_BH_6_tests
  )

reproduction_check <- old_recomputed |>
  left_join(locked_for_check, by = c("target", "test")) |>
  mutate(
    beta_abs_diff = abs(beta - beta_locked),
    SE_abs_diff = abs(SE_cluster - SE_locked),
    statistic_abs_diff = abs(statistic - statistic_locked),
    p_abs_diff = abs(p_raw - p_locked),
    q_abs_diff = abs(q_BH_6_tests - q_locked),
    pass = coalesce(beta_abs_diff < 1e-10, TRUE) &
      coalesce(SE_abs_diff < 1e-10, TRUE) &
      statistic_abs_diff < 1e-10 & p_abs_diff < 1e-10 & q_abs_diff < 1e-10
  )
assert_true(all(reproduction_check$pass), "R clustered-OLS implementation does not reproduce the locked BN29/SMA B08 statistics.")

new_all_tests <- bind_rows(
  fit_target_tests(sma_data, "SMA", "Anticipation", "R_A6m / BN10 (single ROI; unchanged comparator)"),
  fit_target_tests(ifg_composite_model, "IFG", "Outcome", "Mean of BN29/31/33/35/37/39 within Subject x Bin")
) |>
  mutate(
    q_BH_6_tests = p.adjust(p_raw, method = "BH"),
    FDR_significant = q_BH_6_tests < 0.05,
    multiplicity_family = "BH across 2 targets x 3 prespecified tests; IFG rows use the six-ROI composite"
  )

ifg_tests <- new_all_tests |>
  filter(target == "IFG")

old_ifg <- old_recomputed |>
  filter(target == "IFG") |>
  select(test, old_beta = beta, old_statistic = statistic, old_p_raw = p_raw, old_q_BH_6_tests = q_BH_6_tests)
comparison <- ifg_tests |>
  select(test, new_beta = beta, new_statistic = statistic, new_p_raw = p_raw, new_q_BH_6_tests = q_BH_6_tests) |>
  left_join(old_ifg, by = "test") |>
  mutate(
    old_scope = "BN29 only; audit comparator, not IFG-family inference",
    new_scope = "Six-ROI subject-within-bin Fisher-z mean"
  )

curve_summary <- ifg_composite |>
  group_by(Group, Bin) |>
  summarise(n = n_distinct(Subject), mean = mean(FC), sd = sd(FC), .groups = "drop") |>
  mutate(
    se = sd / sqrt(n),
    t_critical = qt(0.975, df = n - 1),
    CI95_low = mean - t_critical * se,
    CI95_high = mean + t_critical * se
  )
assert_true(nrow(curve_summary) == 20L && all(curve_summary$n == 32L), "Composite curve summary is incomplete.")

roi_curve_summary <- ifg_raw |>
  group_by(ROI, ROI_label, Group, Bin) |>
  summarise(n = n_distinct(Subject), mean = mean(FC), sd = sd(FC), .groups = "drop") |>
  mutate(
    se = sd / sqrt(n),
    t_critical = qt(0.975, df = n - 1),
    CI95_low = mean - t_critical * se,
    CI95_high = mean + t_critical * se
  )
assert_true(nrow(roi_curve_summary) == 120L && all(roi_curve_summary$n == 32L), "ROI-level curve summary is incomplete.")

roi_heterogeneity_descriptive <- ifg_raw |>
  group_by(ROI, ROI_label, Group, Subject) |>
  summarise(
    subject_linear_slope = unname(coef(lm(FC ~ Bin))["Bin"]),
    subject_mean_FC = mean(FC),
    .groups = "drop"
  ) |>
  group_by(ROI, ROI_label, Group) |>
  summarise(
    n_subjects = n_distinct(Subject),
    mean_subject_slope = mean(subject_linear_slope),
    sd_subject_slope = sd(subject_linear_slope),
    mean_subject_FC = mean(subject_mean_FC),
    sd_subject_FC = sd(subject_mean_FC),
    .groups = "drop"
  )

write_tsv_stable(ifg_roi_manifest, file.path(out_dir, "tables", "ifg_outcome_6roi_manifest.tsv"))
write_tsv_stable(ifg_composite, file.path(out_dir, "source_data", "ifg_6roi_subject_bin_composite.tsv"))
write_tsv_stable(curve_summary, file.path(out_dir, "source_data", "ifg_6roi_composite_curve_summary.tsv"))
write_tsv_stable(roi_curve_summary, file.path(out_dir, "source_data", "ifg_6roi_individual_curve_summary.tsv"))
write_tsv_stable(roi_heterogeneity_descriptive, file.path(out_dir, "tables", "ifg_6roi_heterogeneity_descriptive.tsv"))
write_tsv_stable(new_all_tests, file.path(out_dir, "tables", "recomputed_2targets_6tests.tsv"))
write_tsv_stable(ifg_tests, file.path(out_dir, "tables", "ifg_6roi_recomputed_tests.tsv"))
write_tsv_stable(comparison, file.path(out_dir, "tables", "bn29_vs_ifg6_scope_comparison.tsv"))
write_tsv_stable(reproduction_check, file.path(out_dir, "tables", "r_reproduction_check_against_locked_B08.tsv"))

COL_HC <- "#2D6F9E"
COL_SZ <- "#C65D3A"
COL_INK <- "#222222"
COL_GREY <- "#686E73"
COL_GRID <- "#D9DEE3"
FONT <- "Arial"

theme_ifg <- function(base_size = 7) {
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
      legend.key.width = grid::unit(13, "pt"),
      panel.grid.major.y = element_line(colour = COL_GRID, linewidth = 0.28),
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = base_size + 1.4, face = "bold", colour = COL_INK),
      plot.subtitle = element_text(size = base_size - 0.1, colour = COL_GREY, lineheight = 1.05),
      plot.caption = element_text(size = base_size - 0.7, colour = COL_GREY, hjust = 0, lineheight = 1.05),
      strip.background = element_rect(fill = "#F2F4F6", colour = NA),
      strip.text = element_text(size = base_size - 0.1, face = "bold", colour = COL_INK),
      plot.margin = margin(7, 8, 7, 8, unit = "pt")
    )
}

fmt_prob <- function(x) {
  ifelse(x < 0.001, format(x, scientific = TRUE, digits = 2), sprintf("%.3f", x))
}

slope_test <- ifg_tests |> filter(test == "slope_difference")
shape_test <- ifg_tests |> filter(test == "categorical_binned_shape")
offset_test <- ifg_tests |> filter(test == "offset_difference")

stat_subtitle <- paste0(
  "Subject x fixed-bin mean across BN29/31/33/35/37/39; n = 32/group.\n",
  "Adjusted slope difference: beta = ", sprintf("%+.4f", slope_test$beta),
  ", 95% CI [", sprintf("%+.4f", slope_test$CI95_low), ", ", sprintf("%+.4f", slope_test$CI95_high),
  "], p = ", fmt_prob(slope_test$p_raw), ", BH q = ", fmt_prob(slope_test$q_BH_6_tests), ".\n",
  "Binned profile: F(", shape_test$df_num, ", ", shape_test$df_denom, ") = ", sprintf("%.2f", shape_test$statistic),
  ", p = ", fmt_prob(shape_test$p_raw), ", BH q = ", fmt_prob(shape_test$q_BH_6_tests), "."
)

group_scales <- list(
  scale_colour_manual(values = c(HC = COL_HC, SZ = COL_SZ)),
  scale_fill_manual(values = c(HC = COL_HC, SZ = COL_SZ)),
  scale_shape_manual(values = c(HC = 21, SZ = 24)),
  scale_linetype_manual(values = c(HC = "22", SZ = "solid"))
)

composite_plot <- ggplot(
  curve_summary,
  aes(x = Bin, y = mean, colour = Group, fill = Group, shape = Group, linetype = Group, group = Group)
) +
  geom_ribbon(aes(ymin = CI95_low, ymax = CI95_high), colour = NA, alpha = 0.12, show.legend = FALSE) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 2.1, stroke = 0.55, fill = "white") +
  geom_hline(yintercept = 0, colour = "#B7BEC5", linewidth = 0.35) +
  group_scales +
  scale_x_continuous(breaks = 1:10, limits = c(0.8, 10.2)) +
  labs(
    title = "IFG Outcome: six-ROI composite HC/SZ curve",
    subtitle = stat_subtitle,
    x = "Fixed G1 bin",
    y = "Mean Fisher-z FC",
    caption = paste0(
      "Points are group means; ribbons are t-based 95% CIs across participants.\n",
      "Models adjust age, sex and MeanFD with participant-clustered HC1 covariance.\n",
      "BH family: two targets x three prespecified tests; IFG is the six-ROI composite."
    )
  ) +
  theme_ifg(7.4)

roi_plot <- ggplot(
  roi_curve_summary,
  aes(x = Bin, y = mean, colour = Group, fill = Group, shape = Group, linetype = Group, group = Group)
) +
  geom_ribbon(aes(ymin = CI95_low, ymax = CI95_high), colour = NA, alpha = 0.10, show.legend = FALSE) +
  geom_line(linewidth = 0.58) +
  geom_point(size = 1.45, stroke = 0.45, fill = "white") +
  geom_hline(yintercept = 0, colour = "#B7BEC5", linewidth = 0.3) +
  facet_wrap(~ ROI_label, ncol = 3, scales = "fixed") +
  group_scales +
  scale_x_continuous(breaks = c(1, 5, 10), limits = c(0.8, 10.2)) +
  labs(
    title = "IFG Outcome: six constituent ROI curves",
    subtitle = "Shared axes expose parcel-level heterogeneity.\nThese panels are descriptive and do not reuse BN29 inference as IFG-family inference.",
    x = "Fixed G1 bin",
    y = "Mean Fisher-z FC",
    caption = "Points are group means; ribbons are t-based 95% CIs across participants.\nn = 32/group for every ROI and bin."
  ) +
  theme_ifg(7.0) +
  theme(
    legend.position = "top",
    panel.spacing = grid::unit(7, "pt")
  )

save_plot_bundle <- function(plot, stem, width_mm, height_mm, dpi = 600) {
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4
  svg_path <- paste0(stem, ".svg")
  pdf_path <- paste0(stem, ".pdf")
  tiff_path <- paste0(stem, ".tiff")
  png_path <- paste0(stem, ".png")

  svglite::svglite(svg_path, width = width_in, height = height_in, bg = "white")
  print(plot)
  dev.off()

  grDevices::cairo_pdf(pdf_path, width = width_in, height = height_in, family = FONT, bg = "white")
  print(plot)
  dev.off()

  ragg::agg_tiff(tiff_path, width = width_in, height = height_in, units = "in", res = dpi, background = "white", compression = "lzw")
  print(plot)
  dev.off()

  ragg::agg_png(png_path, width = width_in, height = height_in, units = "in", res = 300, background = "white")
  print(plot)
  dev.off()

  outputs <- c(svg_path, pdf_path, tiff_path, png_path)
  assert_true(all(file.exists(outputs)), paste("Missing export for", basename(stem)))
  assert_true(all(file.info(outputs)$size > 1000), paste("Trivially small export for", basename(stem)))
  outputs
}

composite_files <- save_plot_bundle(
  composite_plot,
  file.path(out_dir, "figures", "IFG_Outcome_6ROI_composite_HC_SZ"),
  width_mm = 160,
  height_mm = 108
)
roi_files <- save_plot_bundle(
  roi_plot,
  file.path(out_dir, "figures", "IFG_Outcome_6ROI_individual_HC_SZ"),
  width_mm = 183,
  height_mm = 132
)

comparison_to_report <- comparison |>
  filter(test %in% c("slope_difference", "categorical_binned_shape"))
slope_comp <- comparison_to_report |> filter(test == "slope_difference")
shape_comp <- comparison_to_report |> filter(test == "categorical_binned_shape")

report_lines <- c(
  "# IFG Outcome Raw Fisher-z FC 六 ROI 重分析",
  "",
  "## 结论",
  "",
  "现有 B08 IFG 曲线和检验只使用 L_A44d / BN29 单个 ROI；它们不是六 ROI IFG Outcome subfamily 的整体结果。",
  "本次重分析先在每名被试、每个固定 G1 bin 内平均 BN29/31/33/35/37/39 的 Fisher-z FC，再绘图和拟合模型。BN29 的旧统计量仅保留为审计对照，没有作为 IFG 整体检验结果。",
  "",
  "## 六个 ROI 与 FC 数据",
  "",
  "六个 ROI 都来自同一 canonical long/wide FC 表；原始冻结 Fisher-z 指纹按被试存储为 2608 x 65 NPY，ROI 对应列见 tables/ifg_outcome_6roi_manifest.tsv。",
  "",
  paste0("- Canonical long: `", ifg_roi_manifest$canonical_long_file[1], "`"),
  paste0("- Canonical wide: `", ifg_roi_manifest$canonical_wide_file[1], "`"),
  paste0("- Frozen source pattern: `", ifg_roi_manifest$frozen_subject_fingerprint_files[1], "`"),
  "",
  paste0("- ", ifg_roi_manifest$atlas_label, " -> ", ifg_roi_manifest$ROI, " (NPY column ", ifg_roi_manifest$fingerprint_column_1based, " one-based)"),
  "",
  "## 统计方法",
  "",
  "- n 定义：64 名被试（HC 32，SZ 32）；每个被试每个 bin 的 IFG 值为六 ROI Fisher-z FC 的算术平均。",
  "- 连续模型：FC ~ Group * centered_bin + Age + Sex + MeanFD。",
  "- 分类模型：FC ~ Group * categorical_bin + Age + Sex + MeanFD；联合检验 9 个 Group x Bin 项。",
  "- 协方差：按被试聚类的 HC1 有限样本修正；t/F 分母自由度 63。",
  "- 多重比较：BH across 2 targets x 3 prespecified tests；SMA 仍为 BN10，IFG 替换为六 ROI composite；全部 p 与 q 在 R 中重新计算。",
  "",
  "## 关键结果：旧 BN29 与新六 ROI composite",
  "",
  sprintf(
    "- Linear slope：旧 BN29 beta=%+.6f, p=%.6g, q=%.6g；新六 ROI beta=%+.6f, 95%% CI [%+.6f, %+.6f], p=%.6g, q=%.6g。",
    slope_comp$old_beta, slope_comp$old_p_raw, slope_comp$old_q_BH_6_tests,
    slope_test$beta, slope_test$CI95_low, slope_test$CI95_high, slope_test$p_raw, slope_test$q_BH_6_tests
  ),
  sprintf(
    "- Binned shape：旧 BN29 F(9,63)=%.6f, p=%.6g, q=%.6g；新六 ROI F(9,63)=%.6f, p=%.6g, q=%.6g。",
    shape_comp$old_statistic, shape_comp$old_p_raw, shape_comp$old_q_BH_6_tests,
    shape_test$statistic, shape_test$p_raw, shape_test$q_BH_6_tests
  ),
  sprintf(
    "- Offset at Bin 5.5：新六 ROI beta=%+.6f, 95%% CI [%+.6f, %+.6f], p=%.6g, q=%.6g。",
    offset_test$beta, offset_test$CI95_low, offset_test$CI95_high, offset_test$p_raw, offset_test$q_BH_6_tests
  ),
  "",
  "## QA",
  "",
  "- 六 ROI 原始行数：3840 = 64 subjects x 10 bins x 6 ROIs；无重复 key、无缺失、无非有限值。",
  "- Composite 行数：640 = 64 subjects x 10 bins；每行 n_ROI=6。",
  "- R 中重算的原 BN29/SMA clustered-OLS 与 locked B08 beta、SE、统计量、p、q 的绝对差均 < 1e-10。",
  "- 两张图均由 R 导出 SVG/PDF/TIFF/PNG；共享坐标的六 ROI 分面图用于异质性检查。",
  "",
  "## 主要输出",
  "",
  "- `figures/IFG_Outcome_6ROI_composite_HC_SZ.*`",
  "- `figures/IFG_Outcome_6ROI_individual_HC_SZ.*`",
  "- `tables/ifg_6roi_recomputed_tests.tsv`",
  "- `tables/bn29_vs_ifg6_scope_comparison.tsv`",
  "- `source_data/ifg_6roi_subject_bin_composite.tsv`"
)
writeLines(report_lines, file.path(out_dir, "REPORT.md"), useBytes = TRUE)

qa <- tibble(
  check = c(
    "IFG raw rows", "IFG composite rows", "Six ROIs per subject-bin", "HC subjects", "SZ subjects",
    "Old B08 reproduction", "Composite SVG/PDF/TIFF/PNG", "Six-ROI SVG/PDF/TIFF/PNG"
  ),
  observed = c(
    nrow(ifg_raw), nrow(ifg_composite), paste(range(ifg_composite$n_ROI), collapse = "-"),
    n_distinct(ifg_raw$Subject[ifg_raw$Group == "HC"]), n_distinct(ifg_raw$Subject[ifg_raw$Group == "SZ"]),
    all(reproduction_check$pass), all(file.exists(composite_files)), all(file.exists(roi_files))
  ),
  pass = c(
    nrow(ifg_raw) == 3840L, nrow(ifg_composite) == 640L, all(ifg_composite$n_ROI == 6L),
    n_distinct(ifg_raw$Subject[ifg_raw$Group == "HC"]) == 32L,
    n_distinct(ifg_raw$Subject[ifg_raw$Group == "SZ"]) == 32L,
    all(reproduction_check$pass), all(file.exists(composite_files)), all(file.exists(roi_files))
  )
)
write_tsv_stable(qa, file.path(out_dir, "tables", "QA_checks.tsv"))
assert_true(all(qa$pass), "One or more QA checks failed.")

cat("IFG_6ROI_REANALYSIS=PASS\n")
cat("OUTPUT_DIR=", out_dir, "\n", sep = "")
cat("SLOPE_BETA=", format(slope_test$beta, digits = 12), "\n", sep = "")
cat("SLOPE_P=", format(slope_test$p_raw, digits = 12), "\n", sep = "")
cat("SLOPE_Q_BH6=", format(slope_test$q_BH_6_tests, digits = 12), "\n", sep = "")
cat("SHAPE_F=", format(shape_test$statistic, digits = 12), "\n", sep = "")
cat("SHAPE_P=", format(shape_test$p_raw, digits = 12), "\n", sep = "")
cat("SHAPE_Q_BH6=", format(shape_test$q_BH_6_tests, digits = 12), "\n", sep = "")
