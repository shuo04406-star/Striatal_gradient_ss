#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(tibble)
  library(stringr)
  library(patchwork)
  library(sandwich)
  library(ragg)
  library(scales)
})

set.seed(20261030)

# Figure contract
# Core conclusion: the screen was systematic; the most reliable signal is a negative
# TEPS-Anticipation similarity association within SZ, while direct HC/SZ interaction
# tests do not establish disease specificity.
# Archetype: quantitative grid, with Figure 2 as the hero figure.
# Backend/output: R only; PNG only.
# Evidence chain: census -> core association -> group comparison -> LOO robustness;
# supplementary figures retain all 70 interactions and the full 119-endpoint universe.
# Reviewer risks: correlated TEPS scores, small subgroup n, interaction power, and
# confusing "significant in one group" with a statistically significant group difference.

analysis_dir <- Sys.getenv("STRIATAL_FULL_SCALE_OUTPUT", unset = NA_character_)
tables_dir <- file.path(analysis_dir, "tables")
output_dir <- file.path(analysis_dir, "manuscript_figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

paths <- list(
  master = file.path(Sys.getenv("STRIATAL_FINAL64_LOCK", unset = NA_character_), "frozen_data", "subject_level_brain_behaviour_master_final64.tsv"),
  sample = file.path(Sys.getenv("STRIATAL_FINAL64_LOCK", unset = NA_character_), "analysis_preparation", "ANALYSIS_SAMPLE_MATRIX.tsv"),
  registry = file.path(tables_dir, "SCALE_ANALYSIS_REGISTRY.tsv"),
  original = file.path(tables_dir, "ELIGIBLE_SCALE_IMAGING_ASSOCIATIONS.tsv"),
  within = file.path(tables_dir, "ALL_SCALE_IMAGING_WITHIN_GROUP_AUDIT.tsv"),
  interactions = file.path(tables_dir, "ALL_SCALE_IMAGING_GROUP_INTERACTION_AUDIT.tsv"),
  evidence = file.path(tables_dir, "ALL_SCALE_IMAGING_EVIDENCE_CLASSIFICATION.tsv")
)
stopifnot(all(file.exists(unlist(paths))))

palette <- c(
  HC = "#4C78A8",
  SZ = "#D16D5A",
  teal = "#2A9D8F",
  blue = "#3B6FB6",
  red = "#C9574D",
  orange = "#D98E3D",
  purple = "#8A6FB0",
  gold = "#C49A45",
  gray = "#AFAFAF",
  light_gray = "#F1F1F1",
  dark = "#292929",
  mid = "#737373"
)

theme_manuscript <- function(base_size = 7, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.text = element_text(colour = "black", size = base_size - 0.4),
      axis.title = element_text(size = base_size),
      legend.title = element_text(size = base_size - 0.1),
      legend.text = element_text(size = base_size - 0.5),
      strip.text = element_text(size = base_size, face = "bold"),
      plot.title = element_text(size = base_size + 1.0, face = "bold"),
      plot.subtitle = element_text(size = base_size - 0.1, colour = palette[["mid"]]),
      plot.caption = element_text(size = base_size - 0.7, colour = palette[["mid"]], hjust = 0),
      plot.tag = element_text(size = base_size + 1.6, face = "bold"),
      panel.grid = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      legend.key = element_rect(fill = "white", colour = NA)
    )
}
theme_set(theme_manuscript())

save_png <- function(plot, filename, width_mm, height_mm, dpi = 400) {
  ragg::agg_png(
    filename = file.path(output_dir, filename),
    width = width_mm,
    height = height_mm,
    units = "mm",
    res = dpi,
    background = "white"
  )
  print(plot)
  grDevices::dev.off()
}

write_source <- function(x, filename) {
  readr::write_tsv(x, file.path(tables_dir, filename), na = "")
}

zscore <- function(x) {
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(NA_real_, length(x)))
  as.numeric((x - mean(x, na.rm = TRUE)) / s)
}

master <- readr::read_tsv(paths$master, show_col_types = FALSE, progress = FALSE)
sample_matrix <- readr::read_tsv(paths$sample, show_col_types = FALSE, progress = FALSE)
scale_registry <- readr::read_tsv(paths$registry, show_col_types = FALSE, progress = FALSE)
original <- readr::read_tsv(paths$original, show_col_types = FALSE, progress = FALSE)
within <- readr::read_tsv(paths$within, show_col_types = FALSE, progress = FALSE)
interactions <- readr::read_tsv(paths$interactions, show_col_types = FALSE, progress = FALSE)
evidence <- readr::read_tsv(paths$evidence, show_col_types = FALSE, progress = FALSE)

stopifnot(
  nrow(original) == 119,
  nrow(within) == 189,
  nrow(interactions) == 70,
  nrow(evidence) == 119
)

eligible_scales <- scale_registry %>% filter(imaging_eligible)
imaging_registry <- original %>%
  distinct(imaging_variable, imaging_label, imaging_order, stage) %>%
  arrange(imaging_order)

analysis_data <- sample_matrix %>%
  select(Subject_ID, Group, Age, Sex, MeanFD, all_of(imaging_registry$imaging_variable)) %>%
  left_join(
    master %>% select(Subject_ID, all_of(unique(eligible_scales$source_col))),
    by = "Subject_ID"
  )

core_endpoints <- tribble(
  ~scale_variable, ~panel_label, ~short_label, ~panel_order,
  "TEPS_consummatory", "TEPS consummatory", "TEPS consummatory", 1,
  "TEPS_total", "TEPS total", "TEPS total", 2
)

get_scale_source <- function(scale_variable) {
  out <- eligible_scales %>% filter(variable == scale_variable) %>% pull(source_col)
  stopifnot(length(out) == 1)
  out[[1]]
}

prepare_sz_adjusted_data <- function(scale_variable) {
  source_col <- get_scale_source(scale_variable)
  d <- analysis_data %>%
    filter(Group == "SZ") %>%
    transmute(
      Subject_ID,
      scale_value = suppressWarnings(as.numeric(.data[[source_col]])),
      imaging_value = suppressWarnings(as.numeric(Anticipation_similarity)),
      Age = suppressWarnings(as.numeric(Age)),
      Sex = droplevels(factor(Sex)),
      MeanFD = suppressWarnings(as.numeric(MeanFD))
    ) %>%
    filter(complete.cases(.)) %>%
    mutate(
      scale_z = zscore(scale_value),
      imaging_z = zscore(imaging_value),
      age_z = zscore(Age),
      fd_z = zscore(MeanFD)
    )
  stopifnot(nrow(d) == 28)
  cov_terms <- c("age_z", "fd_z")
  if (n_distinct(d$Sex) > 1) cov_terms <- c(cov_terms, "Sex")
  x_formula <- stats::as.formula(paste("scale_z ~", paste(cov_terms, collapse = " + ")))
  y_formula <- stats::as.formula(paste("imaging_z ~", paste(cov_terms, collapse = " + ")))
  d$scale_residual <- stats::residuals(stats::lm(x_formula, data = d))
  d$imaging_residual <- stats::residuals(stats::lm(y_formula, data = d))
  d
}

fit_adjusted_beta <- function(data, scale_variable) {
  source_col <- get_scale_source(scale_variable)
  d <- data %>%
    transmute(
      y = suppressWarnings(as.numeric(Anticipation_similarity)),
      x = suppressWarnings(as.numeric(.data[[source_col]])),
      Age = suppressWarnings(as.numeric(Age)),
      Sex = droplevels(factor(Sex)),
      MeanFD = suppressWarnings(as.numeric(MeanFD))
    ) %>%
    filter(complete.cases(.)) %>%
    mutate(y_z = zscore(y), x_z = zscore(x), age_z = zscore(Age), fd_z = zscore(MeanFD))
  terms <- c("x_z", "age_z", "fd_z")
  if (n_distinct(d$Sex) > 1) terms <- c(terms, "Sex")
  fit <- stats::lm(stats::as.formula(paste("y_z ~", paste(terms, collapse = " + "))), data = d)
  unname(stats::coef(fit)["x_z"])
}

# -----------------------------------------------------------------------------
# Figure 1 | Evidence distribution and complete 119-endpoint audit
# -----------------------------------------------------------------------------

figure_class_levels <- c(
  "Direct group-difference evidence",
  "Reliable within HC",
  "Reliable within SZ",
  "Combined-sample FDR only",
  "Prespecified-family support only",
  "Exploratory nominal signal",
  "No clear signal"
)

figure_class_palette <- c(
  "Direct group-difference evidence" = "#B33B3B",
  "Reliable within HC" = palette[["HC"]],
  "Reliable within SZ" = palette[["SZ"]],
  "Combined-sample FDR only" = palette[["purple"]],
  "Prespecified-family support only" = palette[["gold"]],
  "Exploratory nominal signal" = palette[["gray"]],
  "No clear signal" = palette[["light_gray"]]
)

figure1_data <- evidence %>%
  mutate(
    figure_class = case_when(
      primary_evidence_class == "GROUP_DIFFERENCE_EVIDENCE" ~ "Direct group-difference evidence",
      primary_evidence_class %in% c("RELIABLE_WITHIN_HC_NOT_GROUP_DIFFERENCE", "RELIABLE_WITHIN_BOTH_GROUPS") ~ "Reliable within HC",
      primary_evidence_class == "RELIABLE_WITHIN_SZ_NOT_GROUP_DIFFERENCE" ~ "Reliable within SZ",
      primary_evidence_class == "COMBINED_FDR_NO_WITHIN_GROUP_CONFIRMATION" ~ "Combined-sample FDR only",
      primary_evidence_class == "PRESPECIFIED_FAMILY_ONLY" ~ "Prespecified-family support only",
      primary_evidence_class == "EXPLORATORY_NOMINAL_SIGNAL" ~ "Exploratory nominal signal",
      TRUE ~ "No clear signal"
    ),
    figure_class = factor(figure_class, levels = figure_class_levels),
    imaging_label = factor(imaging_label, levels = imaging_registry$imaging_label),
    scale_label = factor(scale_label, levels = rev(eligible_scales$label))
  )

figure1_counts <- tibble(figure_class = factor(figure_class_levels, levels = figure_class_levels)) %>%
  left_join(figure1_data %>% count(figure_class, name = "n_endpoints"), by = "figure_class") %>%
  mutate(
    n_endpoints = coalesce(n_endpoints, 0L),
    plot_class = factor(as.character(figure_class), levels = rev(figure_class_levels)),
    label_colour = if_else(as.character(figure_class) == "Exploratory nominal signal", "white", "black")
  )

stopifnot(sum(figure1_counts$n_endpoints) == 119, figure1_counts$n_endpoints[figure1_counts$figure_class == "Reliable within SZ"] == 2)

p1a <- ggplot(figure1_counts, aes(x = n_endpoints, y = plot_class, fill = figure_class)) +
  geom_col(width = 0.68, colour = "white", linewidth = 0.25) +
  geom_text(aes(label = n_endpoints, colour = label_colour), hjust = if_else(figure1_counts$n_endpoints > 10, 1.15, -0.35), size = 2.7, fontface = "bold") +
  scale_fill_manual(values = figure_class_palette, drop = FALSE) +
  scale_colour_identity() +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Evidence census",
    subtitle = "All categories retained, including zero-count classes",
    x = "Number of endpoints",
    y = NULL
  ) +
  theme_manuscript(base_size = 7.3) +
  theme(legend.position = "none", axis.line.y = element_blank(), axis.ticks.y = element_blank())

p1b <- ggplot(figure1_data, aes(x = imaging_label, y = scale_label, fill = figure_class)) +
  geom_tile(linewidth = 0.28, colour = "white") +
  scale_fill_manual(values = figure_class_palette, drop = TRUE, name = "Evidence class") +
  labs(
    title = "Complete 119-endpoint audit",
    subtitle = "Every eligible scale x imaging endpoint is shown; no significance-based selection",
    x = NULL,
    y = NULL
  ) +
  theme_manuscript(base_size = 6.8) +
  theme(
    axis.text.x = element_text(angle = 42, hjust = 1, size = 5.4),
    axis.text.y = element_text(size = 5.2),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    legend.position = "bottom"
  ) +
  guides(fill = guide_legend(ncol = 2, byrow = TRUE))

figure1 <- (p1a | p1b) +
  plot_layout(widths = c(0.78, 1.55)) +
  plot_annotation(
    title = "Figure 1 | Evidence distribution across all scale-imaging associations",
    subtitle = "Systematic screening of 119 endpoints identifies two reliable within-SZ associations and no direct group-difference evidence.",
    tag_levels = "A",
    theme = theme_manuscript(base_size = 7.5)
  )

save_png(figure1, "Figure1_all_scale_imaging_evidence_distribution.png", 245, 165)
write_source(figure1_data, "MANUSCRIPT_FIGURE1_ENDPOINT_EVIDENCE.tsv")
write_source(figure1_counts, "MANUSCRIPT_FIGURE1_EVIDENCE_COUNTS.tsv")

# -----------------------------------------------------------------------------
# Figure 2 | Core SZ TEPS-Anticipation similarity associations
# -----------------------------------------------------------------------------

partial_points <- map_dfr(seq_len(nrow(core_endpoints)), function(i) {
  e <- core_endpoints[i, ]
  prepare_sz_adjusted_data(e$scale_variable) %>%
    mutate(
      scale_variable = e$scale_variable,
      panel = paste0(LETTERS[e$panel_order], " | ", e$panel_label)
    )
})

partial_bands <- map_dfr(seq_len(nrow(core_endpoints)), function(i) {
  e <- core_endpoints[i, ]
  d <- partial_points %>% filter(scale_variable == e$scale_variable)
  fit <- stats::lm(imaging_residual ~ scale_residual, data = d)
  vc <- sandwich::vcovHC(fit, type = "HC3")
  grid <- seq(min(d$scale_residual), max(d$scale_residual), length.out = 160)
  mm <- cbind(1, grid)
  pred <- as.numeric(mm %*% stats::coef(fit))
  pred_se <- sqrt(pmax(0, diag(mm %*% vc %*% t(mm))))
  tibble(
    scale_variable = e$scale_variable,
    panel = paste0(LETTERS[e$panel_order], " | ", e$panel_label),
    scale_residual = grid,
    predicted = pred,
    ci_low = pred - 1.96 * pred_se,
    ci_high = pred + 1.96 * pred_se
  )
})

figure2_stats <- within %>%
  filter(
    population == "SZ",
    scale_variable %in% core_endpoints$scale_variable,
    imaging_variable == "Anticipation_similarity"
  ) %>%
  left_join(core_endpoints, by = "scale_variable") %>%
  arrange(panel_order) %>%
  mutate(
    panel = paste0(LETTERS[panel_order], " | ", panel_label),
    annotation = sprintf("Negative association | beta = %.3f | 95%% CI [%.3f, %.3f] | BH q = %.4f | n = %d", effect, ci_low, ci_high, p_adj_bh_within_population, n),
    panel_stats = sprintf(
      "%s\nNegative association | beta = %.3f | BH q = %.4f | n = %d\n95%% CI [%.3f, %.3f]",
      panel, effect, p_adj_bh_within_population, n, ci_low, ci_high
    )
  )

partial_points <- partial_points %>%
  select(-panel) %>%
  left_join(figure2_stats %>% select(scale_variable, panel_stats), by = "scale_variable")

partial_bands <- partial_bands %>%
  select(-panel) %>%
  left_join(figure2_stats %>% select(scale_variable, panel_stats), by = "scale_variable")

p2 <- ggplot(partial_points, aes(x = scale_residual, y = imaging_residual)) +
  geom_ribbon(
    data = partial_bands,
    aes(x = scale_residual, ymin = ci_low, ymax = ci_high),
    inherit.aes = FALSE,
    fill = palette[["SZ"]], alpha = 0.16
  ) +
  geom_line(
    data = partial_bands,
    aes(x = scale_residual, y = predicted),
    inherit.aes = FALSE,
    colour = palette[["SZ"]], linewidth = 0.8
  ) +
  geom_point(shape = 21, size = 2.0, stroke = 0.35, fill = "white", colour = palette[["SZ"]], alpha = 0.92) +
  facet_wrap(~panel_stats, nrow = 1) +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = "#D8D8D8") +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = "#D8D8D8") +
  labs(
    title = "Figure 2 | Core TEPS associations with Anticipation similarity in SZ",
    subtitle = "One point per observed SZ participant; axes are Age-, Sex- and MeanFD-residualized z scores.\nLines and ribbons show the adjusted trend and HC3 95% CI.",
    x = "Covariate-residualized TEPS score (z)",
    y = "Covariate-residualized Anticipation similarity (z)"
  ) +
  theme_manuscript(base_size = 7.5) +
  theme(
    strip.background = element_rect(fill = "#FAF1EF", colour = NA),
    strip.text = element_text(hjust = 0, size = 6.6, lineheight = 1.15),
    panel.spacing = grid::unit(6, "mm")
  )

save_png(p2, "Figure2_core_TEPS_Anticipation_similarity.png", 190, 108)
write_source(partial_points %>% mutate(panel_stats = str_replace_all(panel_stats, "\\n", " | ")), "MANUSCRIPT_FIGURE2_PARTICIPANT_DATA.tsv")
write_source(partial_bands %>% mutate(panel_stats = str_replace_all(panel_stats, "\\n", " | ")), "MANUSCRIPT_FIGURE2_HC3_BANDS.tsv")
write_source(figure2_stats %>% mutate(panel_stats = str_replace_all(panel_stats, "\\n", " | ")), "MANUSCRIPT_FIGURE2_STATISTICS.tsv")

# -----------------------------------------------------------------------------
# Figure 3 | HC versus SZ effect estimates and direct interactions
# -----------------------------------------------------------------------------

figure3_data <- within %>%
  filter(
    population %in% c("HC", "SZ"),
    scale_variable %in% core_endpoints$scale_variable,
    imaging_variable == "Anticipation_similarity"
  ) %>%
  left_join(core_endpoints, by = "scale_variable") %>%
  left_join(
    interactions %>%
      filter(scale_variable %in% core_endpoints$scale_variable, imaging_variable == "Anticipation_similarity") %>%
      select(scale_variable, interaction_effect = effect, interaction_ci_low = ci_low, interaction_ci_high = ci_high, interaction_p = p),
    by = "scale_variable"
  ) %>%
  mutate(
    Group = factor(population, levels = c("SZ", "HC")),
    panel = sprintf("%s\nGroup x Scale p = %.3f", panel_label, interaction_p),
    estimate_label = sprintf("beta = %.2f  [%.2f, %.2f]", effect, ci_low, ci_high)
  )

p3 <- ggplot(figure3_data, aes(x = effect, y = Group, colour = Group)) +
  geom_vline(xintercept = 0, linewidth = 0.4, linetype = "dashed", colour = palette[["mid"]]) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = 0, linewidth = 0.7) +
  geom_point(size = 2.5) +
  geom_text(aes(x = 0.46, label = estimate_label), hjust = 0, colour = palette[["dark"]], size = 2.55) +
  facet_wrap(~panel, nrow = 1) +
  scale_colour_manual(values = palette[c("HC", "SZ")], drop = FALSE) +
  coord_cartesian(xlim = c(-0.82, 1.03), clip = "off") +
  labs(
    title = "Figure 3 | HC and SZ effects for TEPS-Anticipation similarity",
    subtitle = "Points and lines are adjusted standardized beta estimates and HC3 95% CIs. Interaction p values directly test whether slopes differ between groups.",
    x = "Adjusted standardized effect (beta)",
    y = NULL,
    colour = "Group"
  ) +
  theme_manuscript(base_size = 7.5) +
  theme(
    legend.position = "none",
    strip.background = element_blank(),
    strip.text = element_text(hjust = 0, size = 7.1),
    panel.spacing = grid::unit(8, "mm"),
    plot.margin = margin(5, 18, 5, 5, unit = "mm")
  )

save_png(p3, "Figure3_HC_SZ_TEPS_effect_comparison.png", 190, 82)
write_source(figure3_data, "MANUSCRIPT_FIGURE3_EFFECT_COMPARISON.tsv")

# -----------------------------------------------------------------------------
# Figure 4 | Leave-one-subject-out sensitivity
# -----------------------------------------------------------------------------

figure4_data <- map_dfr(seq_len(nrow(core_endpoints)), function(i) {
  e <- core_endpoints[i, ]
  source_col <- get_scale_source(e$scale_variable)
  complete_ids <- analysis_data %>%
    filter(Group == "SZ") %>%
    transmute(Subject_ID, x = .data[[source_col]], y = Anticipation_similarity, Age, Sex, MeanFD) %>%
    filter(complete.cases(.)) %>%
    arrange(Subject_ID) %>%
    pull(Subject_ID)
  full_beta <- fit_adjusted_beta(analysis_data %>% filter(Group == "SZ"), e$scale_variable)
  map_dfr(seq_along(complete_ids), function(j) {
    omitted <- complete_ids[[j]]
    beta <- fit_adjusted_beta(
      analysis_data %>% filter(Group == "SZ", Subject_ID != omitted),
      e$scale_variable
    )
    tibble(
      scale_variable = e$scale_variable,
      panel = paste0(LETTERS[e$panel_order], " | ", e$panel_label),
      omitted_index = j,
      omitted_subject = omitted,
      loo_beta = beta,
      full_beta = full_beta
    )
  })
}) %>%
  group_by(scale_variable) %>%
  mutate(direction_stability = mean(sign(loo_beta) == sign(first(full_beta)))) %>%
  ungroup()

figure4_annotations <- figure4_data %>%
  group_by(panel) %>%
  summarise(
    x = max(omitted_index),
    y = max(loo_beta) + 0.025,
    label = sprintf("Direction stability = %.0f%%", 100 * first(direction_stability)),
    .groups = "drop"
  )

p4 <- ggplot(figure4_data, aes(x = omitted_index, y = loo_beta)) +
  geom_hline(yintercept = 0, linewidth = 0.45, colour = palette[["mid"]]) +
  geom_hline(aes(yintercept = full_beta), linewidth = 0.65, linetype = "dashed", colour = palette[["SZ"]]) +
  geom_line(linewidth = 0.4, colour = palette[["SZ"]], alpha = 0.65) +
  geom_point(shape = 21, size = 1.75, stroke = 0.35, fill = "white", colour = palette[["SZ"]]) +
  geom_text(data = figure4_annotations, aes(x = x, y = y, label = label), inherit.aes = FALSE, hjust = 1, vjust = 0, size = 2.7, fontface = "bold", colour = palette[["SZ"]]) +
  facet_wrap(~panel, nrow = 1) +
  scale_x_continuous(breaks = c(1, 7, 14, 21, 28)) +
  coord_cartesian(ylim = c(-0.56, 0.05)) +
  labs(
    title = "Figure 4 | Leave-one-subject-out sensitivity of the two core associations",
    subtitle = "Each point is the adjusted beta after omitting one SZ participant. Dashed red lines show the full-sample estimate; the gray line marks zero effect.",
    x = "Omitted participant index",
    y = "Leave-one-out adjusted beta"
  ) +
  theme_manuscript(base_size = 7.5) +
  theme(strip.background = element_blank(), strip.text = element_text(hjust = 0, size = 7.2), panel.spacing = grid::unit(7, "mm"))

save_png(p4, "Figure4_leave_one_out_sensitivity.png", 190, 92)
write_source(figure4_data, "MANUSCRIPT_FIGURE4_LEAVE_ONE_OUT.tsv")

# -----------------------------------------------------------------------------
# Figure S1 | All 70 Group x Scale interactions
# -----------------------------------------------------------------------------

min_interaction_p <- min(interactions$p, na.rm = TRUE)
stopifnot(abs(min_interaction_p - 0.06365815601289761) < 1e-10)
interaction_x_min <- min(interactions$ci_low, na.rm = TRUE) - 0.08
interaction_label_x <- max(interactions$ci_high, na.rm = TRUE) + 0.08
interaction_x_max <- interaction_label_x + 0.34

figureS1_data <- interactions %>%
  left_join(imaging_registry %>% select(imaging_variable, stage), by = "imaging_variable") %>%
  mutate(
    scale_label = factor(scale_label, levels = eligible_scales$label[eligible_scales$population == "BOTH" & eligible_scales$imaging_eligible]),
    imaging_label = factor(imaging_label, levels = rev(imaging_registry$imaging_label)),
    p_label = sprintf("p = %.3f", p),
    minimum_p = p == min_interaction_p,
    point_colour = if_else(minimum_p, "Minimum p", stage)
  )

pS1 <- ggplot(figureS1_data, aes(x = effect, y = imaging_label)) +
  geom_vline(xintercept = 0, linewidth = 0.35, linetype = "dashed", colour = palette[["mid"]]) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high, colour = point_colour), orientation = "y", width = 0, linewidth = 0.48) +
  geom_point(aes(colour = point_colour), size = 1.65) +
  geom_text(aes(x = interaction_label_x, label = p_label, fontface = if_else(minimum_p, "bold", "plain"), colour = point_colour), hjust = 0, size = 2.05, show.legend = FALSE) +
  facet_wrap(~scale_label, ncol = 2) +
  scale_colour_manual(values = c("Anticipation" = palette[["teal"]], "Outcome" = palette[["orange"]], "Minimum p" = "#B33B3B"), name = "Imaging stage") +
  coord_cartesian(xlim = c(interaction_x_min, interaction_x_max), clip = "off") +
  labs(
    title = "Figure S1 | Complete audit of 70 Group x Scale interactions",
    subtitle = sprintf("Every interaction estimate, HC3 95%% CI and p value is shown. The smallest uncorrected p value is %.3f; none reaches p < 0.05.", min_interaction_p),
    x = "Interaction beta (SZ-HC difference in scale slope)",
    y = NULL
  ) +
  theme_manuscript(base_size = 6.7) +
  theme(
    legend.position = "top",
    strip.background = element_rect(fill = "#F5F5F5", colour = NA),
    strip.text = element_text(hjust = 0, size = 6.2),
    axis.text.y = element_text(size = 5.3),
    panel.spacing = grid::unit(3.5, "mm"),
    plot.margin = margin(5, 12, 5, 5, unit = "mm")
  )

save_png(pS1, "FigureS1_all_70_group_scale_interactions.png", 220, 245)
write_source(figureS1_data, "MANUSCRIPT_FIGURES1_ALL_INTERACTIONS.tsv")

# -----------------------------------------------------------------------------
# Figure S2 | Full matrices for combined sample, HC and SZ
# -----------------------------------------------------------------------------

endpoint_grid <- expand_grid(
  scale_variable = eligible_scales$variable,
  imaging_variable = imaging_registry$imaging_variable
) %>%
  left_join(eligible_scales %>% select(scale_variable = variable, scale_label = label, scale_order = order, scale_population = population), by = "scale_variable") %>%
  left_join(imaging_registry %>% select(imaging_variable, imaging_label, imaging_order), by = "imaging_variable")

combined_matrix <- endpoint_grid %>%
  left_join(
    original %>%
      filter(population == "HC+SZ") %>%
      select(scale_variable, imaging_variable, effect, p, strict_q = p_adj_bh_global),
    by = c("scale_variable", "imaging_variable")
  ) %>%
  mutate(panel = "Combined sample | adjusted beta", strict_significant = is.finite(strict_q) & strict_q < 0.05)

hc_matrix <- endpoint_grid %>%
  left_join(
    within %>%
      filter(population == "HC") %>%
      select(scale_variable, imaging_variable, effect, p, strict_q = p_adj_bh_within_population, strict_significant = reliable_within_group),
    by = c("scale_variable", "imaging_variable")
  ) %>%
  mutate(panel = "HC | adjusted beta", strict_significant = coalesce(strict_significant, FALSE))

sz_matrix <- endpoint_grid %>%
  left_join(
    within %>%
      filter(population == "SZ") %>%
      select(scale_variable, imaging_variable, effect, p, strict_q = p_adj_bh_within_population, strict_significant = reliable_within_group),
    by = c("scale_variable", "imaging_variable")
  ) %>%
  mutate(panel = "SZ | adjusted beta or Spearman rho", strict_significant = coalesce(strict_significant, FALSE))

figureS2_data <- bind_rows(combined_matrix, hc_matrix, sz_matrix) %>%
  mutate(
    panel = factor(panel, levels = c("Combined sample | adjusted beta", "HC | adjusted beta", "SZ | adjusted beta or Spearman rho")),
    scale_label = factor(scale_label, levels = rev(eligible_scales$label)),
    imaging_label = factor(imaging_label, levels = imaging_registry$imaging_label),
    cell_label = case_when(
      !is.finite(effect) ~ "NA",
      strict_significant ~ paste0(sprintf("%.2f", effect), "*"),
      TRUE ~ sprintf("%.2f", effect)
    )
  )

stopifnot(nrow(figureS2_data) == 357, sum(is.finite(figureS2_data$effect)) == 252)

pS2 <- ggplot(figureS2_data, aes(x = imaging_label, y = scale_label, fill = effect)) +
  geom_tile(linewidth = 0.24, colour = "white") +
  geom_text(aes(label = cell_label), size = 1.85, colour = "black") +
  facet_grid(panel ~ ., scales = "free_y", space = "free_y") +
  scale_fill_gradient2(
    low = palette[["blue"]], mid = "white", high = palette[["red"]],
    midpoint = 0, limits = c(-0.8, 0.8), oob = scales::squish,
    na.value = "#D8D8D8", name = "Effect"
  ) +
  labs(
    title = "Figure S2 | Complete scale-imaging association matrices",
    subtitle = "Every panel retains all 119 registered endpoints; NA marks structurally unavailable or non-estimable cells.\n* denotes strict evidence: combined global BH q<0.05, or group-specific BH q<0.05 plus CI/LOO/influence stability.",
    x = NULL,
    y = NULL
  ) +
  theme_manuscript(base_size = 6.5) +
  theme(
    axis.text.x = element_text(angle = 42, hjust = 1, size = 5.1),
    axis.text.y = element_text(size = 4.8),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    strip.text.y = element_text(angle = 90, size = 5.5),
    strip.background = element_blank(),
    panel.spacing.y = grid::unit(2.5, "mm"),
    legend.position = "right"
  )

save_png(pS2, "FigureS2_complete_scale_imaging_matrices.png", 235, 312)
write_source(figureS2_data, "MANUSCRIPT_FIGURES2_COMPLETE_MATRICES.tsv")

# -----------------------------------------------------------------------------
# Package index and automated QA
# -----------------------------------------------------------------------------

figure_index <- tribble(
  ~figure_id, ~filename, ~core_message,
  "Figure 1", "Figure1_all_scale_imaging_evidence_distribution.png", "All 119 endpoints were screened; two are reliable within SZ.",
  "Figure 2", "Figure2_core_TEPS_Anticipation_similarity.png", "TEPS consummatory and total scores are negatively associated with Anticipation similarity in SZ.",
  "Figure 3", "Figure3_HC_SZ_TEPS_effect_comparison.png", "SZ effects are stronger numerically, but direct interactions do not establish group differences.",
  "Figure 4", "Figure4_leave_one_out_sensitivity.png", "Both core SZ effects retain a negative direction after every leave-one-out refit.",
  "Figure S1", "FigureS1_all_70_group_scale_interactions.png", "None of 70 interactions reaches nominal p<0.05; minimum p=0.064.",
  "Figure S2", "FigureS2_complete_scale_imaging_matrices.png", "Combined, HC and SZ matrices retain the complete endpoint universe with strict markers."
)
write_source(figure_index, "MANUSCRIPT_FIGURE_INDEX.tsv")

output_files <- file.path(output_dir, figure_index$filename)
qa <- tibble(
  check = c(
    "six_requested_figures",
    "all_expected_png_exist",
    "output_folder_png_only",
    "all_png_gt20kb",
    "figure1_seven_classes_sum119",
    "figure1_sz_reliable_count2",
    "figure2_two_panels_n28_each",
    "figure2_both_negative",
    "figure3_four_group_estimates",
    "figure3_two_interaction_p_values",
    "figure4_56_loo_estimates",
    "figure4_direction_stability_100pct",
    "figureS1_all70_interactions",
    "figureS1_min_p_0064",
    "figureS2_357_grid_cells",
    "figureS2_252_estimable_effects"
  ),
  passed = c(
    nrow(figure_index) == 6,
    all(file.exists(output_files)),
    all(tolower(tools::file_ext(list.files(output_dir, full.names = TRUE))) == "png"),
    all(file.info(output_files)$size > 20000),
    nrow(figure1_counts) == 7 && sum(figure1_counts$n_endpoints) == 119,
    figure1_counts$n_endpoints[figure1_counts$figure_class == "Reliable within SZ"] == 2,
    length(table(partial_points$panel_stats)) == 2 && all(table(partial_points$panel_stats) == 28),
    all(figure2_stats$effect < 0),
    nrow(figure3_data) == 4,
    n_distinct(figure3_data$interaction_p) == 2,
    nrow(figure4_data) == 56,
    all(figure4_data$direction_stability == 1),
    nrow(figureS1_data) == 70,
    abs(min_interaction_p - 0.06365815601289761) < 1e-10,
    nrow(figureS2_data) == 357,
    sum(is.finite(figureS2_data$effect)) == 252
  )
)
write_source(qa, "MANUSCRIPT_FIGURE_QA.tsv")
if (!all(qa$passed)) stop("Manuscript figure QA failed: ", paste(qa$check[!qa$passed], collapse = ", "))

message("Requested manuscript figures completed successfully.")
