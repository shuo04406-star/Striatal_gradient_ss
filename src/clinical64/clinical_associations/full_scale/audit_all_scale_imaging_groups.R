#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(tibble)
  library(stringr)
  library(sandwich)
  library(ragg)
  library(scales)
})

set.seed(20261030)

# Figure contract
# Core conclusion: separate reproducible within-group associations, direct evidence
# for HC/SZ slope differences, and signals that remain exploratory after multiplicity
# and influence checks.
# Archetype: quantitative grid.
# Backend/output: R only; PNG only, as requested.
# Hero evidence: endpoint-level evidence classification across all 119 tests.
# Validation: complete within-group and interaction heatmaps plus LOO stability maps.
# Reviewer risks: small subgroup n, correlated endpoints/scales, low interaction power,
# and the invalid inference that "significant in one group only" proves a group difference.

analysis_dir <- Sys.getenv("STRIATAL_FULL_SCALE_OUTPUT", unset = NA_character_)
tables_dir <- file.path(analysis_dir, "tables")
figures_dir <- file.path(analysis_dir, "figures")
logs_dir <- file.path(analysis_dir, "logs")

input_paths <- list(
  master = file.path(Sys.getenv("STRIATAL_FINAL64_LOCK", unset = NA_character_), "frozen_data", "subject_level_brain_behaviour_master_final64.tsv"),
  matrix = file.path(Sys.getenv("STRIATAL_FINAL64_LOCK", unset = NA_character_), "analysis_preparation", "ANALYSIS_SAMPLE_MATRIX.tsv"),
  registry = file.path(tables_dir, "SCALE_ANALYSIS_REGISTRY.tsv"),
  associations = file.path(tables_dir, "ELIGIBLE_SCALE_IMAGING_ASSOCIATIONS.tsv")
)
stopifnot(all(file.exists(unlist(input_paths))))

palette_contract <- c(
  HC = "#4C78A8",
  SZ = "#D16D5A",
  negative = "#3B6FB6",
  positive = "#C9574D",
  neutral_dark = "#2B2B2B",
  neutral_mid = "#7A7A7A",
  neutral_light = "#D9D9D9"
)

theme_nature_contract <- function(base_size = 7, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.4, colour = "black"),
      legend.title = element_text(size = base_size - 0.2),
      legend.text = element_text(size = base_size - 0.5),
      strip.text = element_text(size = base_size - 0.1, face = "bold"),
      plot.title = element_text(size = base_size + 0.8, face = "bold"),
      plot.subtitle = element_text(size = base_size - 0.1, colour = palette_contract[["neutral_mid"]]),
      plot.caption = element_text(size = base_size - 1, colour = palette_contract[["neutral_mid"]], hjust = 0),
      panel.grid = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
}
theme_set(theme_nature_contract())

save_png <- function(plot, filename, width_mm, height_mm, dpi = 300) {
  ragg::agg_png(
    filename = file.path(figures_dir, filename),
    width = width_mm,
    height = height_mm,
    units = "mm",
    res = dpi,
    background = "white"
  )
  print(plot)
  grDevices::dev.off()
}

write_tsv_safe <- function(x, filename) {
  readr::write_tsv(x, file.path(tables_dir, filename), na = "")
}

zscore <- function(x) {
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(NA_real_, length(x)))
  as.numeric((x - mean(x, na.rm = TRUE)) / s)
}

ci_excludes_zero <- function(low, high) {
  is.finite(low) & is.finite(high) & (low > 0 | high < 0)
}

same_nonzero_sign <- function(a, b) {
  is.finite(a) & is.finite(b) & a != 0 & b != 0 & sign(a) == sign(b)
}

master <- readr::read_tsv(input_paths$master, show_col_types = FALSE, progress = FALSE)
sample_matrix <- readr::read_tsv(input_paths$matrix, show_col_types = FALSE, progress = FALSE)
scale_registry <- readr::read_tsv(input_paths$registry, show_col_types = FALSE, progress = FALSE)
original_associations <- readr::read_tsv(input_paths$associations, show_col_types = FALSE, progress = FALSE)

stopifnot(
  nrow(sample_matrix) == 64,
  n_distinct(sample_matrix$Subject_ID) == 64,
  all(table(sample_matrix$Group) == 32),
  nrow(original_associations) == 119
)

eligible_scales <- scale_registry %>% filter(imaging_eligible)
both_scales <- eligible_scales %>% filter(population == "BOTH")
sz_only_scales <- eligible_scales %>% filter(population == "SZ_ONLY")

imaging_registry <- original_associations %>%
  distinct(imaging_variable, imaging_label, imaging_order, stage, metric_family, analysis_role) %>%
  arrange(imaging_order)

analysis_data <- sample_matrix %>%
  select(
    Subject_ID, Group, Age, Sex, MeanFD,
    all_of(imaging_registry$imaging_variable)
  ) %>%
  left_join(
    master %>% select(Subject_ID, all_of(unique(eligible_scales$source_col))),
    by = "Subject_ID"
  )

prepare_model_data <- function(data, x, y, group = NULL) {
  d <- data %>%
    transmute(
      Subject_ID,
      x = suppressWarnings(as.numeric(.data[[x]])),
      y = suppressWarnings(as.numeric(.data[[y]])),
      Group = factor(Group, levels = c("HC", "SZ")),
      Age = suppressWarnings(as.numeric(Age)),
      Sex = factor(Sex),
      MeanFD = suppressWarnings(as.numeric(MeanFD))
    )
  if (!is.null(group)) d <- d %>% filter(Group == group)
  d <- d %>% filter(complete.cases(.))
  if (nrow(d) == 0) return(d)
  d %>%
    mutate(
      Group = droplevels(Group),
      Sex = droplevels(Sex),
      x_z = zscore(x),
      y_z = zscore(y),
      age_z = zscore(Age),
      fd_z = zscore(MeanFD)
    ) %>%
    filter(complete.cases(x_z, y_z, age_z, fd_z))
}

model_formula <- function(d, interaction = FALSE) {
  terms <- if (interaction) "x_z * Group" else "x_z"
  covariates <- c("age_z", "fd_z")
  if (n_distinct(d$Sex) > 1) covariates <- c(covariates, "Sex")
  stats::as.formula(paste("y_z ~", paste(c(terms, covariates), collapse = " + ")))
}

fit_adjusted_hc3 <- function(data, x, y, group = NULL, interaction = FALSE, influence = TRUE) {
  d <- prepare_model_data(data, x, y, group)
  min_n <- if (interaction) 24 else 12
  if (
    nrow(d) < min_n || n_distinct(d$x) < 3 || n_distinct(d$y) < 3 ||
      (interaction && n_distinct(d$Group) < 2)
  ) return(NULL)

  fit <- tryCatch(stats::lm(model_formula(d, interaction), data = d), error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  vc <- tryCatch(sandwich::vcovHC(fit, type = "HC3"), error = function(e) NULL)
  if (is.null(vc) || !("x_z" %in% names(stats::coef(fit)))) return(NULL)

  coef_names <- names(stats::coef(fit))
  if (interaction) {
    int_term <- intersect(c("x_z:GroupSZ", "GroupSZ:x_z"), coef_names)
    if (length(int_term) != 1) return(NULL)
    term <- int_term[[1]]
  } else {
    term <- "x_z"
  }

  estimate <- unname(stats::coef(fit)[term])
  se <- sqrt(vc[term, term])
  p <- 2 * stats::pnorm(abs(estimate / se), lower.tail = FALSE)
  summary <- tibble(
    n = nrow(d),
    estimate = estimate,
    se = se,
    ci_low = estimate - 1.96 * se,
    ci_high = estimate + 1.96 * se,
    p = p
  )

  diagnostics <- tibble()
  if (influence) {
    cooks <- tryCatch(stats::cooks.distance(fit), error = function(e) rep(NA_real_, nrow(d)))
    leverage <- tryCatch(stats::hatvalues(fit), error = function(e) rep(NA_real_, nrow(d)))
    studentized <- tryCatch(stats::rstudent(fit), error = function(e) rep(NA_real_, nrow(d)))
    dfb <- tryCatch(stats::dfbetas(fit), error = function(e) NULL)
    dfb_term <- if (!is.null(dfb) && term %in% colnames(dfb)) dfb[, term] else rep(NA_real_, nrow(d))
    p_model <- length(stats::coef(fit))
    n_model <- nrow(d)
    diagnostics <- tibble(
      Subject_ID = d$Subject_ID,
      cooks_distance = as.numeric(cooks),
      leverage = as.numeric(leverage),
      studentized_residual = as.numeric(studentized),
      dfbeta_target = as.numeric(dfb_term),
      flag_cook = is.finite(cooks) & cooks > 4 / n_model,
      flag_leverage = is.finite(leverage) & leverage > 2 * p_model / n_model,
      flag_studentized = is.finite(studentized) & abs(studentized) > 3,
      flag_dfbeta = is.finite(dfb_term) & abs(dfb_term) > 2 / sqrt(n_model)
    ) %>%
      mutate(any_influence_flag = flag_cook | flag_leverage | flag_studentized | flag_dfbeta)
  }

  list(summary = summary, data = d, diagnostics = diagnostics, term = term)
}

summarise_adjusted_endpoint <- function(data, x, y, group = NULL, interaction = FALSE) {
  full <- fit_adjusted_hc3(data, x, y, group, interaction, influence = TRUE)
  if (is.null(full)) {
    return(tibble(
      n = 0L, effect = NA_real_, se = NA_real_, ci_low = NA_real_, ci_high = NA_real_, p = NA_real_,
      n_influence_flags = NA_integer_, flagged_excluded_effect = NA_real_,
      loo_min = NA_real_, loo_max = NA_real_, loo_sign_stability = NA_real_, max_abs_loo_delta = NA_real_
    ))
  }

  flagged_ids <- full$diagnostics %>% filter(any_influence_flag) %>% pull(Subject_ID)
  sensitivity <- if (length(flagged_ids) > 0) {
    fit_adjusted_hc3(
      data %>% filter(!Subject_ID %in% flagged_ids),
      x, y, group, interaction, influence = FALSE
    )
  } else {
    full
  }
  sensitivity_effect <- if (is.null(sensitivity)) NA_real_ else sensitivity$summary$estimate[[1]]

  loo <- map_dbl(full$data$Subject_ID, function(id) {
    res <- fit_adjusted_hc3(
      data %>% filter(Subject_ID != id),
      x, y, group, interaction, influence = FALSE
    )
    if (is.null(res)) NA_real_ else res$summary$estimate[[1]]
  })
  loo <- loo[is.finite(loo)]
  full_effect <- full$summary$estimate[[1]]

  tibble(
    n = full$summary$n[[1]],
    effect = full_effect,
    se = full$summary$se[[1]],
    ci_low = full$summary$ci_low[[1]],
    ci_high = full$summary$ci_high[[1]],
    p = full$summary$p[[1]],
    n_influence_flags = length(flagged_ids),
    flagged_excluded_effect = sensitivity_effect,
    loo_min = if (length(loo) > 0) min(loo) else NA_real_,
    loo_max = if (length(loo) > 0) max(loo) else NA_real_,
    loo_sign_stability = if (length(loo) > 0 && is.finite(full_effect) && full_effect != 0) mean(sign(loo) == sign(full_effect)) else NA_real_,
    max_abs_loo_delta = if (length(loo) > 0) max(abs(loo - full_effect)) else NA_real_
  )
}

summarise_spearman_endpoint <- function(data, x, y, original_row) {
  d <- data %>%
    filter(Group == "SZ") %>%
    transmute(
      Subject_ID,
      x = suppressWarnings(as.numeric(.data[[x]])),
      y = suppressWarnings(as.numeric(.data[[y]]))
    ) %>%
    filter(complete.cases(.))
  full_effect <- original_row$effect[[1]]
  loo <- map_dbl(seq_len(nrow(d)), function(i) {
    suppressWarnings(stats::cor(d$x[-i], d$y[-i], method = "spearman"))
  })
  loo <- loo[is.finite(loo)]
  tibble(
    n = original_row$n[[1]],
    effect = full_effect,
    se = original_row$se[[1]],
    ci_low = original_row$ci_low[[1]],
    ci_high = original_row$ci_high[[1]],
    p = original_row$p[[1]],
    n_influence_flags = NA_integer_,
    flagged_excluded_effect = NA_real_,
    loo_min = if (length(loo) > 0) min(loo) else NA_real_,
    loo_max = if (length(loo) > 0) max(loo) else NA_real_,
    loo_sign_stability = if (length(loo) > 0 && is.finite(full_effect) && full_effect != 0) mean(sign(loo) == sign(full_effect)) else NA_real_,
    max_abs_loo_delta = if (length(loo) > 0) max(abs(loo - full_effect)) else NA_real_
  )
}

# -----------------------------------------------------------------------------
# Complete within-group audit
# -----------------------------------------------------------------------------

within_both <- pmap_dfr(
  expand_grid(
    scale_variable = both_scales$variable,
    imaging_variable = imaging_registry$imaging_variable,
    population = c("HC", "SZ")
  ),
  function(scale_variable, imaging_variable, population) {
    scale_row <- both_scales %>% filter(variable == scale_variable)
    image_row <- imaging_registry %>% filter(imaging_variable == !!imaging_variable)
    result <- summarise_adjusted_endpoint(
      analysis_data, scale_row$source_col[[1]], imaging_variable,
      group = population, interaction = FALSE
    )
    bind_cols(
      tibble(
        scale_variable = scale_variable,
        scale_label = scale_row$label[[1]],
        scale_order = scale_row$order[[1]],
        scale_population = "BOTH",
        scale_status = scale_row$status[[1]],
        population = population,
        imaging_variable = imaging_variable,
        imaging_label = image_row$imaging_label[[1]],
        imaging_order = image_row$imaging_order[[1]],
        method = "WITHIN_GROUP_STANDARDIZED_OLS_HC3_ADJUSTED_AGE_SEX_MEANFD",
        effect_type = "standardized_beta"
      ),
      result
    )
  }
)

within_sz_only <- pmap_dfr(
  expand_grid(
    scale_variable = sz_only_scales$variable,
    imaging_variable = imaging_registry$imaging_variable
  ),
  function(scale_variable, imaging_variable) {
    scale_row <- sz_only_scales %>% filter(variable == scale_variable)
    image_row <- imaging_registry %>% filter(imaging_variable == !!imaging_variable)
    original_row <- original_associations %>%
      filter(scale_variable == !!scale_variable, imaging_variable == !!imaging_variable)
    result <- summarise_spearman_endpoint(
      analysis_data, scale_row$source_col[[1]], imaging_variable, original_row
    )
    bind_cols(
      tibble(
        scale_variable = scale_variable,
        scale_label = scale_row$label[[1]],
        scale_order = scale_row$order[[1]],
        scale_population = "SZ_ONLY",
        scale_status = scale_row$status[[1]],
        population = "SZ",
        imaging_variable = imaging_variable,
        imaging_label = image_row$imaging_label[[1]],
        imaging_order = image_row$imaging_order[[1]],
        method = "SZ_ONLY_SPEARMAN_WITH_SUBJECT_BOOTSTRAP_CI",
        effect_type = "spearman_rho"
      ),
      result
    )
  }
)

within_group_audit <- bind_rows(within_both, within_sz_only) %>%
  group_by(population) %>%
  mutate(p_adj_bh_within_population = p.adjust(p, method = "BH")) %>%
  ungroup() %>%
  mutate(
    ci_excludes_zero = ci_excludes_zero(ci_low, ci_high),
    influence_direction_stable = if_else(
      method == "SZ_ONLY_SPEARMAN_WITH_SUBJECT_BOOTSTRAP_CI",
      TRUE,
      same_nonzero_sign(effect, flagged_excluded_effect)
    ),
    loo_direction_fully_stable = is.finite(loo_sign_stability) & loo_sign_stability == 1,
    reliable_within_group = is.finite(p_adj_bh_within_population) &
      p_adj_bh_within_population < 0.05 & ci_excludes_zero &
      loo_direction_fully_stable & influence_direction_stable,
    nominal_within_group = is.finite(p) & p < 0.05 & ci_excludes_zero
  ) %>%
  arrange(population, scale_order, imaging_order)

stopifnot(nrow(within_group_audit) == 189)
write_tsv_safe(within_group_audit, "ALL_SCALE_IMAGING_WITHIN_GROUP_AUDIT.tsv")

# -----------------------------------------------------------------------------
# Direct Group x scale interaction audit for every BOTH-population scale
# -----------------------------------------------------------------------------

interaction_audit <- pmap_dfr(
  expand_grid(
    scale_variable = both_scales$variable,
    imaging_variable = imaging_registry$imaging_variable
  ),
  function(scale_variable, imaging_variable) {
    scale_row <- both_scales %>% filter(variable == scale_variable)
    image_row <- imaging_registry %>% filter(imaging_variable == !!imaging_variable)
    result <- summarise_adjusted_endpoint(
      analysis_data, scale_row$source_col[[1]], imaging_variable,
      group = NULL, interaction = TRUE
    )
    bind_cols(
      tibble(
        scale_variable = scale_variable,
        scale_label = scale_row$label[[1]],
        scale_order = scale_row$order[[1]],
        imaging_variable = imaging_variable,
        imaging_label = image_row$imaging_label[[1]],
        imaging_order = image_row$imaging_order[[1]],
        method = "STANDARDIZED_OLS_HC3_GROUP_X_SCALE_ADJUSTED_AGE_SEX_MEANFD",
        effect_type = "interaction_beta_SZ_minus_HC"
      ),
      result
    )
  }
) %>%
  mutate(
    p_adj_bh_70_interactions = p.adjust(p, method = "BH"),
    ci_excludes_zero = ci_excludes_zero(ci_low, ci_high),
    influence_direction_stable = same_nonzero_sign(effect, flagged_excluded_effect),
    loo_direction_fully_stable = is.finite(loo_sign_stability) & loo_sign_stability == 1,
    group_difference_evidence = is.finite(p_adj_bh_70_interactions) &
      p_adj_bh_70_interactions < 0.05 & ci_excludes_zero &
      loo_direction_fully_stable & influence_direction_stable,
    nominal_group_interaction = is.finite(p) & p < 0.05 & ci_excludes_zero
  ) %>%
  arrange(scale_order, imaging_order)

stopifnot(nrow(interaction_audit) == 70)
write_tsv_safe(interaction_audit, "ALL_SCALE_IMAGING_GROUP_INTERACTION_AUDIT.tsv")

# -----------------------------------------------------------------------------
# Endpoint-level evidence classification
# -----------------------------------------------------------------------------

hc_audit <- within_group_audit %>%
  filter(population == "HC") %>%
  select(
    scale_variable, imaging_variable,
    hc_n = n, hc_effect = effect, hc_ci_low = ci_low, hc_ci_high = ci_high,
    hc_p = p, hc_q_bh = p_adj_bh_within_population,
    hc_loo_sign_stability = loo_sign_stability,
    reliable_within_hc = reliable_within_group,
    nominal_within_hc = nominal_within_group
  )

sz_audit <- within_group_audit %>%
  filter(population == "SZ") %>%
  select(
    scale_variable, imaging_variable,
    sz_n = n, sz_effect = effect, sz_ci_low = ci_low, sz_ci_high = ci_high,
    sz_p = p, sz_q_bh = p_adj_bh_within_population,
    sz_loo_sign_stability = loo_sign_stability,
    reliable_within_sz = reliable_within_group,
    nominal_within_sz = nominal_within_group
  )

interaction_join <- interaction_audit %>%
  select(
    scale_variable, imaging_variable,
    interaction_n = n, interaction_effect = effect,
    interaction_ci_low = ci_low, interaction_ci_high = ci_high,
    interaction_p = p, interaction_q_bh = p_adj_bh_70_interactions,
    interaction_loo_sign_stability = loo_sign_stability,
    group_difference_evidence, nominal_group_interaction
  )

evidence_classification <- original_associations %>%
  select(
    scale_variable, scale_label, scale_status, scale_order, population,
    imaging_variable, imaging_label, imaging_order, stage, metric_family, analysis_role,
    combined_n = n, combined_effect = effect, combined_ci_low = ci_low, combined_ci_high = ci_high,
    combined_p = p, combined_q_global = p_adj_bh_global,
    prespecified_family, prespecified_holm_p = p_adj_prespecified
  ) %>%
  left_join(hc_audit, by = c("scale_variable", "imaging_variable")) %>%
  left_join(sz_audit, by = c("scale_variable", "imaging_variable")) %>%
  left_join(interaction_join, by = c("scale_variable", "imaging_variable")) %>%
  mutate(
    reliable_within_hc = coalesce(reliable_within_hc, FALSE),
    reliable_within_sz = coalesce(reliable_within_sz, FALSE),
    group_difference_evidence = coalesce(group_difference_evidence, FALSE),
    nominal_within_hc = coalesce(nominal_within_hc, FALSE),
    nominal_within_sz = coalesce(nominal_within_sz, FALSE),
    nominal_group_interaction = coalesce(nominal_group_interaction, FALSE),
    combined_global_fdr = is.finite(combined_q_global) & combined_q_global < 0.05,
    prespecified_family_support = is.finite(prespecified_holm_p) & prespecified_holm_p < 0.05,
    nominal_combined = is.finite(combined_p) & combined_p < 0.05,
    same_direction_hc_sz = if_else(
      is.finite(hc_effect) & is.finite(sz_effect) & hc_effect != 0 & sz_effect != 0,
      sign(hc_effect) == sign(sz_effect),
      NA
    ),
    any_nominal_signal = nominal_combined | nominal_within_hc | nominal_within_sz | nominal_group_interaction,
    primary_evidence_class = case_when(
      group_difference_evidence ~ "GROUP_DIFFERENCE_EVIDENCE",
      reliable_within_hc & reliable_within_sz ~ "RELIABLE_WITHIN_BOTH_GROUPS",
      reliable_within_hc ~ "RELIABLE_WITHIN_HC_NOT_GROUP_DIFFERENCE",
      reliable_within_sz ~ "RELIABLE_WITHIN_SZ_NOT_GROUP_DIFFERENCE",
      combined_global_fdr ~ "COMBINED_FDR_NO_WITHIN_GROUP_CONFIRMATION",
      prespecified_family_support ~ "PRESPECIFIED_FAMILY_ONLY",
      any_nominal_signal ~ "EXPLORATORY_NOMINAL_SIGNAL",
      TRUE ~ "NO_CLEAR_SIGNAL"
    ),
    reporting_interpretation = case_when(
      group_difference_evidence ~ "Direct multiplicity-controlled evidence that the scale-imaging slope differs between HC and SZ.",
      reliable_within_hc & reliable_within_sz ~ "Multiplicity-controlled, influence-stable association within both groups; not evidence of a group difference unless the interaction is supported.",
      reliable_within_hc ~ "Multiplicity-controlled, influence-stable HC association; absence of an SZ finding does not itself establish a group difference.",
      reliable_within_sz ~ "Multiplicity-controlled, influence-stable SZ association; absence of an HC finding does not itself establish a group difference.",
      combined_global_fdr ~ "Global-FDR combined-sample association without multiplicity-controlled confirmation within either group.",
      prespecified_family_support ~ "Supported only inside a prespecified small family; not supported by the global or within-group audit.",
      any_nominal_signal ~ "Exploratory nominal signal only; use for hypothesis generation, not a confirmatory claim.",
      TRUE ~ "No clear association under the current audit."
    )
  ) %>%
  arrange(scale_order, imaging_order)

stopifnot(nrow(evidence_classification) == 119)
write_tsv_safe(evidence_classification, "ALL_SCALE_IMAGING_EVIDENCE_CLASSIFICATION.tsv")

audit_summary <- bind_rows(
  evidence_classification %>% count(primary_evidence_class, name = "n_endpoints") %>% rename(category = primary_evidence_class),
  tibble(category = "TOTAL_ENDPOINTS", n_endpoints = nrow(evidence_classification)),
  tibble(category = "RELIABLE_WITHIN_HC_FLAG", n_endpoints = sum(evidence_classification$reliable_within_hc)),
  tibble(category = "RELIABLE_WITHIN_SZ_FLAG", n_endpoints = sum(evidence_classification$reliable_within_sz)),
  tibble(category = "GROUP_DIFFERENCE_EVIDENCE_FLAG", n_endpoints = sum(evidence_classification$group_difference_evidence)),
  tibble(category = "NOMINAL_GROUP_INTERACTIONS", n_endpoints = sum(evidence_classification$nominal_group_interaction))
)
write_tsv_safe(audit_summary, "ALL_SCALE_IMAGING_AUDIT_SUMMARY.tsv")

key_findings <- evidence_classification %>%
  filter(primary_evidence_class != "NO_CLEAR_SIGNAL")
write_tsv_safe(key_findings, "ALL_SCALE_IMAGING_KEY_FINDINGS.tsv")

# -----------------------------------------------------------------------------
# Complete PNG figure set for the audit
# -----------------------------------------------------------------------------

within_plot_data <- within_group_audit %>%
  mutate(
    panel = case_when(
      population == "HC" ~ "HC adjusted beta (70-test BH)",
      population == "SZ" & scale_population == "BOTH" ~ "SZ pleasure beta (119-test BH)",
      TRUE ~ "SZ clinical/motor rho (119-test BH)"
    ),
    panel = factor(panel, levels = c(
      "HC adjusted beta (70-test BH)",
      "SZ pleasure beta (119-test BH)",
      "SZ clinical/motor rho (119-test BH)"
    )),
    cell_mark = case_when(
      reliable_within_group ~ "*",
      nominal_within_group ~ "+",
      TRUE ~ ""
    ),
    cell_label = if_else(is.finite(effect), paste0(sprintf("%.2f", effect), cell_mark), "NA"),
    imaging_label = factor(imaging_label, levels = imaging_registry$imaging_label),
    scale_label = factor(scale_label, levels = rev(eligible_scales$label[eligible_scales$imaging_eligible]))
  )

p_within <- ggplot(within_plot_data, aes(x = imaging_label, y = scale_label, fill = effect)) +
  geom_tile(linewidth = 0.25, colour = "white") +
  geom_text(aes(label = cell_label), size = 2.05, colour = "black") +
  facet_grid(panel ~ ., scales = "free_y", space = "free_y") +
  scale_fill_gradient2(
    low = palette_contract[["negative"]], mid = "white", high = palette_contract[["positive"]],
    midpoint = 0, limits = c(-0.8, 0.8), oob = scales::squish, name = "Effect"
  ) +
  labs(
    title = "Fig. 10a | Complete within-group scale-imaging audit",
    subtitle = "* = BH q<0.05, CI excludes 0, 100% LOO sign stability and influence-stable direction; + = nominal p<0.05 only.",
    x = NULL, y = NULL
  ) +
  theme_nature_contract(base_size = 6.6) +
  theme(
    axis.text.x = element_text(angle = 42, hjust = 1, size = 5.4),
    axis.text.y = element_text(size = 5.2),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    strip.text.y = element_text(angle = 90, size = 5.8),
    strip.background = element_blank(),
    panel.spacing.y = grid::unit(2.5, "mm"),
    legend.position = "right"
  )
save_png(p_within, "Fig10A_all_within_group_associations.png", width_mm = 235, height_mm = 255)

interaction_plot_data <- interaction_audit %>%
  mutate(
    cell_mark = case_when(
      group_difference_evidence ~ "*",
      nominal_group_interaction ~ "+",
      TRUE ~ ""
    ),
    cell_label = if_else(is.finite(effect), paste0(sprintf("%.2f", effect), cell_mark), "NA"),
    imaging_label = factor(imaging_label, levels = imaging_registry$imaging_label),
    scale_label = factor(scale_label, levels = rev(both_scales$label))
  )

p_interactions <- ggplot(interaction_plot_data, aes(x = imaging_label, y = scale_label, fill = effect)) +
  geom_tile(linewidth = 0.25, colour = "white") +
  geom_text(aes(label = cell_label), size = 2.2, colour = "black") +
  scale_fill_gradient2(
    low = palette_contract[["negative"]], mid = "white", high = palette_contract[["positive"]],
    midpoint = 0, limits = c(-0.8, 0.8), oob = scales::squish, name = "SZ-HC\ninteraction beta"
  ) +
  labs(
    title = "Fig. 10b | Direct Group x scale interaction audit",
    subtitle = "All 70 HC/SZ interaction tests are shown. * = 70-test BH q<0.05 plus CI/LOO/influence stability; + = nominal p<0.05 only.",
    x = NULL, y = NULL
  ) +
  theme_nature_contract(base_size = 6.8) +
  theme(
    axis.text.x = element_text(angle = 42, hjust = 1, size = 5.5),
    axis.text.y = element_text(size = 5.5),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    legend.position = "right"
  )
save_png(p_interactions, "Fig10B_all_group_interactions.png", width_mm = 220, height_mm = 145)

class_levels <- c(
  "GROUP_DIFFERENCE_EVIDENCE",
  "RELIABLE_WITHIN_BOTH_GROUPS",
  "RELIABLE_WITHIN_HC_NOT_GROUP_DIFFERENCE",
  "RELIABLE_WITHIN_SZ_NOT_GROUP_DIFFERENCE",
  "COMBINED_FDR_NO_WITHIN_GROUP_CONFIRMATION",
  "PRESPECIFIED_FAMILY_ONLY",
  "EXPLORATORY_NOMINAL_SIGNAL",
  "NO_CLEAR_SIGNAL"
)
class_labels <- c(
  "Direct group-difference evidence",
  "Reliable within both groups",
  "Reliable within HC; no group-difference evidence",
  "Reliable within SZ; no group-difference evidence",
  "Combined-sample FDR only",
  "Prespecified-family support only",
  "Exploratory nominal signal",
  "No clear signal"
)
class_palette <- c(
  "GROUP_DIFFERENCE_EVIDENCE" = "#B33B3B",
  "RELIABLE_WITHIN_BOTH_GROUPS" = "#2A9D8F",
  "RELIABLE_WITHIN_HC_NOT_GROUP_DIFFERENCE" = "#4C78A8",
  "RELIABLE_WITHIN_SZ_NOT_GROUP_DIFFERENCE" = "#D98E3D",
  "COMBINED_FDR_NO_WITHIN_GROUP_CONFIRMATION" = "#8A6FB0",
  "PRESPECIFIED_FAMILY_ONLY" = "#C49A45",
  "EXPLORATORY_NOMINAL_SIGNAL" = "#B7B7B7",
  "NO_CLEAR_SIGNAL" = "#F4F4F4"
)

classification_plot_data <- evidence_classification %>%
  left_join(eligible_scales %>% select(variable, scale_population = population, domain), by = c("scale_variable" = "variable")) %>%
  mutate(
    panel = if_else(scale_population == "BOTH", "HC+SZ pleasure scales", "SZ clinical/motor scales"),
    panel = factor(panel, levels = c("HC+SZ pleasure scales", "SZ clinical/motor scales")),
    primary_evidence_class = factor(primary_evidence_class, levels = class_levels, labels = class_labels),
    imaging_label = factor(imaging_label, levels = imaging_registry$imaging_label),
    scale_label = factor(scale_label, levels = rev(eligible_scales$label[eligible_scales$imaging_eligible]))
  )

p_class <- ggplot(classification_plot_data, aes(x = imaging_label, y = scale_label, fill = primary_evidence_class)) +
  geom_tile(linewidth = 0.28, colour = "white") +
  facet_grid(panel ~ ., scales = "free_y", space = "free_y") +
  scale_fill_manual(values = setNames(class_palette, class_labels), drop = TRUE, name = "Primary evidence class") +
  labs(
    title = "Fig. 10c | Evidence classification for all 119 scale-imaging endpoints",
    subtitle = "The hierarchy prioritizes direct group differences, then reliable within-group evidence, combined FDR, prespecified support and nominal exploration.",
    x = NULL, y = NULL
  ) +
  theme_nature_contract(base_size = 6.6) +
  theme(
    axis.text.x = element_text(angle = 42, hjust = 1, size = 5.4),
    axis.text.y = element_text(size = 5.2),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    strip.text.y = element_text(angle = 90, size = 5.8),
    strip.background = element_blank(),
    panel.spacing.y = grid::unit(2.5, "mm"),
    legend.position = "right"
  )
save_png(p_class, "Fig10C_all_endpoint_evidence_classification.png", width_mm = 250, height_mm = 205)

stability_plot_data <- bind_rows(
  within_group_audit %>%
    transmute(
      scale_variable, scale_label, scale_order, imaging_variable, imaging_label, imaging_order,
      panel = case_when(
        population == "HC" ~ "HC within-group effect",
        population == "SZ" & scale_population == "BOTH" ~ "SZ within | pleasure scales",
        TRUE ~ "SZ within | clinical/motor scales"
      ),
      loo_sign_stability
    ),
  interaction_audit %>%
    transmute(
      scale_variable, scale_label, scale_order, imaging_variable, imaging_label, imaging_order,
      panel = "Group x scale interaction",
      loo_sign_stability
    )
) %>%
  mutate(
    panel = factor(panel, levels = c(
      "HC within-group effect",
      "SZ within | pleasure scales",
      "SZ within | clinical/motor scales",
      "Group x scale interaction"
    )),
    stability_label = if_else(is.finite(loo_sign_stability), sprintf("%.0f", 100 * loo_sign_stability), "NA"),
    imaging_label = factor(imaging_label, levels = imaging_registry$imaging_label),
    scale_label = factor(scale_label, levels = rev(eligible_scales$label[eligible_scales$imaging_eligible]))
  )

p_stability <- ggplot(stability_plot_data, aes(x = imaging_label, y = scale_label, fill = loo_sign_stability)) +
  geom_tile(linewidth = 0.25, colour = "white") +
  geom_text(aes(label = stability_label), size = 1.95, colour = "black") +
  facet_grid(panel ~ ., scales = "free_y", space = "free_y") +
  scale_fill_gradient(
    low = "#F3D9D4", high = "#2A9D8F", limits = c(0, 1), oob = scales::squish,
    labels = scales::percent_format(accuracy = 1), name = "LOO sign\nstability"
  ) +
  labs(
    title = "Fig. 10d | Leave-one-subject-out direction stability",
    subtitle = "Cells report the percentage of leave-one-out estimates retaining the full-sample sign; 100 is required for the strict reliable-evidence tier.",
    x = NULL, y = NULL
  ) +
  theme_nature_contract(base_size = 6.4) +
  theme(
    axis.text.x = element_text(angle = 42, hjust = 1, size = 5.2),
    axis.text.y = element_text(size = 4.9),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    strip.text.y = element_text(angle = 90, size = 5.6),
    strip.background = element_blank(),
    panel.spacing.y = grid::unit(2.3, "mm"),
    legend.position = "right"
  )
save_png(p_stability, "Fig10D_all_endpoint_leave_one_out_stability.png", width_mm = 240, height_mm = 300)

# -----------------------------------------------------------------------------
# Index, counts and audit QA
# -----------------------------------------------------------------------------

figure_index_path <- file.path(tables_dir, "FIGURE_INDEX.tsv")
figure_index <- readr::read_tsv(figure_index_path, show_col_types = FALSE)
figure_index <- figure_index %>% filter(!figure_id %in% c("Fig10A", "Fig10B", "Fig10C", "Fig10D"))
new_figure_rows <- tribble(
  ~figure_id, ~filename, ~analysis_part, ~visualized_result, ~interpretation_scope,
  "Fig10A", "Fig10A_all_within_group_associations.png", 7, "All HC and SZ within-group scale-imaging associations", "Multiplicity, CI, influence and LOO audited",
  "Fig10B", "Fig10B_all_group_interactions.png", 7, "All 70 Group x scale interaction tests", "Direct evidence for HC/SZ slope differences",
  "Fig10C", "Fig10C_all_endpoint_evidence_classification.png", 7, "Evidence class for all 119 scale-imaging endpoints", "Primary audit summary",
  "Fig10D", "Fig10D_all_endpoint_leave_one_out_stability.png", 7, "LOO direction stability for all within-group and interaction effects", "Influence robustness"
)
figure_index <- bind_rows(figure_index, new_figure_rows)
write_tsv_safe(figure_index, "FIGURE_INDEX.tsv")

counts_path <- file.path(tables_dir, "ANALYSIS_OUTPUT_COUNTS.tsv")
counts <- readr::read_tsv(counts_path, show_col_types = FALSE) %>%
  filter(!item %in% c(
    "all_within_group_audit_tests", "all_group_interaction_audit_tests",
    "all_endpoint_evidence_classifications", "figures_expected"
  ))
counts <- bind_rows(
  counts,
  tibble(item = "all_within_group_audit_tests", value = nrow(within_group_audit)),
  tibble(item = "all_group_interaction_audit_tests", value = nrow(interaction_audit)),
  tibble(item = "all_endpoint_evidence_classifications", value = nrow(evidence_classification)),
  tibble(item = "figures_expected", value = nrow(figure_index))
)
write_tsv_safe(counts, "ANALYSIS_OUTPUT_COUNTS.tsv")

new_figure_files <- file.path(figures_dir, new_figure_rows$filename)
audit_qa <- tibble(
  check = c(
    "within_group_rows_189",
    "interaction_rows_70",
    "classification_rows_119",
    "hc_family_rows_70",
    "sz_family_rows_119",
    "all_new_figures_exist",
    "all_new_figures_png",
    "all_new_figures_gt10kb",
    "figure_index_total_27"
  ),
  passed = c(
    nrow(within_group_audit) == 189,
    nrow(interaction_audit) == 70,
    nrow(evidence_classification) == 119,
    sum(within_group_audit$population == "HC") == 70,
    sum(within_group_audit$population == "SZ") == 119,
    all(file.exists(new_figure_files)),
    all(str_ends(new_figure_files, regex("\\.png$", ignore_case = TRUE))),
    all(file.info(new_figure_files)$size > 10000),
    nrow(figure_index) == 27
  )
)
write_tsv_safe(audit_qa, "GROUP_AUDIT_QA_CHECKS.tsv")
if (!all(audit_qa$passed)) {
  stop("Group audit QA failed: ", paste(audit_qa$check[!audit_qa$passed], collapse = ", "))
}

writeLines(
  c(
    paste0("Completed: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    "Backend: R only",
    "Output: PNG only",
    paste0("Within-group audit rows: ", nrow(within_group_audit)),
    paste0("Group interaction rows: ", nrow(interaction_audit)),
    paste0("Endpoint classifications: ", nrow(evidence_classification)),
    paste0("Reliable HC within-group endpoints: ", sum(evidence_classification$reliable_within_hc)),
    paste0("Reliable SZ within-group endpoints: ", sum(evidence_classification$reliable_within_sz)),
    paste0("Direct group-difference endpoints: ", sum(evidence_classification$group_difference_evidence)),
    paste0("Nominal-only endpoints: ", sum(evidence_classification$primary_evidence_class == "EXPLORATORY_NOMINAL_SIGNAL"))
  ),
  file.path(logs_dir, "GROUP_AUDIT_RUN_COMPLETION.txt")
)

message("All-scale group audit completed successfully.")
