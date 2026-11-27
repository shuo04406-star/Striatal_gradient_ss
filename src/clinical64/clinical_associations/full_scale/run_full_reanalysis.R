options(stringsAsFactors = FALSE, warn = 1)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(tibble)
  library(stringr)
  library(patchwork)
  library(ragg)
  library(scales)
  library(broom)
  library(sandwich)
  library(boot)
})

set.seed(20261030)

analysis_dir <- Sys.getenv("STRIATAL_FULL_SCALE_OUTPUT", unset = NA_character_)
tables_dir <- file.path(analysis_dir, "tables")
figures_dir <- file.path(analysis_dir, "figures")
logs_dir <- file.path(analysis_dir, "logs")
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)

input_paths <- list(
  master = file.path(Sys.getenv("STRIATAL_FINAL64_LOCK", unset = NA_character_), "frozen_data", "subject_level_brain_behaviour_master_final64.tsv"),
  items = file.path(Sys.getenv("STRIATAL_FINAL64_LOCK", unset = NA_character_), "frozen_data", "subject_level_questionnaire_items_long.tsv"),
  matrix = file.path(Sys.getenv("STRIATAL_FINAL64_LOCK", unset = NA_character_), "analysis_preparation", "ANALYSIS_SAMPLE_MATRIX.tsv"),
  manifest = file.path(Sys.getenv("STRIATAL_FINAL64_LOCK", unset = NA_character_), "manifests", "FROZEN_DATA_MANIFEST.tsv")
)

stopifnot(all(file.exists(unlist(input_paths))))

palette_contract <- c(
  HC = "#4C78A8",
  SZ = "#D16D5A",
  ALL = "#5B6573",
  anticipation = "#2A9D8F",
  outcome = "#D98E3D",
  neutral_dark = "#2B2B2B",
  neutral_mid = "#7A7A7A",
  neutral_light = "#D9D9D9",
  ready = "#2A9D8F",
  qc = "#C49A45",
  blocked = "#A7A7A7"
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

save_png <- function(plot, filename, width_mm = 183, height_mm = 120, dpi = 300) {
  path <- file.path(figures_dir, filename)
  ragg::agg_png(
    filename = path,
    width = width_mm,
    height = height_mm,
    units = "mm",
    res = dpi,
    background = "white"
  )
  print(plot)
  grDevices::dev.off()
  invisible(path)
}

write_tsv_safe <- function(x, filename) {
  readr::write_tsv(x, file.path(tables_dir, filename), na = "")
}

zscore <- function(x) {
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(NA_real_, length(x)))
  as.numeric((x - mean(x, na.rm = TRUE)) / s)
}

skewness_safe <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 3 || stats::sd(x) == 0) return(NA_real_)
  mean((x - mean(x))^3) / stats::sd(x)^3
}

kurtosis_safe <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 4 || stats::sd(x) == 0) return(NA_real_)
  mean((x - mean(x))^4) / stats::sd(x)^4 - 3
}

cronbach_alpha <- function(mat) {
  mat <- as.matrix(mat)
  mat <- mat[stats::complete.cases(mat), , drop = FALSE]
  if (nrow(mat) < 10 || ncol(mat) < 2) return(NA_real_)
  vars <- apply(mat, 2, stats::var)
  if (any(!is.finite(vars))) return(NA_real_)
  s <- stats::cov(mat)
  denom <- sum(s)
  if (!is.finite(denom) || denom <= 0) return(NA_real_)
  k <- ncol(mat)
  as.numeric(k / (k - 1) * (1 - sum(diag(s)) / denom))
}

omega_total <- function(mat) {
  mat <- as.matrix(mat)
  mat <- mat[stats::complete.cases(mat), , drop = FALSE]
  if (nrow(mat) <= ncol(mat) || nrow(mat) < 12 || ncol(mat) < 3) return(NA_real_)
  vars <- apply(mat, 2, stats::var)
  if (any(!is.finite(vars)) || any(vars == 0)) return(NA_real_)
  fit <- tryCatch(
    suppressWarnings(stats::factanal(mat, factors = 1, rotation = "none")),
    error = function(e) NULL
  )
  if (is.null(fit)) return(NA_real_)
  lambda <- as.numeric(unclass(fit$loadings)[, 1])
  if (sum(lambda) < 0) lambda <- -lambda
  denom <- sum(lambda)^2 + sum(fit$uniquenesses)
  if (!is.finite(denom) || denom <= 0) return(NA_real_)
  as.numeric(sum(lambda)^2 / denom)
}

bootstrap_reliability <- function(mat, reps = 500, seed = 20261030) {
  mat <- as.matrix(mat)
  mat <- mat[stats::complete.cases(mat), , drop = FALSE]
  n <- nrow(mat)
  point_alpha <- cronbach_alpha(mat)
  point_omega <- omega_total(mat)
  zero_variance_items <- if (n > 1) sum(apply(mat, 2, stats::var) == 0) else NA_integer_
  if (n < 10 || ncol(mat) < 2 || !is.finite(point_alpha)) {
    return(tibble(
      n_complete = n,
      k_items = ncol(mat),
      zero_variance_items = zero_variance_items,
      alpha = point_alpha,
      alpha_ci_low = NA_real_,
      alpha_ci_high = NA_real_,
      omega_total = point_omega,
      omega_ci_low = NA_real_,
      omega_ci_high = NA_real_,
      valid_alpha_boot = 0L,
      valid_omega_boot = 0L
    ))
  }
  set.seed(seed)
  alpha_boot <- rep(NA_real_, reps)
  omega_boot <- rep(NA_real_, reps)
  for (b in seq_len(reps)) {
    idx <- sample.int(n, size = n, replace = TRUE)
    alpha_boot[b] <- cronbach_alpha(mat[idx, , drop = FALSE])
    omega_boot[b] <- omega_total(mat[idx, , drop = FALSE])
  }
  alpha_valid <- alpha_boot[is.finite(alpha_boot)]
  omega_valid <- omega_boot[is.finite(omega_boot)]
  tibble(
    n_complete = n,
    k_items = ncol(mat),
    zero_variance_items = zero_variance_items,
    alpha = point_alpha,
    alpha_ci_low = if (length(alpha_valid) >= 100) unname(stats::quantile(alpha_valid, 0.025)) else NA_real_,
    alpha_ci_high = if (length(alpha_valid) >= 100) unname(stats::quantile(alpha_valid, 0.975)) else NA_real_,
    omega_total = point_omega,
    omega_ci_low = if (length(omega_valid) >= 100) unname(stats::quantile(omega_valid, 0.025)) else NA_real_,
    omega_ci_high = if (length(omega_valid) >= 100) unname(stats::quantile(omega_valid, 0.975)) else NA_real_,
    valid_alpha_boot = length(alpha_valid),
    valid_omega_boot = length(omega_valid)
  )
}

robust_scale_effect <- function(data, x, y) {
  d <- data %>%
    transmute(
      x = .data[[x]],
      y = .data[[y]],
      Group = factor(Group, levels = c("HC", "SZ")),
      Age = Age,
      Sex = factor(Sex),
      MeanFD = MeanFD
    ) %>%
    filter(complete.cases(.))
  n <- nrow(d)
  if (n < 12 || n_distinct(d$x) < 3 || n_distinct(d$y) < 3 || n_distinct(d$Group) < 2) {
    return(tibble(n = n, effect = NA_real_, se = NA_real_, ci_low = NA_real_, ci_high = NA_real_, p = NA_real_))
  }
  d <- d %>% mutate(x_z = zscore(x), y_z = zscore(y), age_z = zscore(Age), fd_z = zscore(MeanFD))
  fit <- tryCatch(stats::lm(y_z ~ x_z + Group + age_z + Sex + fd_z, data = d), error = function(e) NULL)
  if (is.null(fit) || !("x_z" %in% names(stats::coef(fit)))) {
    return(tibble(n = n, effect = NA_real_, se = NA_real_, ci_low = NA_real_, ci_high = NA_real_, p = NA_real_))
  }
  v <- tryCatch(sandwich::vcovHC(fit, type = "HC3"), error = function(e) NULL)
  if (is.null(v) || !("x_z" %in% rownames(v))) {
    return(tibble(n = n, effect = NA_real_, se = NA_real_, ci_low = NA_real_, ci_high = NA_real_, p = NA_real_))
  }
  est <- unname(stats::coef(fit)["x_z"])
  se <- sqrt(v["x_z", "x_z"])
  p <- 2 * stats::pnorm(abs(est / se), lower.tail = FALSE)
  tibble(n = n, effect = est, se = se, ci_low = est - 1.96 * se, ci_high = est + 1.96 * se, p = p)
}

spearman_effect <- function(data, x, y, reps = 1000, seed = 20261030) {
  d <- data %>% select(x = all_of(x), y = all_of(y)) %>% filter(complete.cases(.))
  n <- nrow(d)
  if (n < 8 || n_distinct(d$x) < 3 || n_distinct(d$y) < 3) {
    return(tibble(n = n, effect = NA_real_, se = NA_real_, ci_low = NA_real_, ci_high = NA_real_, p = NA_real_, valid_boot = 0L))
  }
  rho <- suppressWarnings(stats::cor(d$x, d$y, method = "spearman"))
  p <- suppressWarnings(stats::cor.test(d$x, d$y, method = "spearman", exact = FALSE)$p.value)
  set.seed(seed)
  boots <- replicate(reps, {
    idx <- sample.int(n, size = n, replace = TRUE)
    suppressWarnings(stats::cor(d$x[idx], d$y[idx], method = "spearman"))
  })
  boots <- boots[is.finite(boots)]
  tibble(
    n = n,
    effect = rho,
    se = if (length(boots) > 1) stats::sd(boots) else NA_real_,
    ci_low = if (length(boots) >= 200) unname(stats::quantile(boots, 0.025)) else NA_real_,
    ci_high = if (length(boots) >= 200) unname(stats::quantile(boots, 0.975)) else NA_real_,
    p = p,
    valid_boot = length(boots)
  )
}

robust_group_effect <- function(data, y) {
  d <- data %>%
    transmute(
      y = .data[[y]],
      Group = factor(Group, levels = c("HC", "SZ")),
      Age = Age,
      Sex = factor(Sex),
      MeanFD = MeanFD
    ) %>%
    filter(complete.cases(.))
  n_hc <- sum(d$Group == "HC")
  n_sz <- sum(d$Group == "SZ")
  if (n_hc < 8 || n_sz < 8 || n_distinct(d$y) < 3) {
    return(tibble(n_hc = n_hc, n_sz = n_sz, effect = NA_real_, se = NA_real_, ci_low = NA_real_, ci_high = NA_real_, p = NA_real_, hedges_g = NA_real_))
  }
  d <- d %>% mutate(y_z = zscore(y), age_z = zscore(Age), fd_z = zscore(MeanFD))
  fit <- tryCatch(stats::lm(y_z ~ Group + age_z + Sex + fd_z, data = d), error = function(e) NULL)
  v <- if (!is.null(fit)) tryCatch(sandwich::vcovHC(fit, type = "HC3"), error = function(e) NULL) else NULL
  term <- "GroupSZ"
  if (is.null(v) || !(term %in% names(stats::coef(fit)))) {
    return(tibble(n_hc = n_hc, n_sz = n_sz, effect = NA_real_, se = NA_real_, ci_low = NA_real_, ci_high = NA_real_, p = NA_real_, hedges_g = NA_real_))
  }
  est <- unname(stats::coef(fit)[term])
  se <- sqrt(v[term, term])
  p <- 2 * stats::pnorm(abs(est / se), lower.tail = FALSE)
  raw_hc <- d$y[d$Group == "HC"]
  raw_sz <- d$y[d$Group == "SZ"]
  pooled_sd <- sqrt(((length(raw_hc) - 1) * stats::var(raw_hc) + (length(raw_sz) - 1) * stats::var(raw_sz)) /
    (length(raw_hc) + length(raw_sz) - 2))
  d_raw <- if (is.finite(pooled_sd) && pooled_sd > 0) (mean(raw_sz) - mean(raw_hc)) / pooled_sd else NA_real_
  correction <- 1 - 3 / (4 * (length(raw_hc) + length(raw_sz)) - 9)
  tibble(
    n_hc = n_hc,
    n_sz = n_sz,
    effect = est,
    se = se,
    ci_low = est - 1.96 * se,
    ci_high = est + 1.96 * se,
    p = p,
    hedges_g = d_raw * correction
  )
}

master <- readr::read_tsv(input_paths$master, show_col_types = FALSE, progress = FALSE)
items <- readr::read_tsv(input_paths$items, show_col_types = FALSE, progress = FALSE)
sample_matrix <- readr::read_tsv(input_paths$matrix, show_col_types = FALSE, progress = FALSE)
frozen_manifest <- readr::read_tsv(input_paths$manifest, show_col_types = FALSE, progress = FALSE)

stopifnot(
  nrow(master) == 64,
  n_distinct(master$Subject_ID) == 64,
  nrow(sample_matrix) == 64,
  n_distinct(sample_matrix$Subject_ID) == 64,
  setequal(master$Subject_ID, sample_matrix$Subject_ID),
  identical(sort(unique(sample_matrix$Group)), c("HC", "SZ")),
  all(table(sample_matrix$Group) == 32)
)

scale_registry <- tribble(
  ~variable, ~label, ~scale, ~domain, ~source_col, ~population, ~status, ~imaging_eligible, ~group_eligible, ~theoretical_min, ~theoretical_max,
  "TEPS_abstract_anticipatory", "TEPS abstract anticipatory", "TEPS", "Pleasure/anhedonia", "final64_v3__TEPS_abstract_anticipatory_calc", "BOTH", "READY_SECONDARY", TRUE, TRUE, 4, 24,
  "TEPS_concrete_anticipatory", "TEPS concrete anticipatory", "TEPS", "Pleasure/anhedonia", "final64_v3__TEPS_concrete_anticipatory_calc", "BOTH", "READY_SECONDARY", TRUE, TRUE, 5, 30,
  "TEPS_abstract_consummatory", "TEPS abstract consummatory", "TEPS", "Pleasure/anhedonia", "final64_v3__TEPS_abstract_consummatory_calc", "BOTH", "READY_SECONDARY", TRUE, TRUE, 6, 36,
  "TEPS_concrete_consummatory", "TEPS concrete consummatory", "TEPS", "Pleasure/anhedonia", "final64_v3__TEPS_concrete_consummatory_calc", "BOTH", "READY_SECONDARY", TRUE, TRUE, 4, 24,
  "TEPS_anticipatory", "TEPS anticipatory", "TEPS", "Pleasure/anhedonia", "final64_v3__TEPS_anticipatory_calc", "BOTH", "READY_PRIMARY", TRUE, TRUE, 9, 54,
  "TEPS_consummatory", "TEPS consummatory", "TEPS", "Pleasure/anhedonia", "final64_v3__TEPS_consummatory_calc", "BOTH", "READY_PRIMARY", TRUE, TRUE, 10, 60,
  "TEPS_total", "TEPS total", "TEPS", "Pleasure/anhedonia", "final64_v3__TEPS_total_calc", "BOTH", "READY_SECONDARY", TRUE, TRUE, 20, 120,
  "ACIPS_anticipatory", "ACIPS anticipatory", "ACIPS", "Pleasure/anhedonia", "final64_v3__ACIPS_anticipatory_calc", "BOTH", "READY_PRIMARY", TRUE, TRUE, 7, 42,
  "ACIPS_consummatory", "ACIPS consummatory", "ACIPS", "Pleasure/anhedonia", "final64_v3__ACIPS_consummatory_calc", "BOTH", "READY_PRIMARY", TRUE, TRUE, 10, 60,
  "ACIPS_total", "ACIPS total", "ACIPS", "Pleasure/anhedonia", "final64_v3__ACIPS_total_calc", "BOTH", "READY_SECONDARY", TRUE, TRUE, 17, 102,
  "SHAPS", "SHAPS total", "SHAPS", "Pleasure/anhedonia", "final64_v3__SHAPS", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "EES", "EES total", "EES", "Affect/mood", "final64_v3__EES", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, 17, 102,
  "BDI", "BDI total", "BDI", "Affect/mood", "final64_v3__BDI", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "PANAS_PA", "PANAS positive affect", "PANAS", "Affect/mood", "final64_v3__PANAS_PA", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, 10, 50,
  "PANAS_NA", "PANAS negative affect", "PANAS", "Affect/mood", "final64_v3__PANAS_NA", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, 10, 50,
  "ERQ", "ERQ total", "ERQ", "Affect/mood", "final64_v3__ERQ", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, 10, 70,
  "HSS_affiliative", "HSS affiliative", "HSS", "Humour/interpersonal", "HSS亲和", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "HSS_self_enhancing", "HSS self-enhancing", "HSS", "Humour/interpersonal", "HSS自强", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "HSS_aggressive", "HSS aggressive", "HSS", "Humour/interpersonal", "HSS嘲讽", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "HSS_self_defeating", "HSS self-defeating", "HSS", "Humour/interpersonal", "HSS自贬", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "SHS_1", "SHS subscale I", "SHS", "Humour/interpersonal", "SHS一", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "SHS_3", "SHS subscale III", "SHS", "Humour/interpersonal", "SHS三", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "SHS_4", "SHS subscale IV", "SHS", "Humour/interpersonal", "SHS四", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "MSHS_total", "MSHS total", "MSHS", "Humour/interpersonal", "MSHStotal", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, 24, 120,
  "MSHS_1", "MSHS subscale I", "MSHS", "Humour/interpersonal", "MSHS一", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "MSHS_2", "MSHS subscale II", "MSHS", "Humour/interpersonal", "MSHS二", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "MSHS_3", "MSHS subscale III", "MSHS", "Humour/interpersonal", "MSHS三", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "MSHS_4", "MSHS subscale IV", "MSHS", "Humour/interpersonal", "MSHS四", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "ICQ_total", "ICQ total", "ICQ", "Humour/interpersonal", "ICQtotal", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, 40, 200,
  "ICQ_initiation", "ICQ initiation", "ICQ", "Humour/interpersonal", "ICQ开创关系", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "ICQ_negative_assertion", "ICQ negative assertion", "ICQ", "Humour/interpersonal", "ICQ负面声明", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "ICQ_disclosure", "ICQ disclosure", "ICQ", "Humour/interpersonal", "ICQ自我揭露", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "ICQ_emotional_support", "ICQ emotional support", "ICQ", "Humour/interpersonal", "ICQ情绪支持", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "ICQ_conflict", "ICQ conflict", "ICQ", "Humour/interpersonal", "ICQ人际冲突", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, NA, NA,
  "HR", "HR mean score", "HR", "Humour/interpersonal", "final64_v3__HR", "BOTH", "QC_DEFINITION_PENDING", FALSE, FALSE, 1, 7,
  "BFI_extraversion", "BFI extraversion", "BFI/FIVE", "Personality", "final64_v3__BFI_extraversion_existing", "BOTH", "BLOCKED_SCORING_CONFLICT", FALSE, FALSE, 2, 14,
  "BFI_agreeableness", "BFI agreeableness", "BFI/FIVE", "Personality", "final64_v3__BFI_agreeableness_existing", "BOTH", "BLOCKED_SCORING_CONFLICT", FALSE, FALSE, 2, 14,
  "BFI_conscientiousness", "BFI conscientiousness", "BFI/FIVE", "Personality", "final64_v3__BFI_conscientiousness_existing", "BOTH", "BLOCKED_SCORING_CONFLICT", FALSE, FALSE, 2, 14,
  "BFI_emotional_stability", "BFI emotional stability", "BFI/FIVE", "Personality", "final64_v3__BFI_emotional_stability_existing", "BOTH", "BLOCKED_SCORING_CONFLICT", FALSE, FALSE, 2, 14,
  "BFI_openness", "BFI openness", "BFI/FIVE", "Personality", "final64_v3__BFI_openness_existing", "BOTH", "BLOCKED_SCORING_CONFLICT", FALSE, FALSE, 2, 14,
  "CSAS", "CSAS total", "CSAS/RCSAS", "Unresolved scoring", "final64_v3__CSAS", "BOTH", "BLOCKED_SCORING_CONFLICT", FALSE, FALSE, 0, 40,
  "CPAS", "CPAS total", "CPAS/RCPAS", "Unresolved scoring", "final64_v3__CPAS", "BOTH", "BLOCKED_SCORING_CONFLICT", FALSE, FALSE, 0, 61,
  "PANSS_positive", "PANSS positive", "PANSS", "Clinical symptoms", "final64_v3__PANSS_positive", "SZ_ONLY", "READY_WITH_SOURCE_WARNING", TRUE, FALSE, 7, 49,
  "PANSS_negative", "PANSS negative", "PANSS", "Clinical symptoms", "final64_v3__PANSS_negative", "SZ_ONLY", "READY_WITH_SOURCE_WARNING", TRUE, FALSE, 7, 49,
  "PANSS_general", "PANSS general", "PANSS", "Clinical symptoms", "final64_v3__PANSS_general", "SZ_ONLY", "READY_WITH_SOURCE_WARNING", TRUE, FALSE, 16, 112,
  "PANSS_total", "PANSS total", "PANSS", "Clinical symptoms", "final64_v3__PANSS_total", "SZ_ONLY", "READY_WITH_SOURCE_WARNING", TRUE, FALSE, 30, 210,
  "SANS", "SANS total", "SANS", "Clinical symptoms", "final64_v3__SANS", "SZ_ONLY", "READY_WITH_SOURCE_WARNING", TRUE, FALSE, NA, NA,
  "SAS", "SAS total", "SAS", "Motor symptoms", "final64_v3__SAS", "SZ_ONLY", "READY_EXPLORATORY_MOTOR", TRUE, FALSE, NA, NA,
  "BARS", "BARS total", "BARS", "Motor symptoms", "final64_v3__BARS", "SZ_ONLY", "READY_EXPLORATORY_MOTOR", TRUE, FALSE, NA, NA
) %>%
  mutate(
    order = row_number(),
    status_group = case_when(
      str_starts(status, "READY") ~ "Ready for approved role",
      status == "QC_DEFINITION_PENDING" ~ "QC/definition pending",
      TRUE ~ "Blocked scoring conflict"
    )
  )

# Resolve non-ASCII legacy column names by their stable field order. This avoids
# dependence on the Windows console code page while preserving exact frozen columns.
hss_cols <- grep("^HSS", names(master), value = TRUE)
shs_cols <- grep("^SHS", names(master), value = TRUE)
mshs_cols <- grep("^MSHS", names(master), value = TRUE)
icq_cols <- grep("^ICQ", names(master), value = TRUE)
stopifnot(length(hss_cols) == 4, length(shs_cols) == 3, length(mshs_cols) == 5, length(icq_cols) == 6)
legacy_non_ascii_map <- c(
  HSS_affiliative = hss_cols[1],
  HSS_self_enhancing = hss_cols[2],
  HSS_aggressive = hss_cols[3],
  HSS_self_defeating = hss_cols[4],
  SHS_1 = shs_cols[1],
  SHS_3 = shs_cols[2],
  SHS_4 = shs_cols[3],
  MSHS_total = mshs_cols[1],
  MSHS_1 = mshs_cols[2],
  MSHS_2 = mshs_cols[3],
  MSHS_3 = mshs_cols[4],
  MSHS_4 = mshs_cols[5],
  ICQ_total = icq_cols[1],
  ICQ_initiation = icq_cols[2],
  ICQ_negative_assertion = icq_cols[3],
  ICQ_disclosure = icq_cols[4],
  ICQ_emotional_support = icq_cols[5],
  ICQ_conflict = icq_cols[6]
)
idx_legacy <- match(scale_registry$variable, names(legacy_non_ascii_map))
scale_registry$source_col[!is.na(idx_legacy)] <- unname(legacy_non_ascii_map[idx_legacy[!is.na(idx_legacy)]])

write_tsv_safe(scale_registry, "SCALE_ANALYSIS_REGISTRY.tsv")

missing_source_cols <- setdiff(unique(scale_registry$source_col), names(master))
if (length(missing_source_cols) > 0) {
  stop("Missing scale source columns: ", paste(missing_source_cols, collapse = ", "))
}

scale_wide <- sample_matrix %>%
  select(Subject_ID, Group, Age, Sex, MeanFD, all_of(c(
    "Anticipation_slope", "Outcome_slope", "Anticipation_similarity", "Outcome_similarity",
    "SMA_slope", "IFG_slope", "Cingulate_slope"
  ))) %>%
  left_join(master %>% select(Subject_ID, all_of(unique(scale_registry$source_col))), by = "Subject_ID")

score_long <- map_dfr(seq_len(nrow(scale_registry)), function(i) {
  reg <- scale_registry[i, ]
  scale_wide %>%
    transmute(
      Subject_ID,
      Group,
      variable = reg$variable,
      value = suppressWarnings(as.numeric(.data[[reg$source_col]]))
    )
}) %>%
  left_join(scale_registry, by = "variable")

stopifnot(nrow(score_long) == 64 * nrow(scale_registry))

# -----------------------------------------------------------------------------
# Part 1: scale information content and item-level QC
# -----------------------------------------------------------------------------

summarise_score_population <- function(variable_id, population_id) {
  reg <- scale_registry %>% filter(variable == variable_id)
  d <- score_long %>% filter(variable == variable_id)
  if (population_id %in% c("HC", "SZ")) d <- d %>% filter(Group == population_id)
  x <- d$value
  n_applicable <- nrow(d)
  n_valid <- sum(is.finite(x))
  x_valid <- x[is.finite(x)]
  theor_min <- reg$theoretical_min
  theor_max <- reg$theoretical_max
  theor_span <- theor_max - theor_min
  observed_min <- if (n_valid > 0) min(x_valid) else NA_real_
  observed_max <- if (n_valid > 0) max(x_valid) else NA_real_
  observed_span <- observed_max - observed_min
  floor_pct <- if (n_valid > 0 && is.finite(theor_min)) 100 * mean(x_valid == theor_min) else NA_real_
  ceiling_pct <- if (n_valid > 0 && is.finite(theor_max)) 100 * mean(x_valid == theor_max) else NA_real_
  observed_floor_pct <- if (n_valid > 0) 100 * mean(x_valid == observed_min) else NA_real_
  observed_ceiling_pct <- if (n_valid > 0) 100 * mean(x_valid == observed_max) else NA_real_
  range_coverage_pct <- if (n_valid > 0 && is.finite(theor_span) && theor_span > 0) 100 * observed_span / theor_span else NA_real_
  spread_denominator <- if (is.finite(theor_span) && theor_span > 0) theor_span else observed_span
  normalized_sd_pct <- if (n_valid > 1 && is.finite(spread_denominator) && spread_denominator > 0) {
    100 * stats::sd(x_valid) / spread_denominator
  } else {
    NA_real_
  }
  missing_pct <- if (n_applicable > 0) 100 * (n_applicable - n_valid) / n_applicable else NA_real_
  information_flag <- case_when(
    n_valid < 10 ~ "INSUFFICIENT_N",
    n_distinct(x_valid) <= 1 ~ "NO_VARIATION",
    missing_pct >= 20 ~ "HIGH_MISSINGNESS",
    is.finite(floor_pct) && floor_pct >= 15 ~ "FLOOR_EFFECT_GE15PCT",
    is.finite(ceiling_pct) && ceiling_pct >= 15 ~ "CEILING_EFFECT_GE15PCT",
    is.finite(range_coverage_pct) && range_coverage_pct < 20 ~ "NARROW_RANGE_LT20PCT",
    TRUE ~ "NO_MAJOR_INFORMATION_FLAG"
  )
  tibble(
    variable = variable_id,
    population = population_id,
    n_applicable = n_applicable,
    n_valid = n_valid,
    n_missing = n_applicable - n_valid,
    missing_pct = missing_pct,
    mean = if (n_valid > 0) mean(x_valid) else NA_real_,
    sd = if (n_valid > 1) stats::sd(x_valid) else NA_real_,
    variance = if (n_valid > 1) stats::var(x_valid) else NA_real_,
    median = if (n_valid > 0) stats::median(x_valid) else NA_real_,
    q1 = if (n_valid > 0) unname(stats::quantile(x_valid, 0.25)) else NA_real_,
    q3 = if (n_valid > 0) unname(stats::quantile(x_valid, 0.75)) else NA_real_,
    observed_min = observed_min,
    observed_max = observed_max,
    observed_span = observed_span,
    unique_values = n_distinct(x_valid),
    skewness = skewness_safe(x_valid),
    excess_kurtosis = kurtosis_safe(x_valid),
    theoretical_min = theor_min,
    theoretical_max = theor_max,
    floor_pct = floor_pct,
    ceiling_pct = ceiling_pct,
    observed_floor_pct = observed_floor_pct,
    observed_ceiling_pct = observed_ceiling_pct,
    range_coverage_pct = range_coverage_pct,
    normalized_sd_pct = normalized_sd_pct,
    spread_basis = if_else(is.finite(theor_span), "THEORETICAL_RANGE", "OBSERVED_RANGE"),
    information_flag = information_flag
  )
}

score_information <- map_dfr(scale_registry$variable, function(v) {
  reg <- scale_registry %>% filter(variable == v)
  pops <- if (reg$population == "BOTH") c("ALL", "HC", "SZ") else "SZ"
  map_dfr(pops, ~summarise_score_population(v, .x))
}) %>%
  left_join(scale_registry %>% select(variable, label, scale, domain, status, status_group, order), by = "variable") %>%
  arrange(order, factor(population, levels = c("ALL", "HC", "SZ")))

write_tsv_safe(score_information, "SCALE_INFORMATION_CONTENT.tsv")

plot_distribution_atlas <- function(status_filter, title, subtitle, filename, width_mm, height_mm) {
  vars <- scale_registry %>% filter(status_group %in% status_filter) %>% pull(variable)
  d <- score_long %>%
    filter(variable %in% vars, is.finite(value)) %>%
    mutate(
      label = factor(label, levels = scale_registry$label[scale_registry$variable %in% vars]),
      Group = factor(Group, levels = c("HC", "SZ"))
    )
  n_labels <- d %>% count(label, Group, name = "n")
  p <- ggplot(d, aes(x = Group, y = value, colour = Group, fill = Group)) +
    geom_violin(alpha = 0.22, linewidth = 0.3, trim = FALSE, scale = "width", na.rm = TRUE) +
    geom_boxplot(width = 0.18, alpha = 0.62, linewidth = 0.35, outlier.shape = NA, na.rm = TRUE) +
    geom_point(position = position_jitter(width = 0.08, height = 0), size = 0.65, alpha = 0.55, stroke = 0, na.rm = TRUE) +
    geom_text(
      data = n_labels,
      aes(x = Group, y = Inf, label = paste0("n=", n), colour = Group),
      inherit.aes = FALSE,
      vjust = 1.15,
      size = 2.05,
      show.legend = FALSE
    ) +
    facet_wrap(~label, scales = "free_y", ncol = 4) +
    scale_colour_manual(values = palette_contract[c("HC", "SZ")], drop = FALSE) +
    scale_fill_manual(values = palette_contract[c("HC", "SZ")], drop = FALSE) +
    labs(title = title, subtitle = subtitle, x = NULL, y = "Frozen score") +
    theme_nature_contract(base_size = 7) +
    theme(
      legend.position = "top",
      strip.text = element_text(size = 6.1),
      axis.text.x = element_text(size = 5.8),
      panel.spacing = grid::unit(2.2, "mm")
    )
  save_png(p, filename, width_mm = width_mm, height_mm = height_mm, dpi = 300)
}

plot_distribution_atlas(
  "Ready for approved role",
  "Fig. 1a | Distributions of analysis-eligible scale scores",
  "Dots are participants; boxes show median and IQR. HC values are structurally absent for SZ-only clinical scales.",
  "Fig01A_ready_scale_distributions.png",
  210, 245
)
plot_distribution_atlas(
  "QC/definition pending",
  "Fig. 1b | Distributions of definition-pending scale scores",
  "These scores are shown for measurement QC only and are not used for imaging inference.",
  "Fig01B_definition_pending_scale_distributions.png",
  210, 265
)
plot_distribution_atlas(
  "Blocked scoring conflict",
  "Fig. 1c | Distributions of scoring-blocked scale scores",
  "BFI/FIVE, CSAS and CPAS remain quarantined because historical scoring conflicts are unresolved.",
  "Fig01C_blocked_scale_distributions.png",
  190, 150
)

primary_information <- score_information %>%
  filter((population == "ALL" & scale_registry$population[match(variable, scale_registry$variable)] == "BOTH") |
           (population == "SZ" & scale_registry$population[match(variable, scale_registry$variable)] == "SZ_ONLY")) %>%
  mutate(label = factor(label, levels = rev(scale_registry$label)))

information_plot_data <- primary_information %>%
  select(variable, label, status_group, missing_pct, range_coverage_pct, floor_pct, ceiling_pct, normalized_sd_pct, unique_values) %>%
  pivot_longer(
    cols = c(missing_pct, range_coverage_pct, floor_pct, ceiling_pct, normalized_sd_pct, unique_values),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = factor(
      metric,
      levels = c("missing_pct", "range_coverage_pct", "floor_pct", "ceiling_pct", "normalized_sd_pct", "unique_values"),
      labels = c("Missing (%)", "Range coverage (%)", "Floor (%)", "Ceiling (%)", "Normalized SD (%)", "Unique values")
    ),
    status_group = factor(status_group, levels = c("Ready for approved role", "QC/definition pending", "Blocked scoring conflict"))
  )

p_info <- ggplot(information_plot_data, aes(x = value, y = label, colour = status_group)) +
  geom_point(size = 1.3, alpha = 0.9, na.rm = TRUE) +
  facet_wrap(~metric, nrow = 1, scales = "free_x") +
  scale_colour_manual(values = c(
    "Ready for approved role" = palette_contract[["ready"]],
    "QC/definition pending" = palette_contract[["qc"]],
    "Blocked scoring conflict" = palette_contract[["blocked"]]
  )) +
  labs(
    title = "Fig. 2 | Information content of every frozen scale score",
    subtitle = "Theoretical floor, ceiling and range coverage are blank where the local score definition is not sufficiently verified.",
    x = NULL,
    y = NULL,
    colour = "Analysis status"
  ) +
  theme_nature_contract(base_size = 6.7) +
  theme(
    legend.position = "top",
    axis.text.y = element_text(size = 5.2),
    strip.text = element_text(size = 5.8),
    panel.spacing.x = grid::unit(2.5, "mm")
  )
save_png(p_info, "Fig02_scale_information_content.png", width_mm = 260, height_mm = 235, dpi = 300)

questionnaire_ranges <- tribble(
  ~Questionnaire, ~valid_min, ~valid_max, ~applicable_population,
  "ACIPS", 1, 6, "BOTH",
  "BARS", 0, 4, "SZ_ONLY",
  "BDI", 0, 3, "BOTH",
  "BFI/FIVE", 1, 7, "BOTH",
  "CPAS/RCPAS", 0, 1, "BOTH",
  "CSAS/RCSAS", 0, 1, "BOTH",
  "EES", 1, 6, "BOTH",
  "ERQ", 1, 7, "BOTH",
  "HR", 1, 7, "BOTH",
  "HSS", 1, 7, "BOTH",
  "ICQ", 1, 5, "BOTH",
  "MSHS", 1, 5, "BOTH",
  "PANAS", 1, 5, "BOTH",
  "PANSS", 1, 7, "SZ_ONLY",
  "SANS", 0, 5, "SZ_ONLY",
  "SAS", 0, 4, "SZ_ONLY",
  "SHAPS", 1, 4, "BOTH",
  "SHS", 1, 5, "BOTH",
  "TEPS", 1, 6, "BOTH"
)

items_final64 <- items %>%
  filter(Subject_ID %in% master$Subject_ID) %>%
  inner_join(questionnaire_ranges, by = "Questionnaire") %>%
  mutate(
    item_number = suppressWarnings(as.integer(str_extract(Item_ID, "[0-9]+"))),
    invalid_value = !is.na(Raw_Value) & (Raw_Value < valid_min | Raw_Value > valid_max),
    value_clean = if_else(invalid_value, NA_real_, Raw_Value)
  )

questionnaire_matrix <- function(q_name, item_numbers = NULL, reverse_items = integer(), reverse_constant = NA_real_, item_pattern = NULL) {
  q_info <- questionnaire_ranges %>% filter(Questionnaire == q_name)
  ids <- if (q_info$applicable_population == "SZ_ONLY") {
    sample_matrix %>% filter(Group == "SZ") %>% pull(Subject_ID)
  } else {
    sample_matrix$Subject_ID
  }
  d <- items_final64 %>% filter(Questionnaire == q_name)
  if (!is.null(item_numbers)) d <- d %>% filter(item_number %in% item_numbers)
  if (!is.null(item_pattern)) d <- d %>% filter(str_detect(Item_ID, item_pattern))
  if (length(reverse_items) > 0) {
    d <- d %>% mutate(value_clean = if_else(item_number %in% reverse_items, reverse_constant - value_clean, value_clean))
  }
  w <- d %>%
    select(Subject_ID, Item_ID, value_clean) %>%
    distinct(Subject_ID, Item_ID, .keep_all = TRUE) %>%
    pivot_wider(names_from = Item_ID, values_from = value_clean) %>%
    right_join(tibble(Subject_ID = ids), by = "Subject_ID") %>%
    arrange(match(Subject_ID, ids))
  as.data.frame(w %>% select(-Subject_ID))
}

item_qc <- map_dfr(questionnaire_ranges$Questionnaire, function(q_name) {
  q_info <- questionnaire_ranges %>% filter(Questionnaire == q_name)
  d <- items_final64 %>% filter(Questionnaire == q_name)
  k_items <- n_distinct(d$Item_ID)
  n_applicable <- if (q_info$applicable_population == "SZ_ONLY") 32L else 64L
  expected_cells <- n_applicable * k_items
  mat <- questionnaire_matrix(q_name)
  rel <- bootstrap_reliability(mat, reps = 250, seed = 20261030 + match(q_name, questionnaire_ranges$Questionnaire))
  tibble(
    Questionnaire = q_name,
    applicable_population = q_info$applicable_population,
    n_applicable = n_applicable,
    k_items = k_items,
    expected_item_cells = expected_cells,
    observed_valid_cells = sum(is.finite(d$value_clean)),
    missing_or_invalid_cells = expected_cells - sum(is.finite(d$value_clean)),
    missing_or_invalid_pct = 100 * (expected_cells - sum(is.finite(d$value_clean))) / expected_cells,
    invalid_value_count = sum(d$invalid_value, na.rm = TRUE),
    complete_case_n = sum(stats::complete.cases(mat)),
    raw_unkeyed_alpha = rel$alpha,
    raw_unkeyed_alpha_ci_low = rel$alpha_ci_low,
    raw_unkeyed_alpha_ci_high = rel$alpha_ci_high,
    raw_unkeyed_omega = rel$omega_total,
    reliability_interpretation = "RAW_UNKEYED_QC_ONLY"
  )
})

write_tsv_safe(item_qc, "QUESTIONNAIRE_ITEM_QC.tsv")
write_tsv_safe(items_final64 %>% filter(invalid_value) %>% select(Subject_ID, Group, Questionnaire, Item_ID, Raw_Value, valid_min, valid_max), "ITEM_OUT_OF_RANGE_FLAGS.tsv")

formal_item_sets <- list(
  list(variable = "TEPS_abstract_anticipatory", label = "TEPS abstract anticipatory", q = "TEPS", items = c(4, 6, 18, 20), reverse = integer(), constant = NA_real_, key_status = "AUTHORITATIVE"),
  list(variable = "TEPS_concrete_anticipatory", label = "TEPS concrete anticipatory", q = "TEPS", items = c(1, 5, 8, 10, 11), reverse = integer(), constant = NA_real_, key_status = "AUTHORITATIVE"),
  list(variable = "TEPS_abstract_consummatory", label = "TEPS abstract consummatory", q = "TEPS", items = c(2, 7, 3, 9, 14, 19), reverse = integer(), constant = NA_real_, key_status = "AUTHORITATIVE"),
  list(variable = "TEPS_concrete_consummatory", label = "TEPS concrete consummatory", q = "TEPS", items = c(12, 15, 16, 17), reverse = integer(), constant = NA_real_, key_status = "AUTHORITATIVE"),
  list(variable = "TEPS_anticipatory", label = "TEPS anticipatory", q = "TEPS", items = c(4, 6, 18, 20, 1, 5, 8, 10, 11), reverse = integer(), constant = NA_real_, key_status = "AUTHORITATIVE"),
  list(variable = "TEPS_consummatory", label = "TEPS consummatory", q = "TEPS", items = c(2, 7, 3, 9, 14, 19, 12, 15, 16, 17), reverse = integer(), constant = NA_real_, key_status = "AUTHORITATIVE"),
  list(variable = "TEPS_total", label = "TEPS total", q = "TEPS", items = 1:20, reverse = 13, constant = 7, key_status = "AUTHORITATIVE"),
  list(variable = "ACIPS_anticipatory", label = "ACIPS anticipatory", q = "ACIPS", items = c(1, 3, 7, 8, 10, 14, 15), reverse = 3, constant = 7, key_status = "AUTHORITATIVE"),
  list(variable = "ACIPS_consummatory", label = "ACIPS consummatory", q = "ACIPS", items = c(2, 4, 5, 6, 9, 11, 12, 13, 16, 17), reverse = integer(), constant = NA_real_, key_status = "AUTHORITATIVE"),
  list(variable = "ACIPS_total", label = "ACIPS total", q = "ACIPS", items = 1:17, reverse = 3, constant = 7, key_status = "AUTHORITATIVE"),
  list(variable = "PANSS_positive", label = "PANSS positive", q = "PANSS", items = NULL, pattern = "^P", reverse = integer(), constant = NA_real_, key_status = "SOURCE_PREFIX_KEY"),
  list(variable = "PANSS_negative", label = "PANSS negative", q = "PANSS", items = NULL, pattern = "^N", reverse = integer(), constant = NA_real_, key_status = "SOURCE_PREFIX_KEY"),
  list(variable = "PANSS_general", label = "PANSS general", q = "PANSS", items = NULL, pattern = "^G", reverse = integer(), constant = NA_real_, key_status = "SOURCE_PREFIX_KEY"),
  list(variable = "PANSS_total", label = "PANSS total", q = "PANSS", items = NULL, pattern = "^[PNG]", reverse = integer(), constant = NA_real_, key_status = "SOURCE_PREFIX_KEY"),
  list(variable = "SANS", label = "SANS total", q = "SANS", items = 1:24, reverse = integer(), constant = NA_real_, key_status = "SOURCE_SUM_KEY"),
  list(variable = "SAS", label = "SAS total", q = "SAS", items = 1:9, reverse = integer(), constant = NA_real_, key_status = "SOURCE_SUM_KEY"),
  list(variable = "BARS", label = "BARS total", q = "BARS", items = 1:4, reverse = integer(), constant = NA_real_, key_status = "SOURCE_SUM_KEY"),
  list(variable = "SHAPS", label = "SHAPS total (QC)", q = "SHAPS", items = 1:14, reverse = integer(), constant = NA_real_, key_status = "SOURCE_TOTAL_QC_ONLY"),
  list(variable = "EES", label = "EES total (QC)", q = "EES", items = 1:17, reverse = integer(), constant = NA_real_, key_status = "SOURCE_TOTAL_QC_ONLY"),
  list(variable = "BDI", label = "BDI total (QC)", q = "BDI", items = 1:21, reverse = integer(), constant = NA_real_, key_status = "SOURCE_TOTAL_QC_ONLY"),
  list(variable = "ERQ", label = "ERQ total (QC)", q = "ERQ", items = 1:10, reverse = integer(), constant = NA_real_, key_status = "SOURCE_TOTAL_QC_ONLY"),
  list(variable = "HR", label = "HR mean (QC)", q = "HR", items = 1:96, reverse = integer(), constant = NA_real_, key_status = "SOURCE_TOTAL_QC_ONLY")
)

reliability_results <- imap_dfr(formal_item_sets, function(spec, idx) {
  item_pattern <- if (!is.null(spec$pattern)) spec$pattern else NULL
  mat <- questionnaire_matrix(spec$q, spec$items, spec$reverse, spec$constant, item_pattern = item_pattern)
  bootstrap_reliability(mat, reps = 500, seed = 20262000 + idx) %>%
    mutate(
      variable = spec$variable,
      label = spec$label,
      Questionnaire = spec$q,
      key_status = spec$key_status,
      .before = 1
    )
})

write_tsv_safe(reliability_results, "SCALE_INTERNAL_CONSISTENCY.tsv")

item_qc_plot <- item_qc %>%
  select(Questionnaire, missing_or_invalid_pct, invalid_value_count, complete_case_n) %>%
  pivot_longer(-Questionnaire, names_to = "metric", values_to = "value") %>%
  mutate(
    metric = factor(
      metric,
      levels = c("missing_or_invalid_pct", "invalid_value_count", "complete_case_n"),
      labels = c("Missing/invalid cells (%)", "Out-of-range values (n)", "Complete-case participants (n)")
    ),
    Questionnaire = factor(Questionnaire, levels = rev(questionnaire_ranges$Questionnaire))
  )

p_item_qc <- ggplot(item_qc_plot, aes(x = value, y = Questionnaire)) +
  geom_segment(aes(x = 0, xend = value, yend = Questionnaire), linewidth = 0.35, colour = palette_contract[["neutral_light"]]) +
  geom_point(size = 1.6, colour = palette_contract[["neutral_dark"]]) +
  facet_wrap(~metric, nrow = 1, scales = "free_x") +
  labs(
    title = "Fig. 3a | Item-level completeness and range audit",
    subtitle = "Out-of-range values are flagged in derived QC and treated as missing for reliability calculations; frozen sources are unchanged.",
    x = NULL,
    y = NULL
  ) +
  theme_nature_contract(base_size = 7) +
  theme(strip.text = element_text(size = 6.2), panel.spacing.x = grid::unit(4, "mm"))
save_png(p_item_qc, "Fig03A_item_level_qc.png", width_mm = 190, height_mm = 145, dpi = 300)

rel_plot_data <- reliability_results %>%
  select(label, key_status, n_complete, alpha, alpha_ci_low, alpha_ci_high, omega_total, omega_ci_low, omega_ci_high) %>%
  pivot_longer(
    cols = c(alpha, omega_total),
    names_to = "coefficient",
    values_to = "estimate"
  ) %>%
  mutate(
    ci_low = if_else(coefficient == "alpha", alpha_ci_low, omega_ci_low),
    ci_high = if_else(coefficient == "alpha", alpha_ci_high, omega_ci_high),
    coefficient = recode(coefficient, alpha = "Cronbach alpha", omega_total = "McDonald omega total"),
    label = factor(label, levels = rev(reliability_results$label)),
    interpretation = if_else(key_status == "SOURCE_TOTAL_QC_ONLY", "QC only", "Scoring key verified")
  )

p_rel <- ggplot(rel_plot_data, aes(x = estimate, y = label, colour = coefficient, shape = interpretation)) +
  geom_vline(xintercept = 0.70, linewidth = 0.35, linetype = "dashed", colour = palette_contract[["neutral_mid"]]) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = 0, linewidth = 0.35, position = position_dodge(width = 0.45), na.rm = TRUE) +
  geom_point(size = 1.6, position = position_dodge(width = 0.45), na.rm = TRUE) +
  scale_colour_manual(values = c("Cronbach alpha" = palette_contract[["anticipation"]], "McDonald omega total" = palette_contract[["outcome"]])) +
  scale_shape_manual(values = c("Scoring key verified" = 16, "QC only" = 1)) +
  coord_cartesian(xlim = c(-0.25, 1.05)) +
  labs(
    title = "Fig. 3b | Internal consistency of scoreable scales",
    subtitle = "Intervals are participant-bootstrap 95% CIs; omega is reported only when a one-factor model is estimable.",
    x = "Reliability coefficient",
    y = NULL,
    colour = NULL,
    shape = NULL
  ) +
  theme_nature_contract(base_size = 7) +
  theme(legend.position = "top", axis.text.y = element_text(size = 5.7))
save_png(p_rel, "Fig03B_internal_consistency.png", width_mm = 205, height_mm = 170, dpi = 300)

raw_rel_plot <- item_qc %>%
  mutate(Questionnaire = factor(Questionnaire, levels = rev(Questionnaire)))

p_raw_rel <- ggplot(raw_rel_plot, aes(x = raw_unkeyed_alpha, y = Questionnaire)) +
  geom_vline(xintercept = 0.70, linewidth = 0.35, linetype = "dashed", colour = palette_contract[["neutral_mid"]]) +
  geom_errorbar(aes(xmin = raw_unkeyed_alpha_ci_low, xmax = raw_unkeyed_alpha_ci_high), orientation = "y", width = 0, linewidth = 0.4, colour = palette_contract[["neutral_mid"]], na.rm = TRUE) +
  geom_point(size = 1.7, colour = palette_contract[["neutral_dark"]], na.rm = TRUE) +
  coord_cartesian(xlim = c(-0.5, 1.05)) +
  labs(
    title = "Fig. 3c | Raw unkeyed questionnaire alpha (QC only)",
    subtitle = "These coefficients are not substantive reliability estimates when reverse keys or multidimensional subscales are unresolved.",
    x = "Raw unkeyed Cronbach alpha",
    y = NULL
  ) +
  theme_nature_contract(base_size = 7)
save_png(p_raw_rel, "Fig03C_raw_unkeyed_reliability_qc.png", width_mm = 160, height_mm = 125, dpi = 300)

# -----------------------------------------------------------------------------
# Part 2: associations among scales and subscales
# -----------------------------------------------------------------------------

score_wide_by_variable <- score_long %>%
  select(Subject_ID, Group, variable, value) %>%
  distinct(Subject_ID, Group, variable, .keep_all = TRUE) %>%
  pivot_wider(names_from = variable, values_from = value)

analysis_data <- sample_matrix %>%
  select(Subject_ID, Group, Age, Sex, MeanFD,
         Anticipation_slope, Outcome_slope, Anticipation_similarity, Outcome_similarity,
         SMA_slope, IFG_slope, Cingulate_slope) %>%
  left_join(score_wide_by_variable %>% select(-Group), by = "Subject_ID")

pairwise_scale_correlations <- function(population_id) {
  vars <- if (population_id == "SZ") {
    scale_registry$variable
  } else {
    scale_registry %>% filter(population == "BOTH") %>% pull(variable)
  }
  d <- analysis_data
  if (population_id %in% c("HC", "SZ")) d <- d %>% filter(Group == population_id)
  combos <- t(utils::combn(vars, 2))
  out <- map_dfr(seq_len(nrow(combos)), function(i) {
    v1 <- combos[i, 1]
    v2 <- combos[i, 2]
    xy <- d %>% select(x = all_of(v1), y = all_of(v2)) %>% filter(complete.cases(.))
    n <- nrow(xy)
    if (n < 8 || n_distinct(xy$x) < 3 || n_distinct(xy$y) < 3) {
      return(tibble(population = population_id, variable_1 = v1, variable_2 = v2, n = n, rho = NA_real_, p = NA_real_))
    }
    test <- suppressWarnings(stats::cor.test(xy$x, xy$y, method = "spearman", exact = FALSE))
    tibble(
      population = population_id,
      variable_1 = v1,
      variable_2 = v2,
      n = n,
      rho = unname(test$estimate),
      p = test$p.value
    )
  })
  eligible_vars <- scale_registry %>% filter(imaging_eligible) %>% pull(variable)
  out <- out %>%
    mutate(
      inferential_eligible = variable_1 %in% eligible_vars & variable_2 %in% eligible_vars,
      p_adj_bh_eligible = NA_real_
    )
  idx <- which(out$inferential_eligible & is.finite(out$p))
  out$p_adj_bh_eligible[idx] <- stats::p.adjust(out$p[idx], method = "BH")
  out
}

scale_correlations <- bind_rows(
  pairwise_scale_correlations("ALL"),
  pairwise_scale_correlations("HC"),
  pairwise_scale_correlations("SZ")
) %>%
  left_join(scale_registry %>% select(variable_1 = variable, label_1 = label, status_1 = status, order_1 = order), by = "variable_1") %>%
  left_join(scale_registry %>% select(variable_2 = variable, label_2 = label, status_2 = status, order_2 = order), by = "variable_2") %>%
  arrange(factor(population, levels = c("ALL", "HC", "SZ")), order_1, order_2)

write_tsv_safe(scale_correlations, "SCALE_TO_SCALE_SPEARMAN_CORRELATIONS.tsv")

correlation_matrix_data <- function(population_id) {
  vars <- if (population_id == "SZ") {
    scale_registry$variable
  } else {
    scale_registry %>% filter(population == "BOTH") %>% pull(variable)
  }
  labels <- scale_registry$label[match(vars, scale_registry$variable)]
  off <- scale_correlations %>%
    filter(population == population_id, variable_1 %in% vars, variable_2 %in% vars) %>%
    select(variable_1, variable_2, rho, n, p_adj_bh_eligible, inferential_eligible)
  mirror <- off %>% transmute(
    variable_1 = variable_2,
    variable_2 = variable_1,
    rho,
    n,
    p_adj_bh_eligible,
    inferential_eligible
  )
  diag <- tibble(
    variable_1 = vars,
    variable_2 = vars,
    rho = 1,
    n = NA_integer_,
    p_adj_bh_eligible = NA_real_,
    inferential_eligible = TRUE
  )
  bind_rows(off, mirror, diag) %>%
    mutate(
      label_1 = factor(scale_registry$label[match(variable_1, scale_registry$variable)], levels = rev(labels)),
      label_2 = factor(scale_registry$label[match(variable_2, scale_registry$variable)], levels = labels),
      significant = is.finite(p_adj_bh_eligible) & p_adj_bh_eligible < 0.05
    )
}

plot_correlation_matrix <- function(population_id, filename, title, subtitle, width_mm, height_mm) {
  d <- correlation_matrix_data(population_id)
  p <- ggplot(d, aes(x = label_2, y = label_1, fill = rho)) +
    geom_tile(linewidth = 0.08, colour = "white", na.rm = FALSE) +
    geom_point(data = d %>% filter(significant), shape = 16, size = 0.55, colour = "black") +
    scale_fill_gradient2(
      low = "#3B6FB6",
      mid = "white",
      high = "#C9574D",
      midpoint = 0,
      limits = c(-1, 1),
      na.value = "#EEEEEE",
      name = "Spearman\nrho"
    ) +
    coord_fixed() +
    labs(title = title, subtitle = subtitle, x = NULL, y = NULL) +
    theme_nature_contract(base_size = 6.2) +
    theme(
      axis.text.x = element_text(angle = 55, hjust = 1, vjust = 1, size = 4.2),
      axis.text.y = element_text(size = 4.2),
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      legend.position = "right",
      plot.margin = margin(5, 6, 5, 5, unit = "mm")
    )
  save_png(p, filename, width_mm = width_mm, height_mm = height_mm, dpi = 300)
}

plot_correlation_matrix(
  "ALL",
  "Fig04A_scale_correlations_all.png",
  "Fig. 4a | Scale overlap in the combined sample",
  "Only scales applicable to both groups are shown. Black dots mark BH-FDR < 0.05 within analysis-eligible score pairs; all other cells are descriptive.",
  235, 220
)
plot_correlation_matrix(
  "HC",
  "Fig04B_scale_correlations_hc.png",
  "Fig. 4b | Scale overlap within healthy controls",
  "Within-HC Spearman correlations reduce confounding by diagnostic group; black dots use the same eligibility-restricted BH family.",
  235, 220
)
plot_correlation_matrix(
  "SZ",
  "Fig04C_scale_correlations_sz.png",
  "Fig. 4c | Scale overlap within schizophrenia",
  "The SZ matrix includes clinical and motor scales. Totals and their component subscales remain non-independent by construction.",
  250, 235
)

# -----------------------------------------------------------------------------
# Part 3: all eligible scale-to-imaging associations
# -----------------------------------------------------------------------------

imaging_registry <- tribble(
  ~imaging_variable, ~imaging_label, ~stage, ~metric_family, ~analysis_role,
  "Anticipation_slope", "Anticipation overall slope", "Anticipation", "Overall slope", "PRIMARY_CANDIDATE",
  "Outcome_slope", "Outcome overall slope", "Outcome", "Overall slope", "PRIMARY_CANDIDATE",
  "Anticipation_similarity", "Anticipation similarity", "Anticipation", "Similarity", "SECONDARY_CANDIDATE",
  "Outcome_similarity", "Outcome similarity", "Outcome", "Similarity", "SECONDARY_CANDIDATE",
  "SMA_slope", "SMA Anticipation slope", "Anticipation", "ROI slope", "EXPLORATORY",
  "IFG_slope", "IFG Outcome slope", "Outcome", "ROI slope", "EXPLORATORY",
  "Cingulate_slope", "Cingulate Outcome slope", "Outcome", "ROI slope", "EXPLORATORY"
) %>% mutate(imaging_order = row_number())

eligible_scales <- scale_registry %>% filter(imaging_eligible)

imaging_associations <- map_dfr(seq_len(nrow(eligible_scales)), function(i) {
  reg <- eligible_scales[i, ]
  map_dfr(seq_len(nrow(imaging_registry)), function(j) {
    img <- imaging_registry[j, ]
    if (reg$population == "BOTH") {
      res <- robust_scale_effect(analysis_data, reg$variable, img$imaging_variable)
      method <- "STANDARDIZED_OLS_HC3_ADJUSTED_GROUP_AGE_SEX_MEANFD"
      population <- "HC+SZ"
      effect_type <- "standardized_beta"
    } else {
      res <- spearman_effect(
        analysis_data %>% filter(Group == "SZ"),
        reg$variable,
        img$imaging_variable,
        reps = 1000,
        seed = 20263000 + i * 100 + j
      )
      method <- "SPEARMAN_WITH_SUBJECT_BOOTSTRAP_CI"
      population <- "SZ"
      effect_type <- "spearman_rho"
    }
    bind_cols(
      tibble(
        scale_variable = reg$variable,
        scale_label = reg$label,
        scale_status = reg$status,
        scale_order = reg$order,
        population = population,
        imaging_variable = img$imaging_variable,
        imaging_label = img$imaging_label,
        imaging_order = img$imaging_order,
        stage = img$stage,
        metric_family = img$metric_family,
        analysis_role = img$analysis_role,
        method = method,
        effect_type = effect_type
      ),
      res
    )
  })
}) %>%
  mutate(
    prespecified_family = case_when(
      (scale_variable == "TEPS_anticipatory" & imaging_variable == "Anticipation_slope") |
        (scale_variable == "TEPS_consummatory" & imaging_variable == "Outcome_slope") |
        (scale_variable == "ACIPS_anticipatory" & imaging_variable == "Anticipation_slope") |
        (scale_variable == "ACIPS_consummatory" & imaging_variable == "Outcome_slope") ~ "A_OVERALL_SLOPE_4",
      (scale_variable == "TEPS_anticipatory" & imaging_variable == "Anticipation_similarity") |
        (scale_variable == "TEPS_consummatory" & imaging_variable == "Outcome_similarity") ~ "B_TEPS_SIMILARITY_2",
      (scale_variable == "ACIPS_anticipatory" & imaging_variable == "Anticipation_similarity") |
        (scale_variable == "ACIPS_consummatory" & imaging_variable == "Outcome_similarity") ~ "B_ACIPS_SIMILARITY_2",
      scale_variable == "PANSS_negative" & imaging_variable %in% c("Anticipation_slope", "Outcome_slope") ~ "B_PANSS_NEGATIVE_SLOPE_2",
      scale_variable == "SANS" & imaging_variable %in% c("Anticipation_slope", "Outcome_slope", "Anticipation_similarity", "Outcome_similarity") ~ "B_SANS_OVERALL_4",
      TRUE ~ "FULL_SCREEN_EXPLORATORY"
    ),
    p_adj_bh_global = stats::p.adjust(p, method = "BH"),
    p_adj_bh_population = ave(p, population, FUN = function(z) stats::p.adjust(z, method = "BH")),
    p_adj_prespecified = NA_real_
  )

for (fam in setdiff(unique(imaging_associations$prespecified_family), "FULL_SCREEN_EXPLORATORY")) {
  idx <- which(imaging_associations$prespecified_family == fam & is.finite(imaging_associations$p))
  imaging_associations$p_adj_prespecified[idx] <- stats::p.adjust(imaging_associations$p[idx], method = "holm")
}

stopifnot(nrow(imaging_associations) == 17 * 7)
write_tsv_safe(imaging_associations, "ELIGIBLE_SCALE_IMAGING_ASSOCIATIONS.tsv")

plot_imaging_heatmap <- function(population_id, filename, title, subtitle, width_mm, height_mm) {
  d <- imaging_associations %>%
    filter(population == population_id) %>%
    mutate(
      scale_label = factor(scale_label, levels = rev(unique(scale_label[order(scale_order)]))),
      imaging_label = factor(imaging_label, levels = imaging_registry$imaging_label),
      cell_label = if_else(is.finite(effect), sprintf("%.2f", effect), "NA"),
      global_sig = is.finite(p_adj_bh_global) & p_adj_bh_global < 0.05,
      planned_sig = is.finite(p_adj_prespecified) & p_adj_prespecified < 0.05
    )
  p <- ggplot(d, aes(x = imaging_label, y = scale_label, fill = effect)) +
    geom_tile(linewidth = 0.25, colour = "white") +
    geom_text(aes(label = cell_label), size = 2.15, colour = "black") +
    geom_point(data = d %>% filter(global_sig), shape = 16, size = 0.75, colour = "black", position = position_nudge(x = 0.32, y = 0.32)) +
    geom_point(data = d %>% filter(planned_sig), shape = 0, size = 2.0, stroke = 0.55, colour = "black", position = position_nudge(x = 0.32, y = 0.32)) +
    scale_fill_gradient2(
      low = "#3B6FB6",
      mid = "white",
      high = "#C9574D",
      midpoint = 0,
      limits = c(-1, 1),
      na.value = "#EEEEEE",
      name = if_else(population_id == "HC+SZ", "Adjusted\nbeta", "Spearman\nrho")
    ) +
    labs(title = title, subtitle = subtitle, x = NULL, y = NULL) +
    theme_nature_contract(base_size = 6.7) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 5.5),
      axis.text.y = element_text(size = 5.8),
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      legend.position = "right"
    )
  save_png(p, filename, width_mm = width_mm, height_mm = height_mm, dpi = 300)
}

plot_imaging_heatmap(
  "HC+SZ",
  "Fig05A_pleasure_scale_imaging_associations.png",
  "Fig. 5a | Pleasure-scale associations with all frozen imaging phenotypes",
  "Standardized HC3 coefficients adjusted for Group, Age, Sex and MeanFD.\nDots: global BH-FDR < 0.05; boxes: Holm-significant prespecified endpoints.",
  195, 112
)
plot_imaging_heatmap(
  "SZ",
  "Fig05B_sz_clinical_imaging_associations.png",
  "Fig. 5b | SZ symptom and motor associations with all frozen imaging phenotypes",
  "Spearman rho with participant-bootstrap CIs.\nDots and boxes use the same correction conventions as Fig. 5a.",
  195, 100
)

make_prespecified_scatter_data <- function(row_index) {
  r <- imaging_associations[row_index, ]
  if (r$population == "HC+SZ") {
    d <- analysis_data %>%
      transmute(
        Subject_ID,
        Group,
        scale_value = .data[[r$scale_variable]],
        imaging_value = .data[[r$imaging_variable]],
        Age,
        Sex = factor(Sex),
        MeanFD
      ) %>%
      filter(complete.cases(.)) %>%
      mutate(
        scale_z = zscore(scale_value),
        imaging_z = zscore(imaging_value),
        age_z = zscore(Age),
        fd_z = zscore(MeanFD)
      )
    d$plot_x <- residuals(lm(scale_z ~ Group + age_z + Sex + fd_z, data = d))
    d$plot_y <- residuals(lm(imaging_z ~ Group + age_z + Sex + fd_z, data = d))
    x_axis <- "Covariate-residualized scale (z)"
    y_axis <- "Covariate-residualized imaging (z)"
  } else {
    d <- analysis_data %>%
      filter(Group == "SZ") %>%
      transmute(
        Subject_ID,
        Group,
        scale_value = .data[[r$scale_variable]],
        imaging_value = .data[[r$imaging_variable]]
      ) %>%
      filter(complete.cases(.)) %>%
      mutate(
        plot_x = zscore(rank(scale_value, ties.method = "average")),
        plot_y = zscore(rank(imaging_value, ties.method = "average"))
      )
    x_axis <- "Scale rank (z)"
    y_axis <- "Imaging rank (z)"
  }
  planned_q <- if (is.finite(r$p_adj_prespecified)) sprintf("; Holm p=%.3g", r$p_adj_prespecified) else ""
  d %>%
    mutate(
      panel = paste0(r$scale_label, " x ", r$imaging_label, "\n", r$effect_type, "=", sprintf("%.2f", r$effect), planned_q),
      x_axis = x_axis,
      y_axis = y_axis
    )
}

prespecified_rows <- which(imaging_associations$prespecified_family != "FULL_SCREEN_EXPLORATORY")
scatter_data <- map_dfr(prespecified_rows, make_prespecified_scatter_data)

p_scatter <- ggplot(scatter_data, aes(x = plot_x, y = plot_y, colour = Group)) +
  geom_hline(yintercept = 0, linewidth = 0.25, colour = "#E5E5E5") +
  geom_vline(xintercept = 0, linewidth = 0.25, colour = "#E5E5E5") +
  geom_point(size = 1.0, alpha = 0.75) +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE, linewidth = 0.55, alpha = 0.12) +
  facet_wrap(~panel, scales = "free", ncol = 4) +
  scale_colour_manual(values = palette_contract[c("HC", "SZ")], drop = FALSE) +
  labs(
    title = "Fig. 5c | Prespecified association atlas",
    subtitle = "Combined-sample panels show covariate-residualized values; SZ-only panels show standardized ranks used to visualize Spearman effects.",
    x = "Residualized value or rank (z)",
    y = "Residualized value or rank (z)",
    colour = "Group"
  ) +
  theme_nature_contract(base_size = 6.5) +
  theme(
    legend.position = "top",
    strip.text = element_text(size = 5.2),
    axis.text = element_text(size = 5.2),
    panel.spacing = grid::unit(2.4, "mm")
  )
save_png(p_scatter, "Fig05C_prespecified_association_scatter_atlas.png", width_mm = 235, height_mm = 215, dpi = 300)

# -----------------------------------------------------------------------------
# Part 4: stage-congruent versus cross-stage coupling
# -----------------------------------------------------------------------------

stage_scale_registry <- tribble(
  ~scale_family, ~behavior_stage, ~scale_variable, ~scale_label,
  "TEPS", "Anticipation", "TEPS_anticipatory", "TEPS anticipatory",
  "TEPS", "Outcome", "TEPS_consummatory", "TEPS consummatory",
  "ACIPS", "Anticipation", "ACIPS_anticipatory", "ACIPS anticipatory",
  "ACIPS", "Outcome", "ACIPS_consummatory", "ACIPS consummatory"
)

stage_imaging_registry <- tribble(
  ~imaging_family, ~imaging_stage, ~imaging_variable, ~imaging_label,
  "Overall slope", "Anticipation", "Anticipation_slope", "Anticipation overall slope",
  "Overall slope", "Outcome", "Outcome_slope", "Outcome overall slope",
  "Similarity", "Anticipation", "Anticipation_similarity", "Anticipation similarity",
  "Similarity", "Outcome", "Outcome_similarity", "Outcome similarity"
)

stage_cell_analysis <- function(scale_family_id, imaging_family_id, population_id) {
  sreg <- stage_scale_registry %>% filter(scale_family == scale_family_id)
  ireg <- stage_imaging_registry %>% filter(imaging_family == imaging_family_id)
  required <- c(sreg$scale_variable, ireg$imaging_variable)
  if (population_id == "ALL") required <- c(required, "Group", "Age", "Sex", "MeanFD")
  d <- analysis_data
  if (population_id == "SZ") d <- d %>% filter(Group == "SZ")
  d <- d %>% filter(complete.cases(across(all_of(required))))
  cells <- tidyr::crossing(
    behavior_stage = c("Anticipation", "Outcome"),
    imaging_stage = c("Anticipation", "Outcome")
  )
  map_dfr(seq_len(nrow(cells)), function(i) {
    b_stage <- cells$behavior_stage[i]
    i_stage <- cells$imaging_stage[i]
    x <- sreg$scale_variable[sreg$behavior_stage == b_stage]
    y <- ireg$imaging_variable[ireg$imaging_stage == i_stage]
    if (population_id == "ALL") {
      res <- robust_scale_effect(d, x, y)
      method <- "STANDARDIZED_OLS_HC3_COMMON_SAMPLE"
      population_label <- "HC+SZ"
    } else {
      res <- spearman_effect(d, x, y, reps = 1000, seed = 20264000 + i)
      method <- "SPEARMAN_COMMON_SAMPLE"
      population_label <- "SZ"
    }
    bind_cols(
      tibble(
        scale_family = scale_family_id,
        imaging_family = imaging_family_id,
        population = population_label,
        behavior_stage = b_stage,
        imaging_stage = i_stage,
        scale_variable = x,
        imaging_variable = y,
        method = method,
        common_sample_n = nrow(d),
        stage_congruent = b_stage == i_stage
      ),
      res
    )
  })
}

stage_coupling_cells <- tidyr::crossing(
  scale_family = c("TEPS", "ACIPS"),
  imaging_family = c("Overall slope", "Similarity"),
  population = c("ALL", "SZ")
) %>%
  pmap_dfr(~stage_cell_analysis(..1, ..2, ..3)) %>%
  mutate(p_adj_bh_32cells = stats::p.adjust(p, method = "BH"))

stopifnot(nrow(stage_coupling_cells) == 32)
write_tsv_safe(stage_coupling_cells, "STAGE_COUPLING_CELL_ASSOCIATIONS.tsv")

stage_coefficient_point <- function(d, x, y, population_id) {
  if (population_id == "SZ") {
    if (nrow(d) < 8 || n_distinct(d[[x]]) < 3 || n_distinct(d[[y]]) < 3) return(NA_real_)
    return(suppressWarnings(stats::cor(d[[x]], d[[y]], method = "spearman")))
  }
  dd <- d %>%
    transmute(
      x = .data[[x]],
      y = .data[[y]],
      Group = factor(Group, levels = c("HC", "SZ")),
      Age,
      Sex = factor(Sex),
      MeanFD
    ) %>%
    filter(complete.cases(.))
  if (nrow(dd) < 12 || n_distinct(dd$Group) < 2 || n_distinct(dd$Sex) < 2) return(NA_real_)
  dd <- dd %>% mutate(x_z = zscore(x), y_z = zscore(y), age_z = zscore(Age), fd_z = zscore(MeanFD))
  fit <- tryCatch(lm(y_z ~ x_z + Group + age_z + Sex + fd_z, data = dd), error = function(e) NULL)
  if (is.null(fit) || !("x_z" %in% names(coef(fit)))) return(NA_real_)
  unname(coef(fit)["x_z"])
}

stage_congruence_bootstrap <- function(scale_family_id, imaging_family_id, population_id, reps = 1000, seed = 20265000) {
  sreg <- stage_scale_registry %>% filter(scale_family == scale_family_id)
  ireg <- stage_imaging_registry %>% filter(imaging_family == imaging_family_id)
  ant_scale <- sreg$scale_variable[sreg$behavior_stage == "Anticipation"]
  out_scale <- sreg$scale_variable[sreg$behavior_stage == "Outcome"]
  ant_img <- ireg$imaging_variable[ireg$imaging_stage == "Anticipation"]
  out_img <- ireg$imaging_variable[ireg$imaging_stage == "Outcome"]
  required <- c(ant_scale, out_scale, ant_img, out_img)
  if (population_id == "ALL") required <- c(required, "Group", "Age", "Sex", "MeanFD")
  d <- analysis_data
  if (population_id == "SZ") d <- d %>% filter(Group == "SZ")
  d <- d %>% filter(complete.cases(across(all_of(required))))
  get_contrast <- function(dd) {
    aa <- stage_coefficient_point(dd, ant_scale, ant_img, population_id)
    cc <- stage_coefficient_point(dd, out_scale, out_img, population_id)
    ac <- stage_coefficient_point(dd, ant_scale, out_img, population_id)
    ca <- stage_coefficient_point(dd, out_scale, ant_img, population_id)
    mean(c(aa, cc)) - mean(c(ac, ca))
  }
  point <- get_contrast(d)
  set.seed(seed)
  boots <- replicate(reps, {
    idx <- sample.int(nrow(d), size = nrow(d), replace = TRUE)
    get_contrast(d[idx, , drop = FALSE])
  })
  boots <- boots[is.finite(boots)]
  p_boot <- if (length(boots) >= 200) 2 * min(mean(boots <= 0), mean(boots >= 0)) else NA_real_
  tibble(
    scale_family = scale_family_id,
    imaging_family = imaging_family_id,
    population = if_else(population_id == "ALL", "HC+SZ", "SZ"),
    n = nrow(d),
    congruence_contrast = point,
    ci_low = if (length(boots) >= 200) unname(quantile(boots, 0.025)) else NA_real_,
    ci_high = if (length(boots) >= 200) unname(quantile(boots, 0.975)) else NA_real_,
    bootstrap_p = p_boot,
    valid_boot = length(boots),
    definition = "mean(congruent effects) - mean(cross-stage effects)"
  )
}

stage_congruence_contrasts <- tidyr::crossing(
  scale_family = c("TEPS", "ACIPS"),
  imaging_family = c("Overall slope", "Similarity"),
  population = c("ALL", "SZ")
) %>%
  mutate(row_seed = 20265000 + row_number()) %>%
  pmap_dfr(~stage_congruence_bootstrap(..1, ..2, ..3, reps = 1000, seed = ..4)) %>%
  mutate(p_adj_bh_8contrasts = stats::p.adjust(bootstrap_p, method = "BH"))

stopifnot(nrow(stage_congruence_contrasts) == 8)
write_tsv_safe(stage_congruence_contrasts, "STAGE_CONGRUENCE_CONTRASTS.tsv")

stage_plot_data <- stage_coupling_cells %>%
  mutate(
    behavior_stage = factor(behavior_stage, levels = c("Outcome", "Anticipation")),
    imaging_stage = factor(imaging_stage, levels = c("Anticipation", "Outcome")),
    panel_row = factor(paste(population, scale_family, sep = " | "), levels = c("HC+SZ | TEPS", "HC+SZ | ACIPS", "SZ | TEPS", "SZ | ACIPS")),
    cell_label = if_else(is.finite(effect), sprintf("%.2f\nn=%d", effect, common_sample_n), "NA")
  )

p_stage_cells <- ggplot(stage_plot_data, aes(x = imaging_stage, y = behavior_stage, fill = effect)) +
  geom_tile(linewidth = 0.5, colour = "white") +
  geom_text(aes(label = cell_label), size = 2.45) +
  facet_grid(panel_row ~ imaging_family) +
  scale_fill_gradient2(low = "#3B6FB6", mid = "white", high = "#C9574D", midpoint = 0, limits = c(-1, 1), name = "Effect") +
  labs(
    title = "Fig. 6a | Complete 2 x 2 stage-coupling matrices",
    subtitle = "Diagonal cells are stage-congruent; off-diagonal cells are cross-stage. Each 2 x 2 panel uses one common complete-case sample.",
    x = "Imaging stage",
    y = "Scale dimension"
  ) +
  theme_nature_contract(base_size = 7) +
  theme(
    axis.text.x = element_text(angle = 25, hjust = 1),
    strip.text.y = element_text(angle = 0),
    panel.spacing = grid::unit(3.5, "mm")
  )
save_png(p_stage_cells, "Fig06A_stage_coupling_matrices.png", width_mm = 185, height_mm = 185, dpi = 300)

contrast_plot <- stage_congruence_contrasts %>%
  mutate(
    contrast_label = paste(population, scale_family, imaging_family, sep = " | "),
    contrast_label = factor(contrast_label, levels = rev(contrast_label)),
    significant = is.finite(p_adj_bh_8contrasts) & p_adj_bh_8contrasts < 0.05
  )

p_contrast <- ggplot(contrast_plot, aes(x = congruence_contrast, y = contrast_label, colour = scale_family, shape = population)) +
  geom_vline(xintercept = 0, linewidth = 0.4, linetype = "dashed", colour = palette_contract[["neutral_mid"]]) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = 0, linewidth = 0.5, na.rm = TRUE) +
  geom_point(size = 2.0, na.rm = TRUE) +
  geom_text(data = contrast_plot %>% filter(significant), aes(label = "*"), nudge_y = 0.24, show.legend = FALSE, size = 3) +
  scale_colour_manual(values = c(TEPS = palette_contract[["anticipation"]], ACIPS = palette_contract[["outcome"]])) +
  scale_shape_manual(values = c("HC+SZ" = 16, "SZ" = 17)) +
  labs(
    title = "Fig. 6b | Direct test of stage congruence",
    subtitle = "Positive values mean that the average stage-matched association exceeds the average cross-stage association; intervals are participant-bootstrap 95% CIs.",
    x = "Congruent minus cross-stage effect",
    y = NULL,
    colour = "Scale",
    shape = "Population"
  ) +
  theme_nature_contract(base_size = 7) +
  theme(legend.position = "top")
save_png(p_contrast, "Fig06B_stage_congruence_contrasts.png", width_mm = 180, height_mm = 105, dpi = 300)

# -----------------------------------------------------------------------------
# Part 5: HC/SZ imaging differences, clinical distributions, and integration
# -----------------------------------------------------------------------------

imaging_group_effects <- map_dfr(seq_len(nrow(imaging_registry)), function(i) {
  img <- imaging_registry[i, ]
  robust_group_effect(analysis_data, img$imaging_variable) %>%
    mutate(
      imaging_variable = img$imaging_variable,
      imaging_label = img$imaging_label,
      imaging_order = img$imaging_order,
      stage = img$stage,
      metric_family = img$metric_family,
      .before = 1
    )
}) %>%
  mutate(p_adj_holm_7 = stats::p.adjust(p, method = "holm"))

write_tsv_safe(imaging_group_effects, "HC_SZ_IMAGING_GROUP_EFFECTS.tsv")

group_scale_registry <- scale_registry %>% filter(group_eligible)
scale_group_effects <- map_dfr(seq_len(nrow(group_scale_registry)), function(i) {
  reg <- group_scale_registry[i, ]
  robust_group_effect(analysis_data, reg$variable) %>%
    mutate(
      scale_variable = reg$variable,
      scale_label = reg$label,
      scale_order = reg$order,
      scale_status = reg$status,
      .before = 1
    )
}) %>%
  mutate(p_adj_holm_10 = stats::p.adjust(p, method = "holm"))

write_tsv_safe(scale_group_effects, "HC_SZ_ELIGIBLE_SCALE_GROUP_EFFECTS.tsv")

imaging_long <- analysis_data %>%
  select(Subject_ID, Group, all_of(imaging_registry$imaging_variable)) %>%
  pivot_longer(cols = all_of(imaging_registry$imaging_variable), names_to = "imaging_variable", values_to = "value") %>%
  left_join(imaging_registry, by = "imaging_variable") %>%
  mutate(
    imaging_label = factor(imaging_label, levels = imaging_registry$imaging_label),
    Group = factor(Group, levels = c("HC", "SZ"))
  )

p_img_dist <- ggplot(imaging_long, aes(x = Group, y = value, colour = Group, fill = Group)) +
  geom_violin(alpha = 0.22, linewidth = 0.35, trim = FALSE, scale = "width") +
  geom_boxplot(width = 0.18, alpha = 0.65, linewidth = 0.35, outlier.shape = NA) +
  geom_point(position = position_jitter(width = 0.08), size = 0.7, alpha = 0.55, stroke = 0) +
  facet_wrap(~imaging_label, scales = "free_y", ncol = 4) +
  scale_colour_manual(values = palette_contract[c("HC", "SZ")]) +
  scale_fill_manual(values = palette_contract[c("HC", "SZ")]) +
  labs(
    title = "Fig. 7a | HC/SZ distributions of all frozen imaging phenotypes",
    subtitle = "Dots are participants; boxes show median and IQR. Formal inference is the covariate-adjusted HC3 model in Fig. 7b.",
    x = NULL,
    y = "Frozen imaging value",
    colour = "Group",
    fill = "Group"
  ) +
  theme_nature_contract(base_size = 7) +
  theme(legend.position = "top", strip.text = element_text(size = 6.1))
save_png(p_img_dist, "Fig07A_imaging_group_distributions.png", width_mm = 195, height_mm = 125, dpi = 300)

img_forest_data <- imaging_group_effects %>%
  mutate(
    imaging_label = factor(imaging_label, levels = rev(imaging_registry$imaging_label)),
    significant = is.finite(p_adj_holm_7) & p_adj_holm_7 < 0.05
  )

p_img_forest <- ggplot(img_forest_data, aes(x = effect, y = imaging_label, colour = stage)) +
  geom_vline(xintercept = 0, linewidth = 0.4, linetype = "dashed", colour = palette_contract[["neutral_mid"]]) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = 0, linewidth = 0.55) +
  geom_point(size = 2.0) +
  geom_text(aes(label = if_else(significant, "Holm *", "")), nudge_y = 0.24, size = 2.4, show.legend = FALSE) +
  scale_colour_manual(values = c(Anticipation = palette_contract[["anticipation"]], Outcome = palette_contract[["outcome"]])) +
  labs(
    title = "Fig. 7b | Adjusted HC/SZ imaging differences",
    subtitle = "Positive standardized coefficients indicate higher values in SZ; models adjust for Age, Sex and MeanFD with HC3 uncertainty.",
    x = "Adjusted group effect (SZ - HC, standardized)",
    y = NULL,
    colour = "Stage"
  ) +
  theme_nature_contract(base_size = 7) +
  theme(legend.position = "top")
save_png(p_img_forest, "Fig07B_imaging_group_effects.png", width_mm = 180, height_mm = 105, dpi = 300)

scale_forest_data <- scale_group_effects %>%
  mutate(
    scale_label = factor(scale_label, levels = rev(scale_label[order(scale_order)])),
    significant = is.finite(p_adj_holm_10) & p_adj_holm_10 < 0.05,
    scale_family = if_else(str_starts(scale_variable, "TEPS"), "TEPS", "ACIPS")
  )

p_scale_forest <- ggplot(scale_forest_data, aes(x = effect, y = scale_label, colour = scale_family)) +
  geom_vline(xintercept = 0, linewidth = 0.4, linetype = "dashed", colour = palette_contract[["neutral_mid"]]) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = 0, linewidth = 0.55) +
  geom_point(size = 2.0) +
  geom_text(aes(label = if_else(significant, "Holm *", "")), nudge_y = 0.25, size = 2.4, show.legend = FALSE) +
  scale_colour_manual(values = c(TEPS = palette_contract[["anticipation"]], ACIPS = palette_contract[["outcome"]])) +
  labs(
    title = "Fig. 7c | Adjusted HC/SZ differences in eligible pleasure scales",
    subtitle = "Positive standardized coefficients indicate higher scores in SZ; models adjust for Age, Sex and MeanFD.",
    x = "Adjusted group effect (SZ - HC, standardized)",
    y = NULL,
    colour = "Scale"
  ) +
  theme_nature_contract(base_size = 7) +
  theme(legend.position = "top")
save_png(p_scale_forest, "Fig07C_eligible_scale_group_effects.png", width_mm = 185, height_mm = 125, dpi = 300)

core_assoc <- imaging_associations %>%
  filter(
    population == "HC+SZ",
    scale_variable %in% c("TEPS_anticipatory", "TEPS_consummatory", "ACIPS_anticipatory", "ACIPS_consummatory"),
    imaging_variable %in% c("Anticipation_slope", "Outcome_slope", "Anticipation_similarity", "Outcome_similarity")
  ) %>%
  mutate(
    scale_label = factor(scale_label, levels = rev(c("TEPS anticipatory", "TEPS consummatory", "ACIPS anticipatory", "ACIPS consummatory"))),
    imaging_label = factor(imaging_label, levels = imaging_registry$imaging_label[1:4]),
    cell_label = sprintf("%.2f", effect)
  )

p_core_assoc <- ggplot(core_assoc, aes(x = imaging_label, y = scale_label, fill = effect)) +
  geom_tile(linewidth = 0.35, colour = "white") +
  geom_text(aes(label = cell_label), size = 2.5) +
  scale_fill_gradient2(low = "#3B6FB6", mid = "white", high = "#C9574D", midpoint = 0, limits = c(-1, 1), name = "Adjusted\nbeta") +
  labs(title = "Core scale-imaging effects", x = NULL, y = NULL) +
  theme_nature_contract(base_size = 6.5) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1), axis.line = element_blank(), axis.ticks = element_blank())

p_img_integrated <- p_img_forest + labs(title = "Imaging group effects") + theme(plot.subtitle = element_blank())
p_scale_integrated <- p_scale_forest + labs(title = "Pleasure-scale group effects") + theme(plot.subtitle = element_blank())

integrated_figure <- (p_img_integrated | p_scale_integrated) / p_core_assoc +
  plot_layout(heights = c(1.15, 0.85), guides = "collect") +
  plot_annotation(
    title = "Fig. 8 | Integrated group, clinical and brain-scale evidence",
    subtitle = "Top: adjusted SZ-HC effects. Bottom: covariate-adjusted core pleasure-scale associations with stage-specific imaging.",
    tag_levels = "a"
  ) &
  theme(plot.tag = element_text(size = 8, face = "bold"), legend.position = "right")

save_png(integrated_figure, "Fig08_integrated_evidence_map.png", width_mm = 235, height_mm = 190, dpi = 300)

# -----------------------------------------------------------------------------
# Part 6: focused validation of TEPS consummatory x Anticipation similarity
# -----------------------------------------------------------------------------

validation_endpoint_registry <- tribble(
  ~endpoint_id, ~endpoint_label, ~scale_variable, ~imaging_variable, ~priority,
  "TEPS_CONS", "TEPS consummatory", "TEPS_consummatory", "Anticipation_similarity", "PRIMARY_CORE_SCALE",
  "TEPS_CONCRETE_CONS", "TEPS concrete consummatory", "TEPS_concrete_consummatory", "Anticipation_similarity", "SECONDARY_STRONGEST_GLOBAL_FDR"
) %>% mutate(endpoint_order = row_number())

fit_group_interaction <- function(data, x, y) {
  d <- data %>%
    transmute(
      Subject_ID,
      Group = factor(Group, levels = c("HC", "SZ")),
      x = .data[[x]],
      y = .data[[y]],
      Age,
      Sex = factor(Sex),
      MeanFD
    ) %>%
    filter(complete.cases(.)) %>%
    mutate(
      x_z = zscore(x),
      y_z = zscore(y),
      age_z = zscore(Age),
      fd_z = zscore(MeanFD)
    )
  fit <- stats::lm(y_z ~ x_z * Group + age_z + Sex + fd_z, data = d)
  v <- sandwich::vcovHC(fit, type = "HC3")
  int_term <- intersect(c("x_z:GroupSZ", "GroupSZ:x_z"), names(coef(fit)))
  if (length(int_term) != 1) stop("Interaction term not found for ", x, " x ", y)
  int_term <- int_term[[1]]
  b <- coef(fit)
  beta_hc <- unname(b[["x_z"]])
  beta_int <- unname(b[[int_term]])
  beta_sz <- beta_hc + beta_int
  se_hc <- sqrt(v["x_z", "x_z"])
  se_int <- sqrt(v[int_term, int_term])
  se_sz <- sqrt(v["x_z", "x_z"] + v[int_term, int_term] + 2 * v["x_z", int_term])
  summary <- tibble(
    n = nrow(d),
    n_hc = sum(d$Group == "HC"),
    n_sz = sum(d$Group == "SZ"),
    beta_hc = beta_hc,
    se_hc = se_hc,
    ci_low_hc = beta_hc - 1.96 * se_hc,
    ci_high_hc = beta_hc + 1.96 * se_hc,
    p_hc = 2 * pnorm(abs(beta_hc / se_hc), lower.tail = FALSE),
    beta_sz = beta_sz,
    se_sz = se_sz,
    ci_low_sz = beta_sz - 1.96 * se_sz,
    ci_high_sz = beta_sz + 1.96 * se_sz,
    p_sz = 2 * pnorm(abs(beta_sz / se_sz), lower.tail = FALSE),
    beta_interaction = beta_int,
    se_interaction = se_int,
    ci_low_interaction = beta_int - 1.96 * se_int,
    ci_high_interaction = beta_int + 1.96 * se_int,
    p_interaction = 2 * pnorm(abs(beta_int / se_int), lower.tail = FALSE),
    method = "STANDARDIZED_OLS_GROUP_X_TEPS_HC3_ADJUSTED_AGE_SEX_MEANFD"
  )
  list(data = d, fit = fit, vcov = v, interaction_term = int_term, summary = summary)
}

validation_fits <- map(seq_len(nrow(validation_endpoint_registry)), function(i) {
  e <- validation_endpoint_registry[i, ]
  fit_group_interaction(analysis_data, e$scale_variable, e$imaging_variable)
})

teps_interaction_results <- map_dfr(seq_along(validation_fits), function(i) {
  bind_cols(validation_endpoint_registry[i, ], validation_fits[[i]]$summary)
}) %>%
  mutate(p_adj_holm_2_interactions = p.adjust(p_interaction, method = "holm"))

write_tsv_safe(teps_interaction_results, "TEPS_GROUP_INTERACTION_VALIDATION.tsv")

teps_stratified_correlations <- map_dfr(seq_len(nrow(validation_endpoint_registry)), function(i) {
  e <- validation_endpoint_registry[i, ]
  map_dfr(c("HC", "SZ"), function(g) {
    res <- spearman_effect(
      analysis_data %>% filter(Group == g),
      e$scale_variable,
      e$imaging_variable,
      reps = 10000,
      seed = 20267000 + i * 10 + match(g, c("HC", "SZ"))
    )
    bind_cols(e, tibble(Group = g), res)
  })
}) %>%
  mutate(p_adj_holm_4_stratified = p.adjust(p, method = "holm"))

write_tsv_safe(teps_stratified_correlations, "TEPS_HC_SZ_STRATIFIED_CORRELATIONS.tsv")

bootstrap_stratified_rho_difference <- function(data, x, y, reps = 10000, seed = 20268000) {
  hc <- data %>% filter(Group == "HC") %>% select(x = all_of(x), y = all_of(y)) %>% filter(complete.cases(.))
  sz <- data %>% filter(Group == "SZ") %>% select(x = all_of(x), y = all_of(y)) %>% filter(complete.cases(.))
  rho_hc <- cor(hc$x, hc$y, method = "spearman")
  rho_sz <- cor(sz$x, sz$y, method = "spearman")
  point <- rho_sz - rho_hc
  set.seed(seed)
  boot_diff <- replicate(reps, {
    ih <- sample.int(nrow(hc), nrow(hc), replace = TRUE)
    is <- sample.int(nrow(sz), nrow(sz), replace = TRUE)
    cor(sz$x[is], sz$y[is], method = "spearman") - cor(hc$x[ih], hc$y[ih], method = "spearman")
  })
  boot_diff <- boot_diff[is.finite(boot_diff)]
  tibble(
    n_hc = nrow(hc),
    n_sz = nrow(sz),
    rho_hc = rho_hc,
    rho_sz = rho_sz,
    rho_difference_sz_minus_hc = point,
    ci_low = unname(quantile(boot_diff, 0.025)),
    ci_high = unname(quantile(boot_diff, 0.975)),
    bootstrap_p = 2 * min(mean(boot_diff <= 0), mean(boot_diff >= 0)),
    valid_boot = length(boot_diff)
  )
}

teps_stratified_differences <- map_dfr(seq_len(nrow(validation_endpoint_registry)), function(i) {
  e <- validation_endpoint_registry[i, ]
  res <- bootstrap_stratified_rho_difference(
    analysis_data,
    e$scale_variable,
    e$imaging_variable,
    reps = 10000,
    seed = 20268000 + i
  )
  bind_cols(e, res)
}) %>%
  mutate(p_adj_holm_2_differences = p.adjust(bootstrap_p, method = "holm"))

write_tsv_safe(teps_stratified_differences, "TEPS_HC_SZ_CORRELATION_DIFFERENCES.tsv")

interaction_influence_diagnostics <- map_dfr(seq_along(validation_fits), function(i) {
  e <- validation_endpoint_registry[i, ]
  details <- validation_fits[[i]]
  fit <- details$fit
  d <- details$data
  int_term <- details$interaction_term
  cooks <- cooks.distance(fit)
  leverage <- hatvalues(fit)
  studentized <- rstudent(fit)
  dfb <- dfbetas(fit)
  p_terms <- length(coef(fit))
  n <- nrow(d)
  cook_threshold <- 4 / n
  leverage_threshold <- 2 * p_terms / n
  dfbeta_threshold <- 2 / sqrt(n)
  tibble(
    endpoint_id = e$endpoint_id,
    endpoint_label = e$endpoint_label,
    endpoint_order = e$endpoint_order,
    Subject_ID = d$Subject_ID,
    Group = as.character(d$Group),
    cooks_distance = as.numeric(cooks),
    leverage = as.numeric(leverage),
    studentized_residual = as.numeric(studentized),
    dfbeta_teps = as.numeric(dfb[, "x_z"]),
    dfbeta_interaction = as.numeric(dfb[, int_term]),
    cook_threshold = cook_threshold,
    leverage_threshold = leverage_threshold,
    studentized_threshold = 3,
    dfbeta_threshold = dfbeta_threshold,
    flag_cook = cooks > cook_threshold,
    flag_leverage = leverage > leverage_threshold,
    flag_studentized = abs(studentized) > 3,
    flag_dfbeta_teps = abs(dfb[, "x_z"]) > dfbeta_threshold,
    flag_dfbeta_interaction = abs(dfb[, int_term]) > dfbeta_threshold
  ) %>%
    mutate(any_influence_flag = flag_cook | flag_leverage | flag_studentized | flag_dfbeta_teps | flag_dfbeta_interaction)
})

write_tsv_safe(interaction_influence_diagnostics, "TEPS_INTERACTION_INFLUENCE_DIAGNOSTICS.tsv")

loo_combined <- map_dfr(seq_len(nrow(validation_endpoint_registry)), function(i) {
  e <- validation_endpoint_registry[i, ]
  full_ids <- validation_fits[[i]]$data$Subject_ID
  map_dfr(full_ids, function(omitted_id) {
    d_loo <- analysis_data %>% filter(Subject_ID != omitted_id)
    int_loo <- fit_group_interaction(d_loo, e$scale_variable, e$imaging_variable)$summary
    main_loo <- robust_scale_effect(d_loo, e$scale_variable, e$imaging_variable)
    omitted_group <- analysis_data$Group[match(omitted_id, analysis_data$Subject_ID)]
    bind_rows(
      tibble(
        endpoint_id = e$endpoint_id,
        endpoint_label = e$endpoint_label,
        endpoint_order = e$endpoint_order,
        omitted_subject = omitted_id,
        omitted_group = omitted_group,
        estimand = "Combined adjusted main effect",
        estimate = main_loo$effect,
        ci_low = main_loo$ci_low,
        ci_high = main_loo$ci_high,
        n = main_loo$n
      ),
      tibble(
        endpoint_id = e$endpoint_id,
        endpoint_label = e$endpoint_label,
        endpoint_order = e$endpoint_order,
        omitted_subject = omitted_id,
        omitted_group = omitted_group,
        estimand = "Group x TEPS interaction",
        estimate = int_loo$beta_interaction,
        ci_low = int_loo$ci_low_interaction,
        ci_high = int_loo$ci_high_interaction,
        n = int_loo$n
      )
    )
  })
})

loo_stratified <- map_dfr(seq_len(nrow(validation_endpoint_registry)), function(i) {
  e <- validation_endpoint_registry[i, ]
  map_dfr(c("HC", "SZ"), function(g) {
    d <- analysis_data %>%
      filter(Group == g) %>%
      select(Subject_ID, x = all_of(e$scale_variable), y = all_of(e$imaging_variable)) %>%
      filter(complete.cases(.))
    map_dfr(d$Subject_ID, function(omitted_id) {
      dd <- d %>% filter(Subject_ID != omitted_id)
      tibble(
        endpoint_id = e$endpoint_id,
        endpoint_label = e$endpoint_label,
        endpoint_order = e$endpoint_order,
        omitted_subject = omitted_id,
        omitted_group = g,
        estimand = paste0(g, " Spearman rho"),
        estimate = cor(dd$x, dd$y, method = "spearman"),
        ci_low = NA_real_,
        ci_high = NA_real_,
        n = nrow(dd)
      )
    })
  })
})

teps_leave_one_out <- bind_rows(loo_combined, loo_stratified) %>%
  mutate(subject_index = match(omitted_subject, sort(sample_matrix$Subject_ID)))

write_tsv_safe(teps_leave_one_out, "TEPS_LEAVE_ONE_OUT_SENSITIVITY.tsv")

loo_full_reference <- map_dfr(seq_len(nrow(validation_endpoint_registry)), function(i) {
  e <- validation_endpoint_registry[i, ]
  int_res <- teps_interaction_results %>% filter(endpoint_id == e$endpoint_id)
  main_res <- robust_scale_effect(analysis_data, e$scale_variable, e$imaging_variable)
  strat_res <- teps_stratified_correlations %>% filter(endpoint_id == e$endpoint_id)
  bind_rows(
    tibble(endpoint_id = e$endpoint_id, estimand = "Combined adjusted main effect", full_estimate = main_res$effect, full_ci_low = main_res$ci_low, full_ci_high = main_res$ci_high),
    tibble(endpoint_id = e$endpoint_id, estimand = "Group x TEPS interaction", full_estimate = int_res$beta_interaction, full_ci_low = int_res$ci_low_interaction, full_ci_high = int_res$ci_high_interaction),
    strat_res %>% transmute(endpoint_id, estimand = paste0(Group, " Spearman rho"), full_estimate = effect, full_ci_low = ci_low, full_ci_high = ci_high)
  )
})

influence_sensitivity_summary <- map_dfr(seq_len(nrow(validation_endpoint_registry)), function(i) {
  e <- validation_endpoint_registry[i, ]
  diag <- interaction_influence_diagnostics %>% filter(endpoint_id == e$endpoint_id)
  flagged_ids <- diag %>% filter(any_influence_flag) %>% pull(Subject_ID)
  d_sens <- analysis_data %>% filter(!Subject_ID %in% flagged_ids)
  int_sens <- fit_group_interaction(d_sens, e$scale_variable, e$imaging_variable)$summary
  main_sens <- robust_scale_effect(d_sens, e$scale_variable, e$imaging_variable)
  loo_e <- teps_leave_one_out %>% filter(endpoint_id == e$endpoint_id)
  full_main <- loo_full_reference %>% filter(endpoint_id == e$endpoint_id, estimand == "Combined adjusted main effect")
  full_int <- loo_full_reference %>% filter(endpoint_id == e$endpoint_id, estimand == "Group x TEPS interaction")
  tibble(
    endpoint_id = e$endpoint_id,
    endpoint_label = e$endpoint_label,
    n_full = validation_fits[[i]]$summary$n,
    n_any_influence_flag = length(flagged_ids),
    flagged_subjects = paste(flagged_ids, collapse = ";"),
    max_cook_subject = diag$Subject_ID[which.max(diag$cooks_distance)],
    max_cook_distance = max(diag$cooks_distance),
    full_main_effect = full_main$full_estimate,
    flagged_excluded_main_effect = main_sens$effect,
    loo_main_min = min(loo_e$estimate[loo_e$estimand == "Combined adjusted main effect"], na.rm = TRUE),
    loo_main_max = max(loo_e$estimate[loo_e$estimand == "Combined adjusted main effect"], na.rm = TRUE),
    full_interaction = full_int$full_estimate,
    flagged_excluded_interaction = int_sens$beta_interaction,
    loo_interaction_min = min(loo_e$estimate[loo_e$estimand == "Group x TEPS interaction"], na.rm = TRUE),
    loo_interaction_max = max(loo_e$estimate[loo_e$estimand == "Group x TEPS interaction"], na.rm = TRUE)
  )
})

write_tsv_safe(influence_sensitivity_summary, "TEPS_INFLUENCE_SENSITIVITY_SUMMARY.tsv")

interaction_panel_levels <- teps_interaction_results %>%
  arrange(endpoint_order) %>%
  transmute(panel_label = paste0(endpoint_label, "\ninteraction Holm p=", sprintf("%.3f", p_adj_holm_2_interactions))) %>%
  pull(panel_label)

interaction_forest_data <- teps_interaction_results %>%
  transmute(
    endpoint_id,
    endpoint_label,
    endpoint_order,
    `HC adjusted slope` = beta_hc,
    `HC adjusted slope_low` = ci_low_hc,
    `HC adjusted slope_high` = ci_high_hc,
    `SZ adjusted slope` = beta_sz,
    `SZ adjusted slope_low` = ci_low_sz,
    `SZ adjusted slope_high` = ci_high_sz,
    `Group x TEPS` = beta_interaction,
    `Group x TEPS_low` = ci_low_interaction,
    `Group x TEPS_high` = ci_high_interaction,
    interaction_q = p_adj_holm_2_interactions
  ) %>%
  pivot_longer(
    cols = c(`HC adjusted slope`, `SZ adjusted slope`, `Group x TEPS`),
    names_to = "estimand",
    values_to = "estimate"
  ) %>%
  mutate(
    ci_low = case_when(
      estimand == "HC adjusted slope" ~ `HC adjusted slope_low`,
      estimand == "SZ adjusted slope" ~ `SZ adjusted slope_low`,
      TRUE ~ `Group x TEPS_low`
    ),
    ci_high = case_when(
      estimand == "HC adjusted slope" ~ `HC adjusted slope_high`,
      estimand == "SZ adjusted slope" ~ `SZ adjusted slope_high`,
      TRUE ~ `Group x TEPS_high`
    ),
    panel_label = factor(
      paste0(endpoint_label, "\ninteraction Holm p=", sprintf("%.3f", interaction_q)),
      levels = interaction_panel_levels
    ),
    estimand = factor(estimand, levels = c("Group x TEPS", "SZ adjusted slope", "HC adjusted slope"))
  )

p_interaction <- ggplot(interaction_forest_data, aes(x = estimate, y = estimand, colour = estimand)) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, colour = palette_contract[["neutral_mid"]]) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = 0, linewidth = 0.55) +
  geom_point(size = 2.1) +
  facet_wrap(~panel_label, ncol = 1) +
  scale_colour_manual(values = c(
    "HC adjusted slope" = palette_contract[["HC"]],
    "SZ adjusted slope" = palette_contract[["SZ"]],
    "Group x TEPS" = palette_contract[["neutral_dark"]]
  )) +
  labs(
    title = "Fig. 9a | Group-specific slopes and Group x TEPS interaction",
    subtitle = "Standardized HC3 models adjust for Age, Sex and MeanFD. The interaction directly tests whether the TEPS slope differs by group.",
    x = "Standardized effect",
    y = NULL,
    colour = NULL
  ) +
  theme_nature_contract(base_size = 7) +
  theme(legend.position = "top")
save_png(p_interaction, "Fig09A_teps_group_interaction.png", width_mm = 185, height_mm = 135, dpi = 300)

validation_scatter_data <- map_dfr(seq_len(nrow(validation_endpoint_registry)), function(i) {
  e <- validation_endpoint_registry[i, ]
  analysis_data %>%
    transmute(
      Subject_ID,
      Group = factor(Group, levels = c("HC", "SZ")),
      endpoint_label = e$endpoint_label,
      teps_score = .data[[e$scale_variable]],
      anticipation_similarity = .data[[e$imaging_variable]]
    ) %>%
    filter(complete.cases(.))
})

scatter_annotations <- teps_stratified_correlations %>%
  mutate(
    endpoint_label = factor(endpoint_label, levels = validation_endpoint_registry$endpoint_label),
    annotation = paste0(Group, ": rho=", sprintf("%.2f", effect), " [", sprintf("%.2f", ci_low), ", ", sprintf("%.2f", ci_high), "]"),
    vjust_value = if_else(Group == "HC", 1.1, 2.5)
  )

p_validation_scatter <- ggplot(validation_scatter_data, aes(x = teps_score, y = anticipation_similarity, colour = Group)) +
  geom_point(size = 1.35, alpha = 0.75) +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE, linewidth = 0.65, alpha = 0.12) +
  geom_text(
    data = scatter_annotations,
    aes(x = -Inf, y = Inf, label = annotation, colour = Group, vjust = vjust_value),
    hjust = -0.03,
    size = 2.4,
    inherit.aes = FALSE,
    show.legend = FALSE
  ) +
  facet_wrap(~endpoint_label, scales = "free_x", ncol = 2) +
  scale_colour_manual(values = palette_contract[c("HC", "SZ")]) +
  labs(
    title = "Fig. 9b | HC and SZ stratified TEPS-similarity relationships",
    subtitle = "Lines are descriptive within-group linear fits; annotations report Spearman rho with 10,000-bootstrap 95% CIs.",
    x = "TEPS score",
    y = "Anticipation similarity",
    colour = "Group"
  ) +
  theme_nature_contract(base_size = 7) +
  theme(legend.position = "top")
save_png(p_validation_scatter, "Fig09B_teps_stratified_correlations.png", width_mm = 190, height_mm = 100, dpi = 300)

influence_plot_data <- interaction_influence_diagnostics %>%
  select(endpoint_label, endpoint_order, Subject_ID, Group, any_influence_flag,
         cooks_distance, leverage, studentized_residual, dfbeta_teps, dfbeta_interaction,
         cook_threshold, leverage_threshold, studentized_threshold, dfbeta_threshold) %>%
  pivot_longer(
    cols = c(cooks_distance, leverage, studentized_residual, dfbeta_teps, dfbeta_interaction),
    names_to = "diagnostic",
    values_to = "value"
  ) %>%
  mutate(
    threshold = case_when(
      diagnostic == "cooks_distance" ~ cook_threshold,
      diagnostic == "leverage" ~ leverage_threshold,
      diagnostic == "studentized_residual" ~ studentized_threshold,
      TRUE ~ dfbeta_threshold
    ),
    value_plot = if_else(diagnostic %in% c("studentized_residual", "dfbeta_teps", "dfbeta_interaction"), abs(value), value),
    diagnostic = factor(
      diagnostic,
      levels = c("cooks_distance", "leverage", "studentized_residual", "dfbeta_teps", "dfbeta_interaction"),
      labels = c("Cook distance", "Leverage", "|Studentized residual|", "|DFBETA TEPS|", "|DFBETA interaction|")
    ),
    subject_index = match(Subject_ID, sort(sample_matrix$Subject_ID))
  )

influence_label_data <- influence_plot_data %>%
  filter(value_plot > threshold) %>%
  group_by(endpoint_label, diagnostic) %>%
  slice_max(order_by = value_plot, n = 1, with_ties = FALSE) %>%
  ungroup()

p_influence <- ggplot(influence_plot_data, aes(x = subject_index, y = value_plot, colour = any_influence_flag)) +
  geom_hline(aes(yintercept = threshold), linetype = "dashed", linewidth = 0.35, colour = palette_contract[["neutral_mid"]]) +
  geom_point(size = 0.85, alpha = 0.8) +
  geom_text(
    data = influence_label_data,
    aes(label = Subject_ID),
    size = 1.7,
    vjust = -0.35,
    show.legend = FALSE
  ) +
  facet_grid(endpoint_label ~ diagnostic, scales = "free_y") +
  scale_colour_manual(values = c(`FALSE` = palette_contract[["neutral_mid"]], `TRUE` = "#C9574D")) +
  labs(
    title = "Fig. 9c | Influence and outlier diagnostics",
    subtitle = "Dashed lines are prespecified review thresholds. Flags trigger sensitivity analysis, not automatic subject deletion.",
    x = "Subject index",
    y = "Diagnostic value",
    colour = "Any flag"
  ) +
  theme_nature_contract(base_size = 6.3) +
  theme(
    legend.position = "top",
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    strip.text = element_text(size = 5.4)
  )
save_png(p_influence, "Fig09C_teps_influence_diagnostics.png", width_mm = 235, height_mm = 130, dpi = 300)

loo_plot_data <- teps_leave_one_out %>%
  left_join(loo_full_reference, by = c("endpoint_id", "estimand")) %>%
  mutate(
    endpoint_label = factor(endpoint_label, levels = validation_endpoint_registry$endpoint_label),
    estimand = factor(estimand, levels = c("Combined adjusted main effect", "Group x TEPS interaction", "HC Spearman rho", "SZ Spearman rho")),
    omitted_group = factor(omitted_group, levels = c("HC", "SZ"))
  )

loo_reference_rect <- loo_plot_data %>%
  distinct(endpoint_label, estimand, full_estimate, full_ci_low, full_ci_high)

p_loo <- ggplot(loo_plot_data, aes(x = subject_index, y = estimate, colour = omitted_group)) +
  geom_rect(
    data = loo_reference_rect,
    aes(xmin = -Inf, xmax = Inf, ymin = full_ci_low, ymax = full_ci_high),
    inherit.aes = FALSE,
    fill = palette_contract[["neutral_light"]],
    alpha = 0.45,
    colour = NA
  ) +
  geom_hline(data = loo_reference_rect, aes(yintercept = full_estimate), inherit.aes = FALSE, linewidth = 0.45, colour = palette_contract[["neutral_dark"]]) +
  geom_point(size = 0.85, alpha = 0.75) +
  facet_grid(endpoint_label ~ estimand, scales = "free_y") +
  scale_colour_manual(values = palette_contract[c("HC", "SZ")]) +
  labs(
    title = "Fig. 9d | Leave-one-subject-out stability",
    subtitle = "Each point is the estimate after omitting one participant. The black line and grey band are the full-sample estimate and 95% CI.",
    x = "Omitted subject index",
    y = "Re-estimated effect",
    colour = "Omitted group"
  ) +
  theme_nature_contract(base_size = 6.3) +
  theme(
    legend.position = "top",
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    strip.text = element_text(size = 5.4)
  )
save_png(p_loo, "Fig09D_teps_leave_one_out_stability.png", width_mm = 235, height_mm = 145, dpi = 300)

# -----------------------------------------------------------------------------
# Output index, provenance, and automated QA
# -----------------------------------------------------------------------------

figure_index <- tribble(
  ~figure_id, ~filename, ~analysis_part, ~visualized_result, ~interpretation_scope,
  "Fig01A", "Fig01A_ready_scale_distributions.png", "1", "All analysis-eligible score distributions", "Inferentially eligible according to frozen registry",
  "Fig01B", "Fig01B_definition_pending_scale_distributions.png", "1", "All definition-pending score distributions", "QC only",
  "Fig01C", "Fig01C_blocked_scale_distributions.png", "1", "All scoring-blocked score distributions", "QC only; prohibited from inference",
  "Fig02", "Fig02_scale_information_content.png", "1", "Missingness, range, floor, ceiling, normalized spread and uniqueness", "All 49 scores",
  "Fig03A", "Fig03A_item_level_qc.png", "1", "Item completeness, invalid values and complete-case N", "All 19 questionnaires",
  "Fig03B", "Fig03B_internal_consistency.png", "1", "Alpha and omega with bootstrap CI", "Verified keys plus explicitly marked QC-only totals",
  "Fig03C", "Fig03C_raw_unkeyed_reliability_qc.png", "1", "Raw questionnaire alpha", "QC only; not a keyed reliability conclusion",
  "Fig04A", "Fig04A_scale_correlations_all.png", "2", "Scale correlations in HC+SZ", "Both-group scales",
  "Fig04B", "Fig04B_scale_correlations_hc.png", "2", "Scale correlations in HC", "Both-group scales",
  "Fig04C", "Fig04C_scale_correlations_sz.png", "2", "Scale correlations in SZ", "Includes SZ-only clinical and motor scales",
  "Fig05A", "Fig05A_pleasure_scale_imaging_associations.png", "3", "All eligible pleasure-scale imaging associations", "Adjusted HC3 models",
  "Fig05B", "Fig05B_sz_clinical_imaging_associations.png", "3", "All SZ clinical/motor imaging associations", "Spearman with bootstrap CI",
  "Fig05C", "Fig05C_prespecified_association_scatter_atlas.png", "3", "All prespecified association families", "Residual or rank visualizations matching model class",
  "Fig06A", "Fig06A_stage_coupling_matrices.png", "4", "All 2x2 stage matched and cross-stage cells", "Common complete-case samples",
  "Fig06B", "Fig06B_stage_congruence_contrasts.png", "4", "Direct congruent-minus-cross-stage contrasts", "Bootstrap CI and BH correction",
  "Fig07A", "Fig07A_imaging_group_distributions.png", "5", "All HC/SZ imaging distributions", "Descriptive",
  "Fig07B", "Fig07B_imaging_group_effects.png", "5", "All adjusted HC/SZ imaging effects", "Holm family of seven",
  "Fig07C", "Fig07C_eligible_scale_group_effects.png", "5", "All eligible pleasure-scale HC/SZ effects", "Holm family of ten",
  "Fig08", "Fig08_integrated_evidence_map.png", "5", "Integrated imaging, clinical and association evidence", "Synthesis; no new statistical test",
  "Fig09A", "Fig09A_teps_group_interaction.png", "6", "Adjusted HC slope, SZ slope and Group x TEPS interaction", "Primary group-specificity test",
  "Fig09B", "Fig09B_teps_stratified_correlations.png", "6", "HC/SZ stratified Spearman correlations", "10,000-bootstrap CI",
  "Fig09C", "Fig09C_teps_influence_diagnostics.png", "6", "Cook distance, leverage, studentized residual and DFBETA diagnostics", "Review flags only; no automatic deletion",
  "Fig09D", "Fig09D_teps_leave_one_out_stability.png", "6", "Leave-one-subject-out main, interaction and stratified effects", "Influence robustness"
)

write_tsv_safe(figure_index, "FIGURE_INDEX.tsv")

summary_counts <- tibble(
  item = c(
    "final64_subjects", "questionnaires_audited", "scale_scores_registered", "eligible_scale_imaging_tests",
    "stage_coupling_cells", "stage_congruence_contrasts", "imaging_group_tests", "eligible_scale_group_tests",
    "focused_group_interaction_tests", "focused_stratified_correlations", "focused_rho_difference_tests",
    "focused_influence_subject_rows", "figures_expected"
  ),
  value = c(
    nrow(sample_matrix), n_distinct(items_final64$Questionnaire), nrow(scale_registry), nrow(imaging_associations),
    nrow(stage_coupling_cells), nrow(stage_congruence_contrasts), nrow(imaging_group_effects), nrow(scale_group_effects),
    nrow(teps_interaction_results), nrow(teps_stratified_correlations), nrow(teps_stratified_differences),
    nrow(interaction_influence_diagnostics), nrow(figure_index)
  )
)
write_tsv_safe(summary_counts, "ANALYSIS_OUTPUT_COUNTS.tsv")

input_provenance <- tibble(
  input_name = names(input_paths),
  path = unlist(input_paths),
  bytes = as.numeric(file.info(unlist(input_paths))$size),
  modified = as.character(file.info(unlist(input_paths))$mtime),
  md5 = unname(tools::md5sum(unlist(input_paths)))
)
write_tsv_safe(input_provenance, "RUN_INPUT_PROVENANCE.tsv")

figure_files <- list.files(figures_dir, full.names = TRUE)
qa_checks <- tibble(
  check = c(
    "subject_count_64", "group_balance_32_32", "questionnaires_19", "scale_registry_49",
    "imaging_associations_119", "stage_cells_32", "stage_contrasts_8",
    "focused_interactions_2", "focused_stratified_correlations_4", "focused_rho_differences_2",
    "figure_count_matches_index", "all_figure_extensions_png", "all_figures_nontrivial_size"
  ),
  passed = c(
    nrow(sample_matrix) == 64,
    all(table(sample_matrix$Group) == 32),
    n_distinct(items_final64$Questionnaire) == 19,
    nrow(scale_registry) == 49,
    nrow(imaging_associations) == 119,
    nrow(stage_coupling_cells) == 32,
    nrow(stage_congruence_contrasts) == 8,
    nrow(teps_interaction_results) == 2,
    nrow(teps_stratified_correlations) == 4,
    nrow(teps_stratified_differences) == 2,
    length(figure_files) == nrow(figure_index),
    all(tolower(tools::file_ext(figure_files)) == "png"),
    all(file.info(figure_files)$size > 10000)
  )
)
write_tsv_safe(qa_checks, "AUTOMATED_QA_CHECKS.tsv")

if (!all(qa_checks$passed)) {
  stop("Automated QA failed: ", paste(qa_checks$check[!qa_checks$passed], collapse = ", "))
}

writeLines(capture.output(sessionInfo()), file.path(logs_dir, "R_SESSION_INFO.txt"))
writeLines(
  c(
    "Final64 full scale reanalysis completed.",
    paste0("Completed at: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")),
    paste0("R version: ", R.version.string),
    paste0("Figures: ", length(figure_files), " PNG files"),
    "No PDF, SVG, or TIFF figure was generated."
  ),
  file.path(logs_dir, "RUN_COMPLETION.txt")
)

message("Final64 full scale reanalysis completed successfully.")
