options(stringsAsFactors = FALSE, scipen = 999)
if (.Platform$OS.type == "windows") {
  suppressWarnings(Sys.setlocale("LC_CTYPE", "Chinese (Simplified)_China.utf8"))
}

script_path <- function() {
  arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(arg)) stop("Run this script with Rscript so its location can be resolved.")
  normalizePath(sub("^--file=", "", arg[[1]]), winslash = "/", mustWork = TRUE)
}

SCRIPT_DIR <- dirname(script_path())
ANALYSIS_ROOT <- Sys.getenv("STRIATAL_CLINICAL_EXTENSION_OUTPUT", unset = NA_character_)
PROJECT_ROOT <- Sys.getenv("STRIATAL_PROJECT_ROOT", unset = NA_character_)
INTERNAL_DIR <- file.path(ANALYSIS_ROOT, "data_internal")
FIGURE_DIR <- file.path(ANALYSIS_ROOT, "figures")
LOG_DIR <- file.path(ANALYSIS_ROOT, "logs")
dir.create(INTERNAL_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)

SOURCE_ROOT <- Sys.getenv("STRIATAL_CLINICAL_SOURCE_ROOT", unset = NA_character_)
CLINICAL_AUDIT_ROOT <- file.path(SOURCE_ROOT, "final64_gradient_clinical_stats")

path_must_exist <- function(path) {
  if (!file.exists(path)) stop("Required input is missing: ", path)
  path
}

read_tsv <- function(path, col_classes = "character") {
  path_must_exist(path)
  read.delim(path, header = TRUE, sep = "\t", quote = "\"", comment.char = "",
             check.names = FALSE, colClasses = col_classes,
             na.strings = c("", "NA", "NaN", "nan"), fileEncoding = "UTF-8")
}

write_csv_utf8 <- function(x, path) {
  write.table(x, file = path, sep = ",", row.names = FALSE, col.names = TRUE,
              quote = TRUE, na = "", qmethod = "double", fileEncoding = "UTF-8")
}

write_tsv_utf8 <- function(x, path) {
  write.table(x, file = path, sep = "\t", row.names = FALSE, col.names = TRUE,
              quote = FALSE, na = "", fileEncoding = "UTF-8")
}

normalize_subject <- function(x) {
  x <- tolower(trimws(as.character(x)))
  x <- gsub("_", "-", x, fixed = TRUE)
  n <- suppressWarnings(as.integer(sub("^sub-?", "", x)))
  ifelse(is.na(n), NA_character_, sprintf("sub-%04d", n))
}

as_num <- function(x) suppressWarnings(as.numeric(trimws(as.character(x))))
nonmissing <- function(x) !is.na(x) & nzchar(trimws(as.character(x)))

parse_duration_years <- function(x) {
  x <- trimws(as.character(x))
  out <- suppressWarnings(as.numeric(x))
  idx <- is.na(out) & grepl("^[0-9]+/[0-9]+", x)
  if (any(idx)) {
    num <- as.numeric(sub("^([0-9]+)/([0-9]+).*$", "\\1", x[idx]))
    den <- as.numeric(sub("^([0-9]+)/([0-9]+).*$", "\\2", x[idx]))
    out[idx] <- num / den
  }
  month_idx <- is.na(out) & grepl("[0-9]+月", x)
  if (any(month_idx)) {
    months <- as.numeric(sub("^.*?([0-9]+)月.*$", "\\1", x[month_idx]))
    out[month_idx] <- months / 12
  }
  out
}

parse_dup_years <- function(x) {
  x <- trimws(as.character(x))
  out <- rep(NA_real_, length(x))
  out[grepl("半年", x)] <- 0.5
  month <- grepl("[0-9]+个月", x)
  out[month] <- as.numeric(sub("^.*?([0-9]+)个月.*$", "\\1", x[month])) / 12
  year <- grepl("[0-9]+年", x)
  out[year] <- as.numeric(sub("^.*?([0-9]+)年.*$", "\\1", x[year]))
  out
}

parse_first_number <- function(x) {
  x <- trimws(as.character(x))
  hit <- rep(NA_character_, length(x))
  ok <- !is.na(x) & grepl("[0-9]+(?:\\.[0-9]+)?", x, perl = TRUE)
  hit[ok] <- sub("^.*?([0-9]+(?:\\.[0-9]+)?).*$", "\\1", x[ok], perl = TRUE)
  suppressWarnings(as.numeric(hit))
}

summary_numeric <- function(x, cohort, variable, unit, source_note = "") {
  x <- as.numeric(x)
  z <- x[is.finite(x)]
  data.frame(
    cohort = cohort, variable = variable, unit = unit,
    valid_n = length(z), missing_n = length(x) - length(z),
    mean = if (length(z)) mean(z) else NA_real_,
    sd = if (length(z) > 1) sd(z) else NA_real_,
    median = if (length(z)) median(z) else NA_real_,
    q1 = if (length(z)) unname(quantile(z, .25)) else NA_real_,
    q3 = if (length(z)) unname(quantile(z, .75)) else NA_real_,
    min = if (length(z)) min(z) else NA_real_,
    max = if (length(z)) max(z) else NA_real_,
    source_note = source_note,
    stringsAsFactors = FALSE
  )
}

append_provenance <- function(rows) {
  path <- file.path(INTERNAL_DIR, "provenance_rows.tsv")
  old <- if (file.exists(path)) read_tsv(path) else NULL
  all <- rbind(old, rows)
  write_tsv_utf8(all, path)
}

session_record <- function(script_name) {
  capture.output(sessionInfo(), file = file.path(LOG_DIR, paste0(script_name, "_sessionInfo.txt")))
}

spearman_boot <- function(data, xvar, yvar, seed, B = 10000L) {
  d <- data[is.finite(as.numeric(data[[xvar]])) & is.finite(as.numeric(data[[yvar]])), c(xvar, yvar)]
  x <- as.numeric(d[[xvar]])
  y <- as.numeric(d[[yvar]])
  n <- length(x)
  if (n < 5L) return(list(n=n, rho=NA_real_, ci_low=NA_real_, ci_high=NA_real_, p=NA_real_, boot_valid=0L))
  rho <- suppressWarnings(cor(x, y, method = "spearman"))
  test <- suppressWarnings(cor.test(x, y, method = "spearman", exact = FALSE, continuity = FALSE))
  set.seed(seed)
  boots <- replicate(B, {
    ii <- sample.int(n, n, replace = TRUE)
    suppressWarnings(cor(x[ii], y[ii], method = "spearman"))
  })
  boots <- boots[is.finite(boots)]
  ci <- if (length(boots) >= B * .9) unname(quantile(boots, c(.025, .975), na.rm = TRUE, type = 6)) else c(NA_real_, NA_real_)
  list(n=n, rho=rho, ci_low=ci[1], ci_high=ci[2], p=unname(test$p.value), boot_valid=length(boots))
}

safe_holm <- function(p) {
  out <- rep(NA_real_, length(p))
  ok <- is.finite(p)
  out[ok] <- p.adjust(p[ok], method = "holm")
  out
}

rank_biserial_boot <- function(data, group_var, yvar, seed, B = 10000L) {
  g <- as.numeric(data[[group_var]])
  y <- as.numeric(data[[yvar]])
  ok <- is.finite(g) & is.finite(y) & g %in% c(0, 1)
  g <- g[ok]; y <- y[ok]
  y1 <- y[g == 1]; y0 <- y[g == 0]
  n1 <- length(y1); n0 <- length(y0)
  if (n1 < 3L || n0 < 3L) return(list(n=n1+n0,n1=n1,n0=n0,effect=NA_real_,ci_low=NA_real_,ci_high=NA_real_,p=NA_real_,boot_valid=0L))
  effect_fun <- function(a, b) {
    cmp <- outer(a, b, "-")
    2 * (mean(cmp > 0) + .5 * mean(cmp == 0)) - 1
  }
  eff <- effect_fun(y1, y0)
  wt <- suppressWarnings(wilcox.test(y1, y0, exact = FALSE, correct = FALSE))
  set.seed(seed)
  boots <- replicate(B, effect_fun(sample(y1, n1, TRUE), sample(y0, n0, TRUE)))
  boots <- boots[is.finite(boots)]
  ci <- unname(quantile(boots, c(.025, .975), type=6, na.rm=TRUE))
  list(n=n1+n0,n1=n1,n0=n0,effect=eff,ci_low=ci[1],ci_high=ci[2],p=unname(wt$p.value),boot_valid=length(boots))
}

zscore <- function(x) as.numeric(scale(as.numeric(x)))
rank_z <- function(x) zscore(rank(as.numeric(x), ties.method = "average"))

hc3_term <- function(fit, term) {
  X <- model.matrix(fit)
  e <- residuals(fit)
  h <- hatvalues(fit)
  bread <- solve(crossprod(X))
  meat <- crossprod(X, X * as.numeric((e / (1 - h))^2))
  vc <- bread %*% meat %*% bread
  if (!term %in% names(coef(fit))) stop("Model term not found: ", term)
  idx <- match(term, names(coef(fit)))
  est <- unname(coef(fit)[idx])
  se <- sqrt(vc[idx, idx])
  df <- df.residual(fit)
  crit <- qt(.975, df)
  tval <- est / se
  p <- 2 * pt(abs(tval), df, lower.tail = FALSE)
  list(estimate=est,se=se,ci_low=est-crit*se,ci_high=est+crit*se,p=p,df=df,vcov=vc)
}

model_diagnostics <- function(fit, focal_term) {
  X <- model.matrix(fit)
  cook <- cooks.distance(fit)
  lev <- hatvalues(fit)
  n <- nobs(fit); p <- ncol(X)
  focal_col <- X[, focal_term]
  others <- X[, setdiff(colnames(X), c("(Intercept)", focal_term)), drop=FALSE]
  focal_vif <- if (!ncol(others)) 1 else 1 / (1 - summary(lm(focal_col ~ others))$r.squared)
  data.frame(
    model_n=n, model_parameters=p, residual_df=df.residual(fit),
    max_cooks_distance=max(cook), cooks_threshold=4/n, n_cooks_over_threshold=sum(cook > 4/n),
    max_leverage=max(lev), leverage_threshold=2*p/n, n_leverage_over_threshold=sum(lev > 2*p/n),
    focal_vif=focal_vif, design_condition_number=kappa(X),
    residual_shapiro_p=if(n >= 3 && n <= 5000) shapiro.test(residuals(fit))$p.value else NA_real_,
    model_status="fit_ok", stringsAsFactors=FALSE
  )
}
