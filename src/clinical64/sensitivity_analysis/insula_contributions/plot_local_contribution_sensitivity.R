#!/usr/bin/env Rscript

# R-only figure generation for STEP 3.
# All panels display additive Pearson profile-correlation contributions, not raw FC.

# Use the configured R library (R_LIBS_USER).

if (!requireNamespace("ragg", quietly = TRUE)) {
  stop("Required R package 'ragg' is unavailable; no fallback backend is permitted.")
}

OUT <- file.path(Sys.getenv("STRIATAL_SUPPLEMENT_ROOT", unset = NA_character_), "STEP3_LocalContribution_InsulaSensitivity")
effects_file <- file.path(OUT, "local_effects_full_vs_minus_insula.tsv")
subjects_file <- file.path(OUT, "subject_level_contributions.tsv")

effects <- read.delim(effects_file, check.names = FALSE, stringsAsFactors = FALSE)
subjects <- read.delim(subjects_file, check.names = FALSE, stringsAsFactors = FALSE)

required_effects <- c(
  "subfamily", "stage", "full_beta_SZ_minus_HC", "full_CI95_low", "full_CI95_high",
  "full_q_FDR_13_prespecified", "minus_insula_beta_SZ_minus_HC",
  "minus_insula_CI95_low", "minus_insula_CI95_high",
  "minus_insula_q_FDR_13_prespecified", "delta_beta_minus_insula_minus_full"
)
required_subjects <- c("subject", "group", "roi_set", "subfamily", "contribution_sum")
if (!all(required_effects %in% names(effects))) {
  stop("Effect table is missing required columns.")
}
if (!all(required_subjects %in% names(subjects))) {
  stop("Subject contribution table is missing required columns.")
}

FULL_SET <- "Full_65ROI"
MINUS_SET <- "Minus_Insula_53ROI"

COL <- c(
  neutral_dark = "#333333",
  neutral_mid = "#767676",
  neutral_light = "#D6D6D6",
  signal_blue = "#0F4D92",
  signal_teal = "#2A8C82",
  hc = "#375A7F",
  sz = "#B34A4A",
  zero = "#A0A0A0"
)

open_png <- function(path, width_mm, height_mm, dpi = 300) {
  ragg::agg_png(
    filename = path,
    width = width_mm,
    height = height_mm,
    units = "mm",
    res = dpi,
    background = "white",
    scaling = 1
  )
}

open_pdf <- function(path, width_mm, height_mm) {
  grDevices::cairo_pdf(
    filename = path,
    width = width_mm / 25.4,
    height = height_mm / 25.4,
    family = "Arial",
    bg = "white",
    onefile = TRUE
  )
}

draw_both <- function(stem, width_mm, height_mm, draw_fun) {
  png_path <- file.path(OUT, paste0(stem, ".png"))
  pdf_path <- file.path(OUT, paste0(stem, ".pdf"))
  open_png(png_path, width_mm, height_mm)
  draw_fun()
  invisible(dev.off())
  open_pdf(pdf_path, width_mm, height_mm)
  draw_fun()
  invisible(dev.off())
  c(png_path, pdf_path)
}

format_q <- function(x) {
  ifelse(x < 0.001, formatC(x, format = "e", digits = 1), formatC(x, format = "f", digits = 3))
}

draw_figure1 <- function() {
  target <- effects[effects$subfamily %in% c("SMA", "IFG"), ]
  target <- target[match(c("SMA", "IFG"), target$subfamily), ]
  plot_df <- rbind(
    data.frame(
      subfamily = target$subfamily,
      roi_set = "Full (65 ROIs)",
      beta = target$full_beta_SZ_minus_HC,
      low = target$full_CI95_low,
      high = target$full_CI95_high,
      q = target$full_q_FDR_13_prespecified,
      stringsAsFactors = FALSE
    ),
    data.frame(
      subfamily = target$subfamily,
      roi_set = "Minus Insula (53 ROIs)",
      beta = target$minus_insula_beta_SZ_minus_HC,
      low = target$minus_insula_CI95_low,
      high = target$minus_insula_CI95_high,
      q = target$minus_insula_q_FDR_13_prespecified,
      stringsAsFactors = FALSE
    )
  )
  plot_df <- plot_df[c(1, 3, 2, 4), ]
  y <- c(4.1, 3.45, 2.05, 1.4)
  ci_range <- range(c(plot_df$low, plot_df$high), finite = TRUE)
  span <- diff(ci_range)
  label_x <- max(0, ci_range[2]) + 0.05 * span
  xlim <- c(ci_range[1] - 0.08 * span, label_x + 0.36 * span)

  par(
    mar = c(4.2, 6.8, 2.3, 1.2),
    family = "Arial",
    las = 1,
    mgp = c(2.2, 0.55, 0),
    tcl = -0.25,
    cex.axis = 0.82,
    cex.lab = 0.9,
    bty = "n"
  )
  plot(
    NA,
    xlim = xlim,
    ylim = c(0.75, 4.75),
    axes = FALSE,
    xlab = "Adjusted SZ - HC effect on contribution sum (beta, 95% CI)",
    ylab = ""
  )
  axis(1, col = COL["neutral_dark"], col.axis = COL["neutral_dark"])
  axis(
    2,
    at = y,
    labels = c("SMA - Full", "SMA - Minus Insula", "IFG - Full", "IFG - Minus Insula"),
    tick = FALSE,
    hadj = 1
  )
  abline(v = 0, lty = 2, lwd = 0.9, col = COL["zero"])
  segments(plot_df$low, y, plot_df$high, y, lwd = 1.5, col = c(COL["neutral_mid"], COL["signal_blue"], COL["neutral_mid"], COL["signal_teal"]))
  segments(plot_df$low, y - 0.08, plot_df$low, y + 0.08, lwd = 1.1, col = c(COL["neutral_mid"], COL["signal_blue"], COL["neutral_mid"], COL["signal_teal"]))
  segments(plot_df$high, y - 0.08, plot_df$high, y + 0.08, lwd = 1.1, col = c(COL["neutral_mid"], COL["signal_blue"], COL["neutral_mid"], COL["signal_teal"]))
  points(
    plot_df$beta,
    y,
    pch = c(21, 19, 21, 19),
    bg = c("white", COL["signal_blue"], "white", COL["signal_teal"]),
    col = c(COL["neutral_mid"], COL["signal_blue"], COL["neutral_mid"], COL["signal_teal"]),
    cex = 1.05,
    lwd = 1.2
  )
  text(
    label_x,
    y,
    labels = sprintf("beta=%+.4f; q=%s", plot_df$beta, format_q(plot_df$q)),
    adj = 0,
    cex = 0.72,
    col = COL["neutral_dark"]
  )
  mtext("Bin 10 Pearson profile-correlation contribution (not raw FC)", side = 3, line = 0.45, adj = 0, cex = 0.92, font = 2)
}

draw_subject_panel <- function(subfamily, panel_color) {
  d <- subjects[subjects$subfamily == subfamily & subjects$roi_set %in% c(FULL_SET, MINUS_SET), ]
  if (nrow(d) != 128L) stop(paste("Expected 128 rows for", subfamily))
  wide <- reshape(
    d[, c("subject", "group", "roi_set", "contribution_sum")],
    idvar = c("subject", "group"),
    timevar = "roi_set",
    direction = "wide"
  )
  full_col <- paste0("contribution_sum.", FULL_SET)
  minus_col <- paste0("contribution_sum.", MINUS_SET)
  values <- c(wide[[full_col]], wide[[minus_col]])
  ylim <- range(values, finite = TRUE)
  pad <- max(diff(ylim) * 0.12, 0.005)
  ylim <- ylim + c(-pad, pad)

  plot(
    NA,
    xlim = c(0.45, 5.55),
    ylim = ylim,
    axes = FALSE,
    xlab = "",
    ylab = "Contribution sum to Pearson r",
    main = subfamily
  )
  axis(2, cex.axis = 0.78)
  axis(
    1,
    at = c(1, 2, 4, 5),
    labels = c("HC\nFull", "HC\nMinus", "SZ\nFull", "SZ\nMinus"),
    tick = FALSE,
    cex.axis = 0.75,
    line = -0.3
  )
  abline(h = 0, lty = 3, lwd = 0.8, col = COL["zero"])
  for (grp in c("HC", "SZ")) {
    g <- wide[wide$group == grp, ]
    g <- g[order(g$subject), ]
    base_x <- if (grp == "HC") 1 else 4
    jitter <- seq(-0.13, 0.13, length.out = nrow(g))
    group_col <- if (grp == "HC") COL["hc"] else COL["sz"]
    segments(
      base_x + jitter,
      g[[full_col]],
      base_x + 1 + jitter,
      g[[minus_col]],
      col = grDevices::adjustcolor(group_col, alpha.f = 0.20),
      lwd = 0.65
    )
    points(
      base_x + jitter,
      g[[full_col]],
      pch = 21,
      bg = "white",
      col = grDevices::adjustcolor(group_col, alpha.f = 0.82),
      cex = 0.65,
      lwd = 0.8
    )
    points(
      base_x + 1 + jitter,
      g[[minus_col]],
      pch = 19,
      col = grDevices::adjustcolor(group_col, alpha.f = 0.82),
      cex = 0.62
    )
    means <- c(mean(g[[full_col]]), mean(g[[minus_col]]))
    points(base_x + c(0, 1), means, pch = 18, cex = 1.15, col = panel_color)
    segments(base_x - 0.18 + c(0, 1), means, base_x + 0.18 + c(0, 1), means, lwd = 1.4, col = panel_color)
  }
  box(bty = "l", col = COL["neutral_dark"])
}

draw_figure2 <- function() {
  par(
    mfrow = c(1, 2),
    mar = c(4.5, 4.0, 2.6, 0.8),
    oma = c(1.8, 0.2, 0.4, 0.2),
    family = "Arial",
    las = 1,
    mgp = c(2.2, 0.55, 0),
    tcl = -0.25,
    cex.axis = 0.82,
    cex.lab = 0.86,
    cex.main = 0.96,
    bty = "n"
  )
  draw_subject_panel("SMA", COL["signal_blue"])
  mtext("a", side = 3, line = 1.15, adj = 0, font = 2, cex = 1.0)
  draw_subject_panel("IFG", COL["signal_teal"])
  mtext("b", side = 3, line = 1.15, adj = 0, font = 2, cex = 1.0)
  mtext(
    "Each line links the same subject; Minus recomputes the 53-ROI Pearson centering, scaling and denominator. Not raw FC.",
    side = 1,
    outer = TRUE,
    line = 0.35,
    cex = 0.72,
    col = COL["neutral_dark"]
  )
}

draw_figure3 <- function() {
  d <- effects[effects$subfamily != "Insula" & is.finite(effects$minus_insula_beta_SZ_minus_HC), ]
  d <- d[order(d$minus_insula_beta_SZ_minus_HC, decreasing = TRUE), ]
  y <- seq_len(nrow(d))
  labels <- paste0(d$subfamily, " (", ifelse(d$stage == "Not_assigned", "Not assigned", d$stage), ")")
  xvals <- c(d$full_beta_SZ_minus_HC, d$minus_insula_beta_SZ_minus_HC)
  xlim <- range(xvals, finite = TRUE)
  pad <- max(diff(xlim) * 0.12, 0.006)
  xlim <- xlim + c(-pad, pad)
  line_cols <- rep(COL["neutral_mid"], nrow(d))
  line_cols[d$subfamily == "SMA"] <- COL["signal_blue"]
  line_cols[d$subfamily == "IFG"] <- COL["signal_teal"]
  line_lwd <- ifelse(d$subfamily %in% c("SMA", "IFG"), 1.7, 0.9)

  par(
    mar = c(4.4, 10.2, 2.5, 1.0),
    family = "Arial",
    las = 1,
    mgp = c(2.2, 0.55, 0),
    tcl = -0.25,
    cex.axis = 0.70,
    cex.lab = 0.88,
    bty = "n"
  )
  plot(
    NA,
    xlim = xlim,
    ylim = c(0.5, nrow(d) + 1.2),
    axes = FALSE,
    xlab = "Adjusted SZ - HC effect on Pearson contribution sum (beta)",
    ylab = ""
  )
  axis(1)
  axis(2, at = y, labels = labels, tick = FALSE, hadj = 1)
  abline(v = 0, lty = 2, lwd = 0.85, col = COL["zero"])
  segments(
    d$full_beta_SZ_minus_HC,
    y,
    d$minus_insula_beta_SZ_minus_HC,
    y,
    col = grDevices::adjustcolor(line_cols, alpha.f = 0.72),
    lwd = line_lwd
  )
  points(d$full_beta_SZ_minus_HC, y, pch = 21, bg = "white", col = line_cols, cex = 0.75, lwd = 1.0)
  points(d$minus_insula_beta_SZ_minus_HC, y, pch = 19, col = line_cols, cex = 0.75)
  legend(
    "topleft",
    inset = c(0, -0.01),
    legend = c("Full (65 ROIs)", "Minus Insula (53 ROIs)"),
    pch = c(21, 19),
    pt.bg = c("white", COL["neutral_dark"]),
    col = COL["neutral_dark"],
    bty = "n",
    horiz = TRUE,
    cex = 0.76,
    xpd = NA
  )
  mtext("All retained subfamilies; lines show the normalization-induced shift", side = 3, line = 0.55, adj = 0, cex = 0.92, font = 2)
}

created <- c(
  draw_both("Figure1_SMA_IFG_effects_full_vs_minus_insula", 183, 105, draw_figure1),
  draw_both("Figure2_subject_level_SMA_IFG_contributions", 183, 112, draw_figure2),
  draw_both("Figure3_all_retained_subfamily_effect_changes", 183, 135, draw_figure3)
)

qc <- data.frame(
  file = basename(created),
  bytes = file.info(created)$size,
  exists = file.exists(created),
  stringsAsFactors = FALSE
)
qc$pass_nonempty <- qc$exists & qc$bytes > 10000

png_files <- created[grepl("[.]png$", created)]
if (requireNamespace("png", quietly = TRUE)) {
  dims <- lapply(png_files, function(path) dim(png::readPNG(path, info = TRUE)))
  dim_table <- data.frame(
    file = basename(png_files),
    pixel_height = vapply(dims, function(x) x[1], integer(1)),
    pixel_width = vapply(dims, function(x) x[2], integer(1)),
    stringsAsFactors = FALSE
  )
  qc <- merge(qc, dim_table, by = "file", all.x = TRUE, sort = FALSE)
}

write.table(
  qc,
  file.path(OUT, "figure_export_qc.tsv"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE,
  na = ""
)

if (!all(qc$pass_nonempty)) {
  stop("One or more figure exports failed the non-empty file check.")
}

cat("R figure generation: SUCCESS\n")
print(qc)
cat("R version:", R.version.string, "\n")
cat("ragg version:", as.character(utils::packageVersion("ragg")), "\n")
