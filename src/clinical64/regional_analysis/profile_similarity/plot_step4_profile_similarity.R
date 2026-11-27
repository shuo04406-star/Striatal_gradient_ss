options(stringsAsFactors = FALSE, scipen = 8)

out_dir <- file.path(Sys.getenv("STRIATAL_SUPPLEMENT_ROOT", unset = NA_character_), "STEP4_SMA_IFG_ProfileSimilarity")
table_dir <- file.path(out_dir, "tables")
figure_dir <- file.path(out_dir, "figures")

profiles <- read.delim(file.path(table_dir, "SMA_IFG_subject_10bin_profiles.tsv"), check.names = FALSE, na.strings = c("N/A", ""))
metrics <- read.delim(file.path(table_dir, "subject_level_offset_slope.tsv"), check.names = FALSE, na.strings = c("N/A", ""))
shape <- read.delim(file.path(table_dir, "subject_level_shape_residuals.tsv"), check.names = FALSE, na.strings = c("N/A", ""))
group_stats <- read.delim(file.path(table_dir, "profile_group_statistics.tsv"), check.names = FALSE, na.strings = c("N/A", ""))
tests <- read.delim(file.path(table_dir, "STEP4_statistical_tests.tsv"), check.names = FALSE, na.strings = c("N/A", ""))
similarity <- read.delim(file.path(table_dir, "profile_similarity_metrics.tsv"), check.names = FALSE, na.strings = c("N/A", ""))

target_order <- c("SMA", "IFG")
target_titles <- c(SMA = "SMA - Anticipation-associated", IFG = "IFG - Outcome-associated")
group_cols <- c(HC = "#4C78A8", SZ = "#D65F0E")
target_cols <- c(SMA = "#76528B", IFG = "#3A7D78")
component_cols <- c(Offset = "#4C78A8", Linear = "#E69F00", `Nonlinear shape` = "#7B3294", Total = "#333333")
neutral <- "#4D4D4D"

open_png <- function(path, width_mm, height_mm) {
  grDevices::png(filename = path, width = width_mm, height = height_mm,
                 units = "mm", res = 320, bg = "white", type = "cairo-png")
}

open_pdf <- function(path, width_mm, height_mm) {
  grDevices::pdf(file = path, width = width_mm / 25.4, height = height_mm / 25.4,
                 family = "Helvetica", onefile = TRUE, useDingbats = FALSE, paper = "special")
}

with_device <- function(device_fun, path, width_mm, height_mm, draw_fun) {
  device_fun(path, width_mm, height_mm)
  old <- par(no.readonly = TRUE)
  on.exit({par(old); dev.off()}, add = TRUE)
  draw_fun()
}

draw_group_mean_panel <- function(target, panel_label) {
  d <- group_stats[group_stats$Target == target, ]
  low <- min(d$HC_mean_FC_CI95_low, d$SZ_mean_FC_CI95_low)
  high <- max(d$HC_mean_FC_CI95_high, d$SZ_mean_FC_CI95_high)
  pad <- 0.10 * (high - low)
  par(mar = c(1.4, 3.8, 3.1, 0.7), mgp = c(2.0, 0.55, 0), tcl = -0.2)
  plot(NA, NA, xlim = c(1, 10), ylim = c(low - pad, high + pad), bty = "n",
       xaxt = "n", xlab = "", ylab = "Fisher-z FC", cex.axis = 0.67, cex.lab = 0.73)
  abline(h = pretty(c(low, high), 4), col = "#EEEEEE", lwd = 0.7)
  for (g in c("HC", "SZ")) {
    mn <- d[[paste0(g, "_mean_FC")]]
    lo <- d[[paste0(g, "_mean_FC_CI95_low")]]
    hi <- d[[paste0(g, "_mean_FC_CI95_high")]]
    polygon(c(d$Bin, rev(d$Bin)), c(lo, rev(hi)),
            col = adjustcolor(group_cols[g], alpha.f = 0.17), border = NA)
    lines(d$Bin, mn, col = group_cols[g], lwd = 2.0)
    points(d$Bin, mn, pch = 21, bg = group_cols[g], col = "white", cex = 0.75, lwd = 0.4)
  }
  title(paste0(panel_label, "  ", target_titles[target]), adj = 0, font.main = 2, cex.main = 0.87, line = 1.15)
  if (target == "IFG") {
    legend("topright", legend = c("HC", "SZ"), col = group_cols, lwd = 2,
           pch = 21, pt.bg = group_cols, bty = "n", cex = 0.68, pt.cex = 0.8)
  }
  mtext("Group mean and 95% CI", side = 3, adj = 0, line = 0.05, cex = 0.60, col = neutral)
}

draw_individual_panel <- function(target) {
  d <- profiles[profiles$Target == target, ]
  y <- as.matrix(d[paste0("FC_bin", 1:10)])
  lim <- range(y)
  pad <- 0.05 * diff(lim)
  par(mar = c(3.1, 3.8, 1.0, 0.7), mgp = c(1.9, 0.55, 0), tcl = -0.2)
  plot(NA, NA, xlim = c(1, 10), ylim = lim + c(-pad, pad), bty = "n",
       xlab = "G1 bin", ylab = "Fisher-z FC", cex.axis = 0.65, cex.lab = 0.70,
       xaxt = "n")
  axis(1, at = 1:10, cex.axis = 0.62)
  abline(h = pretty(lim, 4), col = "#F0F0F0", lwd = 0.65)
  for (g in c("HC", "SZ")) {
    idx <- which(d$Group == g)
    for (i in idx) lines(1:10, y[i, ], col = adjustcolor(group_cols[g], alpha.f = 0.13), lwd = 0.7)
    lines(1:10, colMeans(y[idx, , drop = FALSE]), col = group_cols[g], lwd = 1.8)
  }
  mtext("All participants (thin lines); group means (thick lines)", side = 3, adj = 0, line = -0.25, cex = 0.58, col = neutral)
}

draw_figure1 <- function() {
  layout(matrix(1:4, nrow = 2, byrow = TRUE), heights = c(1.12, 1.0))
  par(oma = c(0.3, 0.3, 3.0, 0.3))
  draw_group_mean_panel("SMA", "a")
  draw_group_mean_panel("IFG", "b")
  draw_individual_panel("SMA")
  draw_individual_panel("IFG")
  mtext("SMA and IFG 10-bin functional-connectivity profiles", side = 3, outer = TRUE,
        line = 1.30, font = 2, cex = 0.98)
  mtext("Original subject-level Fisher-z FC; n = 32 per group; CI is uncertainty of the group mean",
        side = 3, outer = TRUE, line = 0.08, cex = 0.66, col = neutral)
}

forest_panel <- function(d, title_text, xlab_text, panel_label) {
  d <- d[match(target_order, d$Target), ]
  y <- c(2, 1)
  span <- max(d$CI95_high) - min(d$CI95_low)
  if (span == 0) span <- 1
  xlim <- c(min(d$CI95_low, 0) - 0.12 * span, max(d$CI95_high, 0) + 0.78 * span)
  par(mar = c(4.1, 4.1, 3.0, 0.8), mgp = c(2.1, 0.58, 0), tcl = -0.2)
  plot(NA, NA, xlim = xlim, ylim = c(0.5, 2.5), yaxt = "n", bty = "n",
       xlab = xlab_text, ylab = "", cex.axis = 0.70, cex.lab = 0.72)
  abline(v = 0, lty = 2, col = "#777777")
  abline(h = y, col = "#EFEFEF", lwd = 0.7)
  axis(2, at = y, labels = target_order, las = 1, tick = FALSE, cex.axis = 0.74)
  for (i in seq_len(nrow(d))) {
    col <- target_cols[d$Target[i]]
    segments(d$CI95_low[i], y[i], d$CI95_high[i], y[i], col = col, lwd = 2.1)
    segments(d$CI95_low[i], y[i] - 0.08, d$CI95_low[i], y[i] + 0.08, col = col, lwd = 1)
    segments(d$CI95_high[i], y[i] - 0.08, d$CI95_high[i], y[i] + 0.08, col = col, lwd = 1)
    points(d$Beta_SZ_minus_HC[i], y[i], pch = 21, bg = col, col = "white", cex = 1.2, lwd = 0.5)
    label_x <- max(d$CI95_high, 0) + 0.08 * span
    text(label_x, y[i], sprintf("beta = %.4f\nHolm p = %.3g", d$Beta_SZ_minus_HC[i], d$Holm_adjusted_p_6_tests[i]),
         adj = c(0, 0.5), cex = 0.62, col = "#333333")
  }
  title(paste0(panel_label, "  ", title_text), adj = 0, font.main = 2, cex.main = 0.84, line = 1.1)
}

draw_figure2 <- function() {
  layout(matrix(1:2, nrow = 1))
  par(oma = c(0.3, 0.2, 3.0, 0.2))
  forest_panel(tests[tests$Analysis_dimension == "FC_Offset", ], "Mean FC offset", "Adjusted SZ - HC beta (Fisher-z FC)", "a")
  forest_panel(tests[tests$Analysis_dimension == "Linear_G1_Slope", ], "Linear G1 slope", "Adjusted SZ - HC beta (FC per G1-position SD)", "b")
  mtext("Adjusted group effects on profile offset and linear slope", side = 3, outer = TRUE,
        line = 1.25, font = 2, cex = 0.98)
  mtext("Subject-level OLS with HC3 SE; 95% Wald CI; Holm correction across six prespecified tests",
        side = 3, outer = TRUE, line = 0.05, cex = 0.65, col = neutral)
}

draw_shape_mean_panel <- function(target, panel_label) {
  d <- group_stats[group_stats$Target == target, ]
  low <- min(d$HC_mean_shape_CI95_low, d$SZ_mean_shape_CI95_low)
  high <- max(d$HC_mean_shape_CI95_high, d$SZ_mean_shape_CI95_high)
  pad <- 0.10 * max(high - low, 0.001)
  test <- tests[tests$Target == target & tests$Analysis_dimension == "Nonlinear_Profile_Shape", ]
  par(mar = c(1.4, 3.8, 3.4, 0.7), mgp = c(2.0, 0.55, 0), tcl = -0.2)
  plot(NA, NA, xlim = c(1, 10), ylim = c(low - pad, high + pad), bty = "n", xaxt = "n",
       xlab = "", ylab = "Detrended FC", cex.axis = 0.67, cex.lab = 0.71)
  abline(h = 0, col = "#777777", lty = 2)
  for (g in c("HC", "SZ")) {
    mn <- d[[paste0(g, "_mean_shape_residual")]]
    lo <- d[[paste0(g, "_mean_shape_CI95_low")]]
    hi <- d[[paste0(g, "_mean_shape_CI95_high")]]
    polygon(c(d$Bin, rev(d$Bin)), c(lo, rev(hi)), col = adjustcolor(group_cols[g], alpha.f = 0.15), border = NA)
    lines(d$Bin, mn, col = group_cols[g], lwd = 1.8)
    points(d$Bin, mn, pch = 21, bg = group_cols[g], col = "white", cex = 0.68, lwd = 0.4)
  }
  title(paste0(panel_label, "  ", target_titles[target]), adj = 0, font.main = 2, cex.main = 0.83, line = 1.45)
  mtext(sprintf("Pillai V = %.3f, F(%d,%d) = %.2f, raw p = %.3f, Holm p = %.3f",
                test$Pillai_trace, test$df_num, test$df_den, test$Approx_F, test$raw_p, test$Holm_adjusted_p_6_tests),
        side = 3, adj = 0, line = 0.2, cex = 0.53, col = neutral)
  if (target == "IFG") legend("topright", legend = c("HC", "SZ"), col = group_cols, lwd = 1.8,
                                pch = 21, pt.bg = group_cols, bty = "n", cex = 0.62)
}

draw_shape_difference_panel <- function(target) {
  d <- group_stats[group_stats$Target == target, ]
  yy <- d$adjusted_nonlinear_component
  pad <- 0.12 * max(diff(range(c(yy, 0))), 0.001)
  par(mar = c(3.2, 3.8, 1.0, 0.7), mgp = c(1.9, 0.55, 0), tcl = -0.2)
  plot(d$Bin, yy, type = "o", pch = 21, bg = target_cols[target], col = target_cols[target],
       lwd = 1.8, cex = 0.75, xlim = c(1, 10), ylim = range(c(yy, 0)) + c(-pad, pad),
       bty = "n", xaxt = "n", xlab = "G1 bin", ylab = "Adjusted SZ - HC\nnonlinear component",
       cex.axis = 0.64, cex.lab = 0.67)
  axis(1, at = 1:10, cex.axis = 0.61)
  abline(h = 0, lty = 2, col = "#777777")
  mtext("Covariate-adjusted nonlinear difference (descriptive)", side = 3, adj = 0, line = -0.25, cex = 0.56, col = neutral)
}

draw_figure3 <- function() {
  layout(matrix(1:4, nrow = 2, byrow = TRUE), heights = c(1.12, 1.0))
  par(oma = c(0.3, 0.3, 3.0, 0.3))
  draw_shape_mean_panel("SMA", "a")
  draw_shape_mean_panel("IFG", "b")
  draw_shape_difference_panel("SMA")
  draw_shape_difference_panel("IFG")
  mtext("Nonlinear profile shape after removing subject offset and linear G1 slope",
        side = 3, outer = TRUE, line = 1.28, font = 2, cex = 0.96)
  mtext("Top: group mean and 95% CI of subject residuals; bottom: adjusted shape component; subject is the independent unit",
        side = 3, outer = TRUE, line = 0.05, cex = 0.62, col = neutral)
}

draw_metric_lollipop <- function(values, xlim, title_text, xlab_text, panel_label, refs = NULL, digits = 3) {
  y <- c(2, 1)
  par(mar = c(4.0, 3.7, 3.0, 0.6), mgp = c(2.0, 0.55, 0), tcl = -0.2)
  plot(NA, NA, xlim = xlim, ylim = c(0.5, 2.5), yaxt = "n", bty = "n",
       xlab = xlab_text, ylab = "", cex.axis = 0.66, cex.lab = 0.69)
  if (!is.null(refs)) abline(v = refs, lty = c(3, 2)[seq_along(refs)], col = "#D0D0D0")
  axis(2, at = y, labels = target_order, las = 1, tick = FALSE, cex.axis = 0.70)
  segments(xlim[1], y, values, y, col = target_cols[target_order], lwd = 1.8)
  points(values, y, pch = 21, bg = target_cols[target_order], col = "white", cex = 1.1, lwd = 0.5)
  text(values, y + 0.22, formatC(values, format = "f", digits = digits), cex = 0.61, col = "#333333")
  title(paste0(panel_label, "  ", title_text), adj = 0, font.main = 2, cex.main = 0.80, line = 1.0)
}

draw_component_panel <- function() {
  d <- similarity[match(target_order, similarity$Target), ]
  comps <- c("Offset", "Linear", "Nonlinear shape", "Total")
  vals <- rbind(
    c(d$Adjusted_offset_component_RMS[1], d$Adjusted_linear_component_RMS[1], d$Adjusted_nonlinear_shape_RMSE[1], d$Adjusted_total_profile_RMSE[1]),
    c(d$Adjusted_offset_component_RMS[2], d$Adjusted_linear_component_RMS[2], d$Adjusted_nonlinear_shape_RMSE[2], d$Adjusted_total_profile_RMSE[2])
  )
  rownames(vals) <- target_order
  ymax <- max(vals) * 1.18
  par(mar = c(4.0, 4.2, 3.0, 0.6), mgp = c(2.1, 0.55, 0), tcl = -0.2)
  plot(NA, NA, xlim = c(0.6, 4.4), ylim = c(0, ymax), xaxt = "n", bty = "n",
       xlab = "Orthogonal component", ylab = "RMS magnitude (Fisher-z FC)", cex.axis = 0.66, cex.lab = 0.68)
  axis(1, at = 1:4, labels = comps, cex.axis = 0.60)
  abline(h = pretty(c(0, ymax), 4), col = "#EEEEEE", lwd = 0.7)
  offsets <- c(-0.10, 0.10)
  for (i in seq_along(target_order)) {
    points((1:4) + offsets[i], vals[i, ], pch = if (i == 1) 21 else 24,
           bg = target_cols[target_order[i]], col = "white", cex = 1.05, lwd = 0.5)
  }
  legend("topright", legend = target_order, pch = c(21, 24), pt.bg = target_cols[target_order],
         col = "white", bty = "n", cex = 0.61, pt.cex = 0.9)
  title("c  Adjusted profile decomposition", adj = 0, font.main = 2, cex.main = 0.80, line = 1.0)
}

draw_figure4 <- function() {
  d <- similarity[match(target_order, similarity$Target), ]
  layout(matrix(1:3, nrow = 1), widths = c(0.82, 0.92, 1.35))
  par(oma = c(0.2, 0.2, 3.0, 0.2))
  draw_metric_lollipop(d$Pearson_r_group_mean_profiles, c(0, 1), "Group-mean profile similarity",
                       "Pearson r", "a", refs = c(0.5, 0.9), digits = 3)
  draw_metric_lollipop(d$Raw_profile_RMSE, c(0, max(d$Raw_profile_RMSE) * 1.18), "Raw profile difference",
                       "RMSE (Fisher-z FC)", "b", refs = NULL, digits = 4)
  draw_component_panel()
  mtext("Profile similarity and orthogonal difference decomposition", side = 3, outer = TRUE,
        line = 1.25, font = 2, cex = 0.96)
  mtext("Similarity is descriptive; Pearson r equals correlation after mean-centering and does not capture offset",
        side = 3, outer = TRUE, line = 0.05, cex = 0.62, col = neutral)
}

figures <- list(
  list(name = "Figure_1_SMA_IFG_FC_Profiles", width = 183, height = 150, draw = draw_figure1),
  list(name = "Figure_2_Offset_Slope_Effects", width = 183, height = 92, draw = draw_figure2),
  list(name = "Figure_3_Profile_Shape", width = 183, height = 150, draw = draw_figure3),
  list(name = "Figure_4_Profile_Similarity", width = 183, height = 92, draw = draw_figure4)
)

qc_rows <- list()
for (fig in figures) {
  png_path <- file.path(figure_dir, paste0(fig$name, ".png"))
  pdf_path <- file.path(figure_dir, paste0(fig$name, ".pdf"))
  with_device(open_png, png_path, fig$width, fig$height, fig$draw)
  with_device(open_pdf, pdf_path, fig$width, fig$height, fig$draw)
  qc_rows[[length(qc_rows) + 1]] <- data.frame(
    figure = fig$name,
    width_mm = fig$width,
    height_mm = fig$height,
    png = png_path,
    pdf = pdf_path,
    png_exists = file.exists(png_path),
    pdf_exists = file.exists(pdf_path),
    png_bytes = file.info(png_path)$size,
    pdf_bytes = file.info(pdf_path)$size,
    stringsAsFactors = FALSE
  )
}

qc <- do.call(rbind, qc_rows)
write.table(qc, file.path(table_dir, "figure_export_qc.tsv"), sep = "\t", row.names = FALSE, quote = FALSE, na = "N/A")
message("STEP 4 R figure generation completed.")
print(qc)
