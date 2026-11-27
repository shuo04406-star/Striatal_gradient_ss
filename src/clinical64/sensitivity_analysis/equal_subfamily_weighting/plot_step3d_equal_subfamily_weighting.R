#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE, scipen = 8)

out_dir <- file.path(Sys.getenv("STRIATAL_SUPPLEMENT_ROOT", unset = NA_character_), "STEP3_Equal_Subfamily_Weighting")
model_results <- read.delim(file.path(out_dir, "four_model_mixedlm_results.tsv"), check.names = FALSE)
effect_deltas <- read.delim(file.path(out_dir, "four_model_effect_deltas.tsv"), check.names = FALSE)
profiles <- read.delim(file.path(out_dir, "ten_bin_group_stage_fc_profiles.tsv"), check.names = FALSE)
diff_profiles <- read.delim(file.path(out_dir, "ten_bin_sz_minus_hc_difference_profiles.tsv"), check.names = FALSE)
stability <- read.delim(file.path(out_dir, "profile_stability_metrics.tsv"), check.names = FALSE)
weights <- read.delim(file.path(out_dir, "subfamily_aggregation_weights.tsv"), check.names = FALSE)

model_order <- c("A", "B", "C", "D")
model_labels <- c(
  A = "A  Full / ROI-equal",
  B = "B  Minus Insula / ROI-equal",
  C = "C  Full / subfamily-equal",
  D = "D  Minus Insula / subfamily-equal"
)
model_cols <- c(A = "#2F4B7C", B = "#7394BE", C = "#C65A1E", D = "#E19B5B")
group_cols <- c(HC = "#2878B5", SZ = "#D65F0E")
stage_cols <- c(Anticipation = "#7B3294", Outcome = "#008B8B")
neutral <- "#6B7280"
accent <- "#B2182B"
target_families <- c("Insula", "Thalamus", "SMA", "IFG")

open_png <- function(path, width_mm, height_mm) {
  grDevices::png(
    filename = path, width = width_mm, height = height_mm, units = "mm",
    res = 320, bg = "white", type = "cairo-png"
  )
}

open_pdf <- function(path, width_mm, height_mm) {
  grDevices::pdf(
    file = path, width = width_mm / 25.4, height = height_mm / 25.4,
    family = "Helvetica", onefile = TRUE, useDingbats = FALSE, paper = "special"
  )
}

with_device <- function(device_fun, path, width_mm, height_mm, draw_fun) {
  device_fun(path, width_mm, height_mm)
  old <- par(no.readonly = TRUE)
  on.exit({par(old); dev.off()}, add = TRUE)
  draw_fun()
}

draw_figure1 <- function() {
  d <- model_results[match(model_order, model_results$model), ]
  y <- rev(seq_along(model_order))
  ci_min <- min(d$CI95_low)
  ci_max <- max(d$CI95_high)
  span <- ci_max - ci_min
  xlim <- c(ci_min - 0.10 * span, ci_max + 0.62 * span)

  par(mar = c(4.2, 10.2, 3.8, 1.0), mgp = c(2.2, 0.65, 0), tcl = -0.22)
  plot(NA, NA, xlim = xlim, ylim = c(0.5, 4.5), yaxt = "n", bty = "n",
       xlab = expression(beta~"for Group x Stage x G1"), ylab = "",
       cex.axis = 0.82, cex.lab = 0.88)
  abline(v = 0, lty = 2, lwd = 1.1, col = "#777777")
  abline(h = y, col = "#ECECEC", lwd = 0.8)
  axis(2, at = y, labels = model_labels[model_order], las = 1,
       tick = FALSE, cex.axis = 0.78, line = -0.25)
  for (i in seq_along(model_order)) {
    col <- model_cols[model_order[i]]
    segments(d$CI95_low[i], y[i], d$CI95_high[i], y[i], col = col, lwd = 2.2)
    segments(d$CI95_low[i], y[i] - 0.11, d$CI95_low[i], y[i] + 0.11, col = col, lwd = 1.2)
    segments(d$CI95_high[i], y[i] - 0.11, d$CI95_high[i], y[i] + 0.11, col = col, lwd = 1.2)
    points(d$beta_group_stage_g1[i], y[i], pch = 21, bg = col, col = "white", lwd = 0.7, cex = 1.25)
    text(ci_max + 0.08 * span, y[i],
         sprintf("beta = %.6f   p = %.4g", d$beta_group_stage_g1[i], d$p_wald[i]),
         adj = 0, cex = 0.72, col = "#333333")
  }
  title("Equal-subfamily-weighting sensitivity of the three-way effect",
        adj = 0, font.main = 2, cex.main = 1.03, line = 1.8)
  mtext("Adjusted MixedLM; 95% Wald CI; all intervals cross zero",
        side = 3, adj = 0, line = 0.45, cex = 0.76, col = "#4D4D4D")
}

draw_figure2 <- function() {
  y_low <- min(profiles$mean_FC - profiles$SE_subject)
  y_high <- max(profiles$mean_FC + profiles$SE_subject)
  pad <- 0.06 * (y_high - y_low)
  y_lim <- c(y_low - pad, y_high + pad)

  layout(matrix(seq_len(8), nrow = 4, ncol = 2, byrow = TRUE))
  par(oma = c(3.4, 4.2, 3.2, 0.7))
  for (i in seq_along(model_order)) {
    for (j in seq_along(c("Anticipation", "Outcome"))) {
      model_id <- model_order[i]
      stage_name <- c("Anticipation", "Outcome")[j]
      d <- profiles[profiles$model == model_id & profiles$Stage == stage_name, ]
      par(mar = c(if (i == 4) 3.1 else 1.1, if (j == 1) 3.1 else 1.25, if (i == 1) 2.5 else 1.0, 0.7),
          mgp = c(1.8, 0.45, 0), tcl = -0.18)
      plot(NA, NA, xlim = c(1, 10), ylim = y_lim, bty = "n", xaxt = "n", yaxt = "n",
           xlab = "", ylab = "")
      abline(h = pretty(y_lim, 4), col = "#EFEFEF", lwd = 0.7)
      if (i == 4) axis(1, at = 1:10, cex.axis = 0.65)
      if (j == 1) axis(2, las = 1, cex.axis = 0.62)
      for (group_name in c("HC", "SZ")) {
        g <- d[d$Group == group_name, ][order(d[d$Group == group_name, ]$Bin), ]
        polygon(
          c(g$Bin, rev(g$Bin)),
          c(g$mean_FC - g$SE_subject, rev(g$mean_FC + g$SE_subject)),
          col = grDevices::adjustcolor(group_cols[group_name], alpha.f = 0.14), border = NA
        )
        lines(g$Bin, g$mean_FC, col = group_cols[group_name], lwd = 1.55)
        points(g$Bin, g$mean_FC, pch = 21, bg = group_cols[group_name], col = "white", cex = 0.55, lwd = 0.4)
      }
      if (i == 1) title(stage_name, font.main = 2, cex.main = 0.82, line = 1.1)
      if (j == 1) mtext(model_labels[model_id], side = 2, line = 2.55, cex = 0.66, font = 2)
      if (i == 1 && j == 2) {
        legend("topright", legend = c("HC", "SZ"), col = group_cols, lwd = 1.6,
               pch = 21, pt.bg = group_cols, pt.cex = 0.8, bty = "n", cex = 0.65)
      }
    }
  }
  mtext("G1 bin", side = 1, outer = TRUE, line = 1.7, cex = 0.82)
  mtext("Stage-level FC (mean +/- SE across subjects)", side = 2, outer = TRUE, line = 2.7, cex = 0.82)
  mtext("HC and SZ 10-bin Stage-level FC profiles", side = 3, outer = TRUE,
        line = 1.45, font = 2, cex = 1.0)
  mtext("These curves are Stage-level FC, not Pearson profile-correlation contributions",
        side = 3, outer = TRUE, line = 0.15, cex = 0.70, col = "#4D4D4D")
}

weight_panel <- function(models, stage_name, panel_title, order_names = NULL, show_legend = FALSE) {
  d <- weights[weights$model %in% models & weights$stage == stage_name, ]
  if (is.null(order_names)) {
    base <- d[d$model == models[1], ]
    order_names <- base$subfamily[order(base$subfamily_total_weight, decreasing = FALSE)]
  }
  y <- seq_along(order_names)
  m1 <- d[d$model == models[1], ][match(order_names, d[d$model == models[1], ]$subfamily), ]
  m2 <- d[d$model == models[2], ][match(order_names, d[d$model == models[2], ]$subfamily), ]
  x1 <- 100 * m1$subfamily_total_weight
  x2 <- 100 * m2$subfamily_total_weight
  xmax <- max(c(x1, x2)) * 1.18

  par(mar = c(3.7, 5.4, 3.1, 0.7), mgp = c(1.8, 0.5, 0), tcl = -0.2)
  plot(NA, NA, xlim = c(0, xmax), ylim = c(0.5, length(y) + 0.5),
       yaxt = "n", bty = "n", xlab = "Subfamily total weight (%)", ylab = "",
       cex.axis = 0.67, cex.lab = 0.72)
  abline(v = pretty(c(0, xmax), 4), col = "#EFEFEF", lwd = 0.7)
  axis(2, at = y, labels = FALSE, tick = FALSE)
  usr <- par("usr")
  label_x <- usr[1] - 0.025 * diff(usr[1:2])
  for (k in seq_along(order_names)) {
    is_target <- order_names[k] %in% target_families
    text(label_x, y[k], gsub("_", " ", order_names[k], fixed = TRUE), adj = 1,
         xpd = NA, cex = 0.66, col = if (is_target) accent else "#333333",
         font = if (is_target) 2 else 1)
    segments(x1[k], y[k], x2[k], y[k],
             col = if (is_target) grDevices::adjustcolor(accent, 0.65) else "#B7BDC5", lwd = 1.2)
  }
  points(x1, y, pch = 21, bg = model_cols[models[1]], col = "white", cex = 0.95, lwd = 0.5)
  points(x2, y, pch = 24, bg = model_cols[models[2]], col = "white", cex = 1.0, lwd = 0.5)
  target_idx <- which(order_names %in% target_families)
  points(x1[target_idx], y[target_idx], pch = 21, bg = model_cols[models[1]], col = accent, cex = 1.1, lwd = 1.2)
  points(x2[target_idx], y[target_idx], pch = 24, bg = model_cols[models[2]], col = accent, cex = 1.15, lwd = 1.2)
  title(panel_title, adj = 0, font.main = 2, cex.main = 0.79, line = 1.25)
  if (show_legend) {
    legend("bottomright", legend = model_labels[models], pch = c(21, 24),
           pt.bg = model_cols[models], col = "white", bty = "n", cex = 0.60,
           pt.cex = 0.95)
  }
}

draw_figure3 <- function() {
  layout(matrix(1:3, nrow = 1), widths = c(1, 1, 1.03))
  par(oma = c(0.2, 0, 3.1, 0.2))
  ant_order <- weights$subfamily[weights$model == "A" & weights$stage == "Anticipation"]
  ant_order <- unique(ant_order)
  ant_order <- ant_order[order(weights$subfamily_total_weight[weights$model == "A" & weights$stage == "Anticipation"])]
  out_order <- weights$subfamily[weights$model == "A" & weights$stage == "Outcome"]
  out_order <- unique(out_order)
  out_order <- out_order[order(weights$subfamily_total_weight[weights$model == "A" & weights$stage == "Outcome"])]
  weight_panel(c("A", "C"), "Anticipation", "a  Full: Anticipation", ant_order, TRUE)
  weight_panel(c("A", "C"), "Outcome", "b  Full: Outcome", out_order, TRUE)
  weight_panel(c("B", "D"), "Anticipation", "c  Minus Insula: Anticipation", ant_order, TRUE)
  mtext("Actual Stage aggregation weights under ROI-equal and subfamily-equal schemes",
        side = 3, outer = TRUE, line = 1.45, font = 2, cex = 0.98)
  mtext("Red labels emphasize Insula, Thalamus, SMA and IFG; Outcome weights are unchanged by Insula deletion",
        side = 3, outer = TRUE, line = 0.15, cex = 0.68, col = "#4D4D4D")
}

draw_figure4 <- function() {
  comp_order <- c("B_minus_A", "C_minus_A", "D_minus_C", "D_minus_B")
  comp_labels <- c(B_minus_A = "B - A", C_minus_A = "C - A", D_minus_C = "D - C", D_minus_B = "D - B")
  d_effect <- effect_deltas[match(comp_order, effect_deltas$comparison), ]
  d_prof <- stability[stability$curve_type == "SZ_minus_HC", ]
  d_prof$comparison <- factor(d_prof$comparison, levels = comp_order)
  d_prof$stage <- factor(d_prof$stage, levels = c("Anticipation", "Outcome"))
  d_prof <- d_prof[order(d_prof$comparison, d_prof$stage), ]

  layout(matrix(1:4, nrow = 2, byrow = TRUE))
  par(oma = c(0.3, 0.3, 3.0, 0.2))

  par(mar = c(4.1, 4.0, 2.5, 0.8), mgp = c(2.0, 0.55, 0), tcl = -0.2)
  y <- rev(seq_along(comp_order))
  x <- d_effect$signed_delta_beta
  xlim <- range(c(x, 0)) + c(-1, 1) * 0.12 * diff(range(c(x, 0)))
  plot(NA, NA, xlim = xlim, ylim = c(0.5, 4.5), yaxt = "n", bty = "n",
       xlab = expression("Signed "*Delta*beta), ylab = "", cex.axis = 0.70, cex.lab = 0.78)
  abline(v = 0, lty = 2, col = "#777777")
  axis(2, at = y, labels = comp_labels[comp_order], las = 1, tick = FALSE, cex.axis = 0.72)
  segments(0, y, x, y, col = ifelse(x < 0, "#2166AC", "#B2182B"), lwd = 2)
  points(x, y, pch = 21, bg = ifelse(x < 0, "#2166AC", "#B2182B"), col = "white", cex = 1.0)
  title("a  Three-way effect change", adj = 0, font.main = 2, cex.main = 0.82, line = 1.0)

  par(mar = c(4.1, 4.0, 2.5, 0.8), mgp = c(2.0, 0.55, 0), tcl = -0.2)
  y <- rev(seq_along(comp_order))
  rmse_max <- max(d_prof$RMSE) * 1.12
  plot(NA, NA, xlim = c(0, rmse_max), ylim = c(0.5, 4.5), yaxt = "n", bty = "n",
       xlab = "RMSE of SZ - HC curve", ylab = "", cex.axis = 0.70, cex.lab = 0.74)
  axis(2, at = y, labels = comp_labels[comp_order], las = 1, tick = FALSE, cex.axis = 0.72)
  for (s in c("Anticipation", "Outcome")) {
    sub <- d_prof[d_prof$stage == s, ][match(comp_order, as.character(d_prof[d_prof$stage == s, ]$comparison)), ]
    off <- if (s == "Anticipation") 0.09 else -0.09
    points(sub$RMSE, y + off, pch = if (s == "Anticipation") 21 else 24,
           bg = stage_cols[s], col = "white", cex = 1.0, lwd = 0.5)
  }
  legend("bottomright", legend = names(stage_cols), pch = c(21, 24), pt.bg = stage_cols,
         col = "white", bty = "n", cex = 0.62, pt.cex = 0.9)
  title("b  Difference-profile RMSE", adj = 0, font.main = 2, cex.main = 0.82, line = 1.0)

  row_labels <- paste(comp_labels[as.character(d_prof$comparison)], substr(as.character(d_prof$stage), 1, 3))
  y8 <- rev(seq_len(nrow(d_prof)))
  par(mar = c(4.1, 5.3, 2.5, 0.8), mgp = c(2.0, 0.55, 0), tcl = -0.2)
  metric_max <- max(c(abs(d_prof$signed_offset_change), d_prof$shape_RMSE_after_offset_removal)) * 1.12
  plot(NA, NA, xlim = c(0, metric_max), ylim = c(0.5, length(y8) + 0.5), yaxt = "n", bty = "n",
       xlab = "Absolute FC change", ylab = "", cex.axis = 0.68, cex.lab = 0.74)
  axis(2, at = y8, labels = row_labels, las = 1, tick = FALSE, cex.axis = 0.60)
  x_offset <- abs(d_prof$signed_offset_change)
  x_shape <- d_prof$shape_RMSE_after_offset_removal
  segments(x_offset, y8, x_shape, y8, col = "#B7BDC5", lwd = 1)
  points(x_offset, y8, pch = 21, bg = "#B2182B", col = "white", cex = 0.85)
  points(x_shape, y8, pch = 24, bg = "#2166AC", col = "white", cex = 0.9)
  legend("bottomright", legend = c("|offset|", "shape RMSE"), pch = c(21, 24),
         pt.bg = c("#B2182B", "#2166AC"), col = "white", bty = "n", cex = 0.60)
  title("c  Offset versus shape change", adj = 0, font.main = 2, cex.main = 0.82, line = 1.0)

  par(mar = c(4.1, 5.3, 2.5, 0.8), mgp = c(2.0, 0.55, 0), tcl = -0.2)
  plot(NA, NA, xlim = c(0, 1.02), ylim = c(0.5, length(y8) + 0.5), yaxt = "n", bty = "n",
       xlab = "Profile correlation", ylab = "", cex.axis = 0.68, cex.lab = 0.74)
  abline(v = c(0.5, 0.9), col = "#E1E1E1", lty = c(3, 2))
  axis(2, at = y8, labels = row_labels, las = 1, tick = FALSE, cex.axis = 0.60)
  point_cols <- stage_cols[as.character(d_prof$stage)]
  points(d_prof$profile_correlation, y8, pch = 21, bg = point_cols, col = "white", cex = 0.95)
  title("d  Difference-profile correlation", adj = 0, font.main = 2, cex.main = 0.82, line = 1.0)

  mtext("Effect-size change and Stage-level profile stability",
        side = 3, outer = TRUE, line = 1.35, font = 2, cex = 0.98)
  mtext("Profile metrics use 10-bin SZ - HC Stage-level FC curves; no bootstrap inference",
        side = 3, outer = TRUE, line = 0.10, cex = 0.68, col = "#4D4D4D")
}

figures <- list(
  list(name = "Figure1_four_model_beta_CI", width = 183, height = 105, draw = draw_figure1),
  list(name = "Figure2_four_model_HC_SZ_profiles", width = 183, height = 225, draw = draw_figure2),
  list(name = "Figure3_subfamily_weight_changes", width = 183, height = 132, draw = draw_figure3),
  list(name = "Figure4_effect_profile_stability", width = 183, height = 180, draw = draw_figure4)
)

qc_rows <- list()
for (fig in figures) {
  png_path <- file.path(out_dir, paste0(fig$name, ".png"))
  pdf_path <- file.path(out_dir, paste0(fig$name, ".pdf"))
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
write.table(qc, file.path(out_dir, "figure_export_qc.tsv"), sep = "\t",
            row.names = FALSE, quote = FALSE, na = "N/A")
message("STEP 3D R figure generation completed.")
print(qc)
