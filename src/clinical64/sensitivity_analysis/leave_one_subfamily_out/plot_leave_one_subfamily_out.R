#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE, scipen = 8)
# Use the configured R library (R_LIBS_USER).

out_dir <- file.path(Sys.getenv("STRIATAL_SUPPLEMENT_ROOT", unset = NA_character_), "STEP3_LocalContribution_AllSubfamilies")
table1_path <- file.path(out_dir, "table1_sma_ifg_stability_overview.tsv")
table2_path <- file.path(out_dir, "table2_complete_contribution_results.tsv")

t1 <- read.delim(table1_path, check.names = FALSE, na.strings = c("N/A", ""))
t2 <- read.delim(table2_path, check.names = FALSE, na.strings = c("N/A", ""))

required_t1 <- c(
  "deleted_subfamily", "SMA_beta", "SMA_FDR_q_13_fixed",
  "IFG_beta", "IFG_FDR_q_13_fixed"
)
required_t2 <- c(
  "deleted_subfamily", "tested_subfamily", "beta_SZ_minus_HC",
  "CI95_low", "CI95_high", "q_FDR_13_fixed", "delta_beta_vs_full"
)
stopifnot(all(required_t1 %in% names(t1)), all(required_t2 %in% names(t2)))

condition_order <- t1$deleted_subfamily
assigned_order <- condition_order[condition_order != "Full"]

pretty_label <- function(x) {
  x <- gsub("_", " ", x, fixed = TRUE)
  x
}

open_png <- function(path, width_mm, height_mm) {
  grDevices::png(
    filename = path,
    width = width_mm,
    height = height_mm,
    units = "mm",
    res = 320,
    bg = "white",
    type = "cairo-png"
  )
}

open_pdf <- function(path, width_mm, height_mm) {
  grDevices::pdf(
    file = path,
    width = width_mm / 25.4,
    height = height_mm / 25.4,
    family = "Helvetica",
    onefile = TRUE,
    useDingbats = FALSE,
    paper = "special"
  )
}

draw_forest_panel <- function(target, panel_letter, panel_title, color) {
  d <- t2[t2$tested_subfamily == target & t2$deleted_subfamily %in% condition_order, ]
  d <- d[match(condition_order, d$deleted_subfamily), ]
  y <- rev(seq_along(condition_order))
  xlim <- range(c(d$CI95_low, d$CI95_high, 0), na.rm = TRUE)
  pad <- diff(xlim) * 0.09
  xlim <- xlim + c(-pad, pad)

  par(mar = c(3.5, 7.1, 2.7, 1.1), mgp = c(1.9, 0.55, 0), tcl = -0.22)
  plot(
    NA, NA, xlim = xlim, ylim = c(0.5, length(y) + 0.5),
    xlab = expression(beta~"(adjusted SZ - HC)"), ylab = "", yaxt = "n",
    bty = "n", cex.axis = 0.82, cex.lab = 0.88
  )
  abline(v = 0, col = "#777777", lty = 2, lwd = 1)
  abline(h = y, col = "#ECECEC", lwd = 0.8)
  axis(2, at = y, labels = pretty_label(condition_order), las = 1,
       tick = FALSE, cex.axis = 0.76, line = -0.3)

  for (i in seq_along(y)) {
    if (!is.na(d$beta_SZ_minus_HC[i])) {
      is_full <- condition_order[i] == "Full"
      point_col <- if (is_full) "#202020" else color
      segments(d$CI95_low[i], y[i], d$CI95_high[i], y[i],
               col = point_col, lwd = if (is_full) 2.4 else 1.7)
      segments(d$CI95_low[i], y[i] - 0.13, d$CI95_low[i], y[i] + 0.13,
               col = point_col, lwd = 1.2)
      segments(d$CI95_high[i], y[i] - 0.13, d$CI95_high[i], y[i] + 0.13,
               col = point_col, lwd = 1.2)
      points(d$beta_SZ_minus_HC[i], y[i], pch = 21,
             bg = point_col, col = "white", lwd = 0.7,
             cex = if (is_full) 1.15 else 1.0)
    } else {
      text(mean(xlim), y[i], "N/A (subfamily deleted)", col = "#777777",
           cex = 0.72, font = 3)
    }
  }
  mtext(paste0(panel_letter, "  ", panel_title), side = 3, adj = 0,
        line = 0.75, font = 2, cex = 0.97)
}

draw_figure1 <- function(device_fun, path) {
  device_fun(path, 183, 235)
  old <- par(no.readonly = TRUE)
  on.exit({par(old); dev.off()}, add = TRUE)
  layout(matrix(c(1, 2), nrow = 2), heights = c(1, 1))
  draw_forest_panel("SMA", "a", "SMA contribution effect", "#2C7FB8")
  draw_forest_panel("IFG", "b", "IFG contribution effect", "#D95F0E")
  mtext("Leave-one-subfamily-out effects on Pearson profile-correlation contribution",
        side = 3, outer = TRUE, line = -1.25, font = 2, cex = 1.05)
}

fig1_png <- file.path(out_dir, "Figure1_SMA_IFG_beta_CI_leave_one_out.png")
fig1_pdf <- file.path(out_dir, "Figure1_SMA_IFG_beta_CI_leave_one_out.pdf")
draw_figure1(open_png, fig1_png)
draw_figure1(open_pdf, fig1_pdf)

heat <- matrix(
  NA_real_, nrow = length(assigned_order), ncol = length(assigned_order),
  dimnames = list(assigned_order, assigned_order)
)
heat_rows <- t2[
  t2$deleted_subfamily %in% assigned_order &
    t2$tested_subfamily %in% assigned_order,
]
for (i in seq_len(nrow(heat_rows))) {
  heat[heat_rows$deleted_subfamily[i], heat_rows$tested_subfamily[i]] <-
    heat_rows$delta_beta_vs_full[i]
}
lim <- max(abs(heat), na.rm = TRUE)
pal <- grDevices::colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(200)
breaks <- seq(-lim, lim, length.out = length(pal) + 1)

draw_figure2 <- function(device_fun, path) {
  device_fun(path, 183, 185)
  old <- par(no.readonly = TRUE)
  on.exit({par(old); dev.off()}, add = TRUE)
  layout(matrix(c(1, 2), nrow = 2), heights = c(12, 1.25))

  par(mar = c(8.1, 7.6, 3.2, 1.2), xpd = NA)
  z <- t(heat[rev(seq_len(nrow(heat))), , drop = FALSE])
  image(
    x = seq_along(assigned_order), y = seq_along(assigned_order), z = z,
    col = pal, breaks = breaks, axes = FALSE, xlab = "", ylab = "",
    useRaster = TRUE
  )
  axis(1, at = seq_along(assigned_order), labels = pretty_label(assigned_order),
       las = 2, tick = FALSE, cex.axis = 0.67, line = -0.6)
  axis(2, at = seq_along(assigned_order),
       labels = pretty_label(rev(assigned_order)), las = 1,
       tick = FALSE, cex.axis = 0.70, line = -0.5)
  abline(v = seq(1.5, length(assigned_order) - 0.5, by = 1),
         h = seq(1.5, length(assigned_order) - 0.5, by = 1),
         col = grDevices::adjustcolor("white", 0.80), lwd = 0.65)
  for (i in seq_along(assigned_order)) {
    y_i <- length(assigned_order) - i + 1
    rect(i - 0.49, y_i - 0.49, i + 0.49, y_i + 0.49,
         col = "white", border = "#BDBDBD", lwd = 0.7)
  }
  mtext("Tested subfamily", side = 1, line = 6.65, cex = 0.82)
  mtext("Deleted subfamily", side = 2, line = 6.45, cex = 0.82)
  title("Signed change in group effect relative to Full", adj = 0,
        font.main = 2, cex.main = 1.02, line = 1.25)
  mtext("Color encodes delta beta; diagonal cells are not estimable",
        side = 3, adj = 0, line = 0.15, cex = 0.75, col = "#4D4D4D")

  par(mar = c(2.8, 7.6, 0.5, 1.2))
  color_x <- seq(-lim, lim, length.out = 200)
  image(x = color_x, y = 1, z = matrix(color_x, nrow = 200, ncol = 1),
        col = pal, breaks = breaks, axes = FALSE, xlab = "", ylab = "")
  axis(1, at = c(-lim, -lim / 2, 0, lim / 2, lim),
       labels = formatC(c(-lim, -lim / 2, 0, lim / 2, lim), format = "f", digits = 3),
       cex.axis = 0.72, tcl = -0.22, mgp = c(1.5, 0.45, 0))
  mtext(expression(Delta*beta~"relative to Full"), side = 1, line = 1.6, cex = 0.76)
}

fig2_png <- file.path(out_dir, "Figure2_all_subfamilies_delta_beta_heatmap.png")
fig2_pdf <- file.path(out_dir, "Figure2_all_subfamilies_delta_beta_heatmap.pdf")
draw_figure2(open_png, fig2_png)
draw_figure2(open_pdf, fig2_pdf)

draw_q_panel <- function(target, panel_letter, color) {
  q_col <- paste0(target, "_FDR_q_13_fixed")
  beta_col <- paste0(target, "_beta")
  q <- t1[[q_col]]
  beta <- t1[[beta_col]]
  x <- seq_along(condition_order)
  finite_q <- q[is.finite(q) & q > 0]
  ylim <- c(min(finite_q) / 1.7, 0.12)

  par(mar = c(9.0, 4.4, 2.7, 1.0), mgp = c(2.2, 0.62, 0), tcl = -0.24)
  plot(
    x, q, type = "n", log = "y", ylim = ylim,
    xaxt = "n", xlab = "", ylab = "BH-FDR q (fixed 13-test family)",
    bty = "n", cex.axis = 0.76, cex.lab = 0.82
  )
  axis(1, at = x, labels = pretty_label(condition_order), las = 2,
       tick = FALSE, cex.axis = 0.64, line = -0.55)
  mtext("Deleted subfamily", side = 1, line = 7.7, cex = 0.82)
  abline(h = 0.05, col = "#B2182B", lty = 2, lwd = 1.5)
  abline(h = t1[[q_col]][1], col = "#666666", lty = 3, lwd = 1)
  ok <- is.finite(q)
  segments(x[ok], q[ok] / 1.12, x[ok], q[ok] * 1.12,
           col = color, lwd = 1.2)
  points(x[ok], q[ok], pch = 21, bg = color, col = "white", lwd = 0.7, cex = 1.08)
  direction_symbol <- ifelse(beta[ok] < 0, "-", "+")
  text(x[ok], q[ok], labels = direction_symbol, col = "white",
       cex = 0.57, font = 2)
  if (any(!ok)) {
    na_y <- ifelse(x[!ok] > 10, 0.006, 0.03)
    text(x[!ok], na_y, "N/A", srt = 90,
         cex = 0.66, col = "#777777", font = 3)
  }
  text(1.0, 0.05, "q = 0.05", pos = 4,
       col = "#B2182B", cex = 0.67)
  mtext(paste0(panel_letter, "  ", target), side = 3, adj = 0,
        line = 0.8, font = 2, cex = 0.98)
  legend("topright", legend = c("Negative beta", "Positive beta", "Full q"),
         pch = c(21, 21, NA), pt.bg = c(color, "#B2182B", NA),
         col = c("white", "white", "#666666"), lty = c(NA, NA, 3),
         lwd = c(NA, NA, 1), bty = "o", bg = "white", box.col = "white", cex = 0.67,
         pt.cex = 1.0, inset = c(0.01, 0.01))
}

draw_figure3 <- function(device_fun, path) {
  device_fun(path, 183, 210)
  old <- par(no.readonly = TRUE)
  on.exit({par(old); dev.off()}, add = TRUE)
  layout(matrix(c(1, 2), nrow = 2), heights = c(1, 1))
  par(oma = c(0.2, 0, 2.4, 0))
  draw_q_panel("SMA", "a", "#2C7FB8")
  draw_q_panel("IFG", "b", "#D95F0E")
  mtext("FDR stability after leave-one-subfamily-out recomputation",
        side = 3, outer = TRUE, line = 0.55, font = 2, cex = 1.05)
}

fig3_png <- file.path(out_dir, "Figure3_SMA_IFG_FDR_q_leave_one_out.png")
fig3_pdf <- file.path(out_dir, "Figure3_SMA_IFG_FDR_q_leave_one_out.pdf")
draw_figure3(open_png, fig3_png)
draw_figure3(open_pdf, fig3_pdf)

qc <- data.frame(
  figure = c("Figure 1", "Figure 2", "Figure 3"),
  png = c(fig1_png, fig2_png, fig3_png),
  pdf = c(fig1_pdf, fig2_pdf, fig3_pdf),
  png_exists = file.exists(c(fig1_png, fig2_png, fig3_png)),
  pdf_exists = file.exists(c(fig1_pdf, fig2_pdf, fig3_pdf)),
  png_bytes = file.info(c(fig1_png, fig2_png, fig3_png))$size,
  pdf_bytes = file.info(c(fig1_pdf, fig2_pdf, fig3_pdf))$size,
  stringsAsFactors = FALSE
)
write.table(qc, file.path(out_dir, "figure_export_qc.tsv"), sep = "\t",
            row.names = FALSE, quote = FALSE, na = "N/A")

message("R figure generation completed.")
print(qc)
