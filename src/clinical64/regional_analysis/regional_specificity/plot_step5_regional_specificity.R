options(stringsAsFactors = FALSE, scipen = 8)

out_dir <- file.path(Sys.getenv("STRIATAL_SUPPLEMENT_ROOT", unset = NA_character_), "STEP5_Regional_Specificity")
table_dir <- file.path(out_dir, "tables")
figure_dir <- file.path(out_dir, "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

main <- read.delim(file.path(table_dir, "STEP5_main_regional_specificity.tsv"), check.names = FALSE)
subject <- read.delim(file.path(table_dir, "STEP5_subject_level_contrasts.tsv"), check.names = FALSE)
components <- read.delim(file.path(table_dir, "STEP5_target_vs_reference.tsv"), check.names = FALSE)
bootstrap <- read.delim(file.path(table_dir, "STEP5_bootstrap_results.tsv"), check.names = FALSE)
loo <- read.delim(file.path(table_dir, "STEP5_leave_one_subject_out.tsv"), check.names = FALSE)

target_order <- c("SMA", "IFG")
target_cols <- c(SMA = "#7A5195", IFG = "#2A9D8F")
group_cols <- c(HC = "#3B6FB6", SZ = "#D95F02")
neutral <- "#4D4D4D"
light <- "#EAEAEA"

open_png <- function(path, width_mm, height_mm) {
  grDevices::png(path, width = width_mm, height = height_mm, units = "mm",
                 res = 320, bg = "white", type = "cairo-png")
}

open_pdf <- function(path, width_mm, height_mm) {
  grDevices::pdf(path, width = width_mm / 25.4, height = height_mm / 25.4,
                 family = "Helvetica", useDingbats = FALSE, onefile = TRUE,
                 paper = "special")
}

with_device <- function(device_fun, path, width_mm, height_mm, draw_fun) {
  device_fun(path, width_mm, height_mm)
  old <- par(no.readonly = TRUE)
  on.exit({par(old); dev.off()}, add = TRUE)
  draw_fun()
}

effect_label <- function(beta, low, high, p) {
  sprintf("beta = %.5f  [%.5f, %.5f]\nHolm p = %.3g", beta, low, high, p)
}

draw_figure1 <- function() {
  d <- main[match(target_order, main$`Target Subfamily`), ]
  y <- c(2, 1)
  x_min <- min(d$`CI Lower`) - 0.008
  x_max <- 0.038
  par(mar = c(4.4, 7.4, 2.8, 0.7), mgp = c(2.3, 0.65, 0), tcl = -0.22)
  plot(NA, NA, xlim = c(x_min, x_max), ylim = c(0.45, 2.55), yaxt = "n",
       bty = "n", xlab = "Adjusted SZ - HC difference in Target-minus-Other\nPearson contribution per ROI",
       ylab = "", cex.axis = 0.75, cex.lab = 0.78)
  abline(v = 0, lty = 2, col = "#777777", lwd = 0.9)
  abline(h = y, col = light, lwd = 0.8)
  labels <- c("SMA vs other Anticipation\n(1 vs 42 ROIs)",
              "IFG vs other Outcome\n(6 vs 13 ROIs)")
  axis(2, at = y, labels = labels, las = 1, tick = FALSE, cex.axis = 0.72, line = -0.3)
  for (i in seq_len(nrow(d))) {
    col <- target_cols[d$`Target Subfamily`[i]]
    segments(d$`CI Lower`[i], y[i], d$`CI Upper`[i], y[i], col = col, lwd = 2.4)
    segments(d$`CI Lower`[i], y[i] - 0.08, d$`CI Lower`[i], y[i] + 0.08, col = col, lwd = 1.0)
    segments(d$`CI Upper`[i], y[i] - 0.08, d$`CI Upper`[i], y[i] + 0.08, col = col, lwd = 1.0)
    points(d$`Adjusted Group Beta`[i], y[i], pch = 21, bg = col, col = "white", cex = 1.3, lwd = 0.6)
    text(0.004, y[i], effect_label(d$`Adjusted Group Beta`[i], d$`CI Lower`[i], d$`CI Upper`[i], d$`Holm p`[i]),
         adj = c(0, 0.5), cex = 0.67, col = neutral)
  }
  title("Regional specificity of Pearson profile-correlation contributions", adj = 0,
        font.main = 2, cex.main = 0.98, line = 1.0)
  mtext("Subject-level OLS with Age, Sex, MeanFD; HC3 95% Wald CI; Holm correction across two post-hoc contrasts",
        side = 3, adj = 0, line = 0.05, cex = 0.63, col = neutral)
}

draw_subject_panel <- function(target, panel_label) {
  value_col <- if (target == "SMA") "SMA contrast L" else "IFG contrast L"
  values <- subject[[value_col]]
  groups <- subject$Group
  dmain <- main[main$`Target Subfamily` == target, ]
  ylim <- range(values)
  pad <- 0.11 * diff(ylim)
  if (pad == 0) pad <- 0.01
  par(mar = c(3.8, 4.2, 3.5, 0.8), mgp = c(2.15, 0.60, 0), tcl = -0.22)
  plot(NA, NA, xlim = c(0.55, 2.45), ylim = ylim + c(-pad, pad), xaxt = "n",
       bty = "n", xlab = "Group", ylab = "Target-minus-Other Pearson contribution\n(per ROI)",
       cex.axis = 0.68, cex.lab = 0.70)
  axis(1, at = 1:2, labels = c("HC", "SZ"), tick = FALSE, cex.axis = 0.72)
  abline(h = 0, lty = 2, col = "#888888")
  for (j in seq_along(c("HC", "SZ"))) {
    g <- c("HC", "SZ")[j]
    vg <- sort(values[groups == g])
    xg <- j + seq(-0.12, 0.12, length.out = length(vg))
    boxplot(vg, at = j, add = TRUE, axes = FALSE, outline = FALSE, boxwex = 0.42,
            col = adjustcolor(group_cols[g], alpha.f = 0.12), border = group_cols[g], lwd = 1.0)
    points(xg, vg, pch = 21, bg = adjustcolor(group_cols[g], alpha.f = 0.72),
           col = "white", cex = 0.72, lwd = 0.35)
    mn <- mean(vg)
    ci <- mn + c(-1, 1) * qt(0.975, length(vg) - 1) * sd(vg) / sqrt(length(vg))
    segments(j - 0.20, mn, j + 0.20, mn, col = group_cols[g], lwd = 2.0)
    segments(j, ci[1], j, ci[2], col = group_cols[g], lwd = 1.4)
  }
  title(sprintf("%s  %s", panel_label, target), adj = 0, font.main = 2, cex.main = 0.86, line = 1.15)
  mtext(sprintf("Adjusted beta = %.5f; Holm p = %.3g", dmain$`Adjusted Group Beta`, dmain$`Holm p`),
        side = 3, adj = 0, line = 0.10, cex = 0.60, col = neutral)
}

draw_figure2 <- function() {
  layout(matrix(1:2, nrow = 1))
  par(oma = c(0.2, 0.3, 2.8, 0.3))
  draw_subject_panel("SMA", "a")
  draw_subject_panel("IFG", "b")
  mtext("Subject-level regional contrasts", side = 3, outer = TRUE, line = 1.2,
        adj = 0, font = 2, cex = 0.98)
  mtext("Points are unadjusted subject-level contrasts; horizontal bar and whisker show group mean and descriptive 95% CI",
        side = 3, outer = TRUE, line = 0.02, adj = 0, cex = 0.62, col = neutral)
}

draw_component_panel <- function(target, panel_label) {
  d <- components[components$Target == target & components$Group == "Adjusted SZ-HC", ]
  component_order <- c("Target", "Other", "Target-minus-Other")
  d <- d[match(component_order, d$Component), ]
  y <- 3:1
  xlim <- range(c(d$`CI Lower`, d$`CI Upper`, 0))
  pad <- 0.18 * diff(xlim)
  if (pad == 0) pad <- 0.01
  par(mar = c(4.0, 4.8, 3.4, 0.8), mgp = c(2.1, 0.60, 0), tcl = -0.22)
  plot(NA, NA, xlim = xlim + c(-pad, pad), ylim = c(0.45, 3.55), yaxt = "n",
       bty = "n", xlab = "Adjusted SZ - HC beta\n(per-ROI Pearson contribution)", ylab = "",
       cex.axis = 0.68, cex.lab = 0.70)
  abline(v = 0, lty = 2, col = "#777777")
  abline(h = y, col = light, lwd = 0.7)
  axis(2, at = y, labels = c("Target", "Other ROI mean", "Target - Other"),
       las = 1, tick = FALSE, cex.axis = 0.66)
  for (i in seq_len(nrow(d))) {
    col <- if (d$Component[i] == "Target-minus-Other") target_cols[target] else adjustcolor(target_cols[target], alpha.f = 0.62)
    segments(d$`CI Lower`[i], y[i], d$`CI Upper`[i], y[i], col = col, lwd = 2.0)
    points(d$`Adjusted Group Effect`[i], y[i], pch = 21, bg = col, col = "white", cex = 1.05, lwd = 0.5)
  }
  title(sprintf("%s  %s", panel_label, target), adj = 0, font.main = 2, cex.main = 0.86, line = 1.15)
  mtext(if (target == "SMA") "Target: 1 ROI; Other Anticipation: 42 ROIs" else "Target: 6 ROIs; Other Outcome: 13 ROIs",
        side = 3, adj = 0, line = 0.10, cex = 0.60, col = neutral)
}

draw_figure3 <- function() {
  layout(matrix(1:2, nrow = 1))
  par(oma = c(0.2, 0.3, 2.8, 0.3))
  draw_component_panel("SMA", "a")
  draw_component_panel("IFG", "b")
  mtext("Target, Stage-matched reference, and direct regional contrast", side = 3,
        outer = TRUE, line = 1.2, adj = 0, font = 2, cex = 0.98)
  mtext("All effects use per-ROI Pearson contribution; error bars are HC3 95% Wald confidence intervals",
        side = 3, outer = TRUE, line = 0.02, adj = 0, cex = 0.62, col = neutral)
}

draw_robustness_panel <- function(target, panel_label) {
  d <- loo[loo$Target == target, ]
  b <- bootstrap[bootstrap$Target == target, ]
  xlim <- range(c(d$`Beta L`, b$`HC3 CI Lower`, b$`HC3 CI Upper`, b$`Bootstrap CI Lower`, b$`Bootstrap CI Upper`, 0))
  pad <- 0.10 * diff(xlim)
  if (pad == 0) pad <- 0.01
  par(mar = c(4.1, 4.7, 3.5, 0.8), mgp = c(2.15, 0.60, 0), tcl = -0.22)
  plot(NA, NA, xlim = xlim + c(-pad, pad), ylim = c(0.55, 2.65), yaxt = "n",
       bty = "n", xlab = "Adjusted SZ - HC beta for Target-minus-Other", ylab = "",
       cex.axis = 0.68, cex.lab = 0.70)
  abline(v = 0, lty = 2, col = "#777777")
  axis(2, at = c(2.15, 1.55, 0.95), labels = c("Bootstrap 95% CI", "HC3 95% CI", "LOO refits"),
       las = 1, tick = FALSE, cex.axis = 0.63)
  col <- target_cols[target]
  segments(b$`Bootstrap CI Lower`, 2.15, b$`Bootstrap CI Upper`, 2.15, col = col, lwd = 4.0)
  points(b$`Bootstrap Median`, 2.15, pch = 23, bg = "white", col = col, cex = 1.0, lwd = 1.1)
  segments(b$`HC3 CI Lower`, 1.55, b$`HC3 CI Upper`, 1.55, col = col, lwd = 2.3)
  points(b$`Original Adjusted Beta L`, 1.55, pch = 21, bg = col, col = "white", cex = 1.1, lwd = 0.5)
  yjit <- seq(0.79, 1.11, length.out = nrow(d))
  points(sort(d$`Beta L`), yjit, pch = 21, bg = adjustcolor(col, alpha.f = 0.55),
         col = "white", cex = 0.58, lwd = 0.25)
  abline(v = b$`Original Adjusted Beta L`, col = col, lty = 3, lwd = 1.0)
  title(sprintf("%s  %s", panel_label, target), adj = 0, font.main = 2, cex.main = 0.86, line = 1.15)
  mtext(sprintf("LOO beta range %.5f to %.5f; direction retained %d/64",
                min(d$`Beta L`), max(d$`Beta L`), sum(d$`Direction Matches Full` == "True" | d$`Direction Matches Full` == TRUE)),
        side = 3, adj = 0, line = 0.10, cex = 0.58, col = neutral)
}

draw_figure4 <- function() {
  layout(matrix(1:2, nrow = 1))
  par(oma = c(0.2, 0.3, 2.8, 0.3))
  draw_robustness_panel("SMA", "a")
  draw_robustness_panel("IFG", "b")
  mtext("Influence and uncertainty of regional-specificity effects", side = 3,
        outer = TRUE, line = 1.2, adj = 0, font = 2, cex = 0.98)
  mtext("Full-sample beta shown by dotted vertical line; stratified bootstrap used 5000 resamples with seed 20261012",
        side = 3, outer = TRUE, line = 0.02, adj = 0, cex = 0.62, col = neutral)
}

figures <- list(
  list(name = "Figure_1_Regional_Specificity", width = 183, height = 92, draw = draw_figure1),
  list(name = "Figure_2_Subject_Regional_Contrasts", width = 183, height = 104, draw = draw_figure2),
  list(name = "Figure_3_Target_vs_Reference", width = 183, height = 98, draw = draw_figure3),
  list(name = "Figure_4_Robustness", width = 183, height = 98, draw = draw_figure4)
)

qc <- list()
for (f in figures) {
  png_path <- file.path(figure_dir, paste0(f$name, ".png"))
  pdf_path <- file.path(figure_dir, paste0(f$name, ".pdf"))
  with_device(open_png, png_path, f$width, f$height, f$draw)
  with_device(open_pdf, pdf_path, f$width, f$height, f$draw)
  qc[[length(qc) + 1]] <- data.frame(
    figure = f$name,
    width_mm = f$width,
    height_mm = f$height,
    dpi_png = 320,
    png = png_path,
    pdf = pdf_path,
    png_exists = file.exists(png_path),
    pdf_exists = file.exists(pdf_path),
    png_bytes = file.info(png_path)$size,
    pdf_bytes = file.info(pdf_path)$size
  )
}

qc_frame <- do.call(rbind, qc)
write.table(qc_frame, file.path(table_dir, "STEP5_figure_export_qc.tsv"), sep = "\t",
            row.names = FALSE, quote = FALSE)
print(qc_frame)
cat("STEP 5 R figure generation completed.\n")
