#!/usr/bin/env Rscript
# Draw the seven figures from their CSV source data with the project's R palette.
suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})
argv <- commandArgs(trailingOnly = TRUE)
stopifnot(length(argv) == 2L, argv[2] %in% c("png", "pdf"))
out <- normalizePath(argv[1], winslash = "/", mustWork = TRUE)
figure_dir <- file.path(out, "figures")
data_dir <- file.path(figure_dir, "source_data")
format <- argv[2]
hc <- "#4C78A8"
sz <- "#D16D5A"
deep_sz <- "#AF4934"
gray <- "#535353"
faint <- "#858585"
family <- "Arial"
if (!grepl("arial", basename(systemfonts::match_fonts(family)$path[1]), ignore.case = TRUE)) {
  family <- "sans"
}
theme_set(theme_classic(base_size = 8.6, base_family = family) +
  theme(
    text = element_text(colour = "#222222"),
    axis.text = element_text(colour = "#333333"),
    axis.line = element_line(colour = "#2B2B2B", linewidth = .3),
    axis.ticks = element_line(colour = "#2B2B2B", linewidth = .3),
    axis.ticks.length = grid::unit(1.5, "mm"),
    plot.title = element_text(size = 10, face = "bold", margin = margin(b = 3)),
    plot.subtitle = element_text(size = 8.1, colour = gray, lineheight = 1.08, margin = margin(b = 5)),
    plot.caption = element_text(size = 7.3, colour = gray, hjust = 0, lineheight = 1.1),
    plot.tag = element_text(size = 11, face = "bold"),
    strip.text = element_text(face = "bold", size = 8.6),
    strip.background = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(size = 8.0),
    legend.text = element_text(size = 7.6),
    legend.key.height = grid::unit(3, "mm"),
    plot.margin = margin(4, 5, 4, 5)
  ))
scale_levels <- c("TEPS_consummatory", "TEPS_total")
pretty <- c(TEPS_anticipatory = "TEPS anticipatory", TEPS_consummatory = "TEPS consummatory",
  TEPS_total = "TEPS total", ACIPS_anticipatory = "ACIPS anticipatory",
  ACIPS_consummatory = "ACIPS consummatory", ACIPS_total = "ACIPS total")
source_data <- function(name) {
  utils::read.csv(file.path(data_dir, name), fileEncoding = "UTF-8-BOM", na.strings = "", check.names = FALSE)
}
tag <- function(caption) {
  plot_annotation(tag_levels = "a", caption = caption,
                  theme = theme(plot.caption = element_text(margin = margin(t = 4))))
}
save_figure <- function(p, stem, w, h) {
  filename <- file.path(figure_dir, paste0(stem, ".", format))
  if (format == "png") {
    ggsave(filename, p, width = w, height = h, units = "mm", dpi = 400,
           bg = "white", device = ragg::agg_png)
  } else {
    ggsave(filename, p, width = w, height = h, units = "mm", bg = "white",
           device = grDevices::cairo_pdf, family = family)
  }
}

hc_references <- function(a) {
  list(
    geom_hline(yintercept = c(a$HC_p10[1], a$HC_p90[1]), colour = hc, alpha = .53, linewidth = .32, linetype = "dotted"),
    geom_hline(yintercept = a$HC_mean[1], colour = hc, alpha = .9, linewidth = .37, linetype = "longdash")
  )
}

# Figure 1 uses the observed Pearson-r outcome and the unadjusted fitted line.
d1 <- source_data("Figure_01_data.csv")
raw_ylim <- range(c(d1$raw_r, d1$HC_p10, d1$HC_p90), na.rm = TRUE) + c(-.045, .045)
raw_panel <- function(endpoint) {
  d <- subset(d1, kind == "point" & d1$scale == endpoint)
  fit <- subset(d1, kind == "line" & d1$scale == endpoint)
  a <- subset(d1, kind == "annotation" & d1$scale == endpoint)
  stopifnot(nrow(d) == 28L, nrow(a) == 1L)
  ggplot(d, aes(score, raw_r)) +
    hc_references(a) +
    geom_ribbon(data = fit, aes(x = score, ymin = line_ci_low, ymax = line_ci_high),
                inherit.aes = FALSE, fill = sz, alpha = .14) +
    geom_line(data = fit, aes(score, line_r), inherit.aes = FALSE, colour = deep_sz, linewidth = .57) +
    geom_point(shape = 21, fill = sz, colour = "#9B4536", stroke = .25, size = 2.05, alpha = .88) +
    coord_cartesian(ylim = raw_ylim) +
    labs(title = unname(pretty[endpoint]), x = "Raw score (points)", y = "Bin 10 similarity (Pearson r)",
         subtitle = sprintf("SZ n = %d   Raw r = %.2f, p = %.3f\nAdjusted beta = %.2f [%.2f, %.2f]",
                            a$n, a$pearson_r, a$pearson_p, a$beta, a$ci_low, a$ci_high))
}
p1 <- (raw_panel(scale_levels[1]) | raw_panel(scale_levels[2])) +
  tag(paste("Orange line and band show unadjusted OLS and its HC3 95% mean CI.",
            "Blue dashed line marks the HC mean (n = 32); blue dotted lines mark P10-P90.",
            "Adjusted beta uses Fisher-z similarity and controls age, sex and MeanFD.", sep = "\n"))
save_figure(p1, "Figure_01_Raw_TEPS_Similarity", 183, 91)

# Figure 2 shows raw scores and shared-reference sensitivity without fold-level inference.
d2 <- source_data("Figure_02_data.csv")
points2 <- subset(d2, kind == "point")
a2 <- subset(d2, kind == "group_gap_annotation")
p2a <- ggplot(points2, aes(display_x, raw_r)) + hc_references(points2) +
  geom_point(data = subset(points2, Group == "HC"), colour = hc, size = 1.9, alpha = .85) +
  geom_point(data = subset(points2, Group == "SZ" & !is.na(TEPS_consummatory)),
             aes(colour = TEPS_consummatory), size = 2.35, alpha = .94) +
  geom_point(data = subset(points2, Group == "SZ" & is.na(TEPS_consummatory)),
             colour = deep_sz, shape = 1, stroke = .48, size = 2.25) +
  scale_x_continuous(breaks = c(1,2), labels = c("HC (n = 32)", "SZ (n = 32)"), limits = c(.65,2.35)) +
  scale_colour_gradient(low = "#E8A070", high = "#A23729", name = "TEPS consummatory",
                        guide = guide_colourbar(display = "rectangles", barwidth = grid::unit(21, "mm"))) +
  labs(title = "Observed raw similarity", subtitle = "HC31 leave-one-out and SZ HC32 templates",
       x = NULL, y = "Bin 10 similarity (Pearson r)") +
  coord_cartesian(ylim = range(c(points2$raw_r, raw_ylim)) + c(-.035, .035))
p2b <- ggplot(subset(d2, kind == "shared_HC16_fold"), aes(SZ32_minus_HC)) +
  geom_histogram(binwidth = .016, boundary = 0, fill = "#A9BDD2", colour = "white", linewidth = .13) +
  geom_vline(xintercept = a2$frozen_group_gap, colour = hc, linetype = "longdash", linewidth = .47) +
  geom_vline(xintercept = 0, colour = "#777777", linewidth = .37) +
  scale_x_continuous(breaks = seq(-.4, 0, .1)) +
  coord_cartesian(xlim = c(-.405, .015)) +
  labs(title = "Both groups scored on the same template",
       subtitle = sprintf("HC16 splits (1,024 folds)\nOriginal gap %.3f; matched HC31 gap %.3f",
                          a2$frozen_group_gap, a2$matched_HC31_group_gap),
       x = "Mean similarity difference (SZ - HC)", y = "Held-out folds")
p2 <- (p2a | p2b) +
  tag(paste("a. Open SZ points indicate missing TEPS. Blue references show the HC mean and sample P10-P90.",
            "b. Each fold scores 16 held-out HC and 32 SZ using the same 16-HC template. The blue line marks the original gap.",
            "Folds reuse participants. The histogram describes sensitivity to template selection.", sep = "\n"))
save_figure(p2, "Figure_02_HC_Reference_and_SZ", 183, 103)

# Figure 3 preserves the exact historical residual axes and maps their signs to raw r.
d3 <- source_data("Figure_03_data.csv")
lim_x <- max(abs(d3$x_resid), na.rm = TRUE) + .40
lim_y <- max(abs(d3$y_resid), na.rm = TRUE) + .29
quad_panel <- function(endpoint) {
  d <- subset(d3, kind == "point" & d3$scale == endpoint)
  a <- subset(d3, kind == "annotation" & d3$scale == endpoint)
  count <- subset(d3, kind == "count" & d3$scale == endpoint)
  stopifnot(nrow(d) == 28L, nrow(a) == 1L, sum(count$n) == 28L)
  count$x <- ifelse(substr(count$quadrant,2,2) == "+", lim_x - .09, -lim_x + .09)
  count$y <- ifelse(substr(count$quadrant,5,5) == "+", lim_y - .14, -lim_y + .14)
  count$hjust <- ifelse(count$x > 0, 1, 0)
  ggplot(d, aes(x_resid, y_resid)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = faint, linewidth = .32) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = faint, linewidth = .32) +
    geom_abline(slope = a$beta, intercept = 0, colour = deep_sz, linewidth = .49) +
    geom_point(shape = 21, fill = sz, colour = deep_sz, stroke = .25, size = 1.95, alpha = .90) +
    geom_text(data = count, aes(x, y, label = paste("n =", n), hjust = hjust), inherit.aes = FALSE,
              size = 2.65, family = family, colour = gray) +
    coord_cartesian(xlim = c(-lim_x,lim_x), ylim = c(-lim_y,lim_y), expand = FALSE) +
    labs(title = unname(pretty[endpoint]), subtitle = sprintf("SZ n = %d   Adjusted beta = %.2f [%.2f, %.2f]",
                                                           a$n, a$beta, a$ci_low, a$ci_high),
         x = "Residual of z-scored TEPS", y = "Residual of z-scored similarity")
}
shape_panel <- function(endpoint) {
  d <- subset(d3, kind == "point" & d3$scale == endpoint)
  a <- subset(d3, kind == "annotation" & d3$scale == endpoint)
  d$sign <- factor(ifelse(d$x_resid > 0, "Positive", "Negative"), levels = c("Negative","Positive"))
  stopifnot(nrow(d) == 28L, nrow(a) == 1L)
  ggplot(d, aes(score, raw_r)) + hc_references(a) +
    geom_point(aes(shape = sign, fill = sign), colour = deep_sz, size = 2.25, stroke = .42) +
    scale_shape_manual(values = c(Negative = 21, Positive = 24), name = "TEPS residual") +
    scale_fill_manual(values = c(Negative = "white", Positive = sz), name = "TEPS residual") +
    coord_cartesian(ylim = raw_ylim) +
    labs(x = "Raw TEPS (points)", y = "Raw similarity (Pearson r)")
}
p3 <- ((quad_panel(scale_levels[1]) | quad_panel(scale_levels[2])) /
         (shape_panel(scale_levels[1]) | shape_panel(scale_levels[2]))) +
  plot_layout(guides = "collect") +
  tag(paste("a-b. Coordinates are residuals of complete-case z scores after age, sex and MeanFD adjustment.",
            "Text gives full-model HC3 95% CIs. c-d. The same people appear in raw units; shape encodes the TEPS residual sign.",
            "Blue dashed and dotted lines mark the HC mean and sample P10-P90 (n = 32).", sep = "\n"))
p3 <- p3 & theme(legend.position = "bottom")
save_figure(p3, "Figure_03_Residual_Quadrants", 183, 151)

# Figure 4 shows one leave-out estimate per subject and percentile intervals from resampling.
d4 <- source_data("Figure_04_data.csv")
boot_xlim <- range(d4$beta[d4$kind == "bootstrap"], na.rm = TRUE) + c(-.03,.03)
loo_panel <- function(endpoint) {
  d <- subset(d4, kind == "leave_one_out" & d4$scale == endpoint)
  stopifnot(nrow(d) == 28L)
  a <- d[which.max(abs(d$delta)), ]
  ggplot(d, aes(effect_order, beta)) +
    geom_hline(yintercept = d$full_beta[1], linewidth = .42, linetype = "longdash", colour = gray) +
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = .18, linewidth = .36, colour = "#C99285") +
    geom_point(colour = deep_sz, size = 1.32) +
    geom_point(data = a, colour = deep_sz, size = 2.25, shape = 1, stroke = .52) +
    scale_x_continuous(breaks = c(1,7,14,21,28)) +
    coord_cartesian(ylim = c(-.84,.015)) +
    labs(title = unname(pretty[endpoint]),
         subtitle = sprintf("Largest change after omitting %s", a$omitted_ID),
         x = "Omissions ordered by estimate (28 refits)", y = "Adjusted beta")
}
boot_panel <- function(endpoint) {
  d <- subset(d4, kind == "bootstrap" & d4$scale == endpoint)
  a <- subset(d4, kind == "summary" & d4$scale == endpoint)
  stopifnot(nrow(d) == 20000L, nrow(a) == 1L)
  ggplot(d, aes(beta)) +
    geom_histogram(binwidth = .025, boundary = 0, fill = "#DE9C8D", colour = "white", linewidth = .16) +
    geom_vline(xintercept = a$beta, linetype = "longdash", colour = gray, linewidth = .44) +
    geom_vline(xintercept = c(a$ci_low, a$ci_high), linetype = "dotted", colour = deep_sz, linewidth = .51) +
    coord_cartesian(xlim = boot_xlim) +
    labs(subtitle = sprintf("Participant bootstrap 95%% CI [%.2f, %.2f]", a$ci_low, a$ci_high),
         x = "Resampled adjusted beta", y = "Resamples")
}
p4 <- ((loo_panel(scale_levels[1]) | loo_panel(scale_levels[2])) /
         (boot_panel(scale_levels[1]) | boot_panel(scale_levels[2]))) +
  tag(paste("a-b. Each refit uses 27 SZ participants. Bars show HC3 Wald 95% CIs and the dashed line shows the full-sample beta.",
            "c-d. 20,000 draws from the 28 participants. Dotted lines mark percentile limits.", sep = "\n"))
save_figure(p4, "Figure_04_Leave_One_Out_and_Bootstrap", 183, 146)

# Figure 5 contrasts comparable standardized coefficients across models.
d5 <- source_data("Figure_05_data.csv")
methods <- c("Unadjusted OLS on Fisher z", "Adjusted OLS HC3 (historical Wald)", "Adjusted OLS HC3 (t)",
             "Adjusted OLS participant bootstrap", "Huber M regression (H2)", "Exclude all influence flags",
             "Matched HC31 template sensitivity")
short_methods <- c("Unadjusted OLS", "Adjusted HC3 Wald", "Adjusted HC3 t", "Participant bootstrap",
                   "Huber regression", "Exclude flagged cases", "Matched HC31 templates")
d5$method_label <- factor(d5$method, levels = rev(methods), labels = rev(short_methods))
d5$scale_label <- factor(pretty[d5$scale], levels = pretty[scale_levels])
d5$label_n <- sprintf("n = %d", d5$n)
p5 <- ggplot(d5, aes(beta, method_label)) +
  geom_vline(xintercept = 0, colour = faint, linetype = "dotted", linewidth = .40) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = .16, colour = gray, linewidth = .38) +
  geom_point(shape = 21, fill = sz, colour = deep_sz, stroke = .3, size = 2.15) +
  geom_text(aes(x = .14, label = label_n), family = family, size = 2.50, hjust = 1, colour = gray) +
  facet_wrap(~scale_label, nrow = 1) +
  coord_cartesian(xlim = c(-.84,.16)) +
  labs(x = "Standardized TEPS beta (95% CI)", y = NULL,
       caption = paste("Similarity is in Fisher-z units before standardization. Adjusted models include age, sex and MeanFD.",
                       "The exclusion rule uses Cook's distance, leverage, studentized residuals and DFBETAS.", sep = "\n"))
save_figure(p5, "Figure_05_Model_Sensitivity", 183, 98)

# Figure 6 joins the raw G1 trajectories, binwise estimates and the 43 target patterns.
d6 <- source_data("Figure_06_data.csv")
p6a <- ggplot(subset(d6, kind == "profile"), aes(bin, current_mean_FC)) +
  geom_line(aes(group = Subject_ID, colour = TEPS_consummatory), alpha = .44, linewidth = .27) +
  geom_line(data = subset(d6, kind == "profile_mean" & population == "HC32"), aes(y = mean),
            colour = hc, linewidth = .88) +
  geom_line(data = subset(d6, kind == "profile_mean" & population == "SZ28"), aes(y = mean),
            colour = deep_sz, linewidth = .86, linetype = "longdash") +
  scale_colour_gradient(low = "#EAB181", high = deep_sz, name = "TEPS consummatory",
                        guide = guide_colourbar(display = "rectangles", barwidth = grid::unit(18, "mm"))) +
  scale_x_continuous(breaks = c(1,3,5,7,9,10)) +
  labs(title = "Anticipation mean FC along G1", subtitle = "SZ trajectories (n = 28); HC mean (n = 32)",
       x = "Fixed G1 bin", y = "Mean FC (Fisher z)") +
  theme(legend.margin = margin(-2,0,-3,0))
effects6 <- subset(d6, kind == "bin_effect")
effects6$category <- factor(effects6$category, levels = c("Profile similarity","Mean FC"))
p6b <- ggplot(effects6, aes(bin, beta, colour = category, shape = category)) +
  geom_hline(yintercept = 0, colour = faint, linetype = "dotted", linewidth = .36) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = .14, position = position_dodge(width = .54),
                linewidth = .36) +
  geom_point(position = position_dodge(width = .54), size = 1.73) +
  scale_colour_manual(values = c("Profile similarity" = deep_sz, "Mean FC" = gray), name = NULL) +
  scale_shape_manual(values = c("Profile similarity" = 16, "Mean FC" = 1), name = NULL) +
  scale_x_continuous(breaks = c(1,3,5,7,9,10)) +
  labs(title = "TEPS effect by G1 bin", subtitle = "Age-, sex- and MeanFD-adjusted; SZ n = 28",
       x = "Fixed G1 bin", y = "TEPS beta (HC3 95% CI)") +
  theme(legend.margin = margin(-2,0,-3,0))
tail6 <- subset(d6, kind == "tail_pattern")
hc6 <- subset(tail6, group == "Lower TEPS quartile")
tail6$label <- ifelse(tail6$group == "Lower TEPS quartile", "Lower TEPS quartile", "Upper TEPS quartile")
hc6$label <- "HC32 template"
hc6$mean_centered_unit_FC <- hc6$HC32_centered_unit_FC
tail6 <- rbind(tail6, hc6)
tail6$label <- factor(tail6$label, levels = c("HC32 template","Lower TEPS quartile","Upper TEPS quartile"))
tail6$x <- tail6$HC_mean_order + c("HC32 template" = 0, "Lower TEPS quartile" = -.18,
                                  "Upper TEPS quartile" = .18)[as.character(tail6$label)]
p6c <- ggplot(tail6, aes(x, mean_centered_unit_FC, colour = label, shape = label)) +
  geom_hline(yintercept = 0, colour = faint, linetype = "dotted", linewidth = .33) +
  geom_point(size = 1.73, stroke = .50, alpha = .95) +
  scale_x_continuous(breaks = c(1,5,10,15,20,25,30,35,40,43)) +
  scale_colour_manual(values = c("HC32 template" = hc, "Lower TEPS quartile" = "#CE8951",
                                  "Upper TEPS quartile" = deep_sz), name = NULL) +
  scale_shape_manual(values = c("HC32 template" = 15, "Lower TEPS quartile" = 1,
                                 "Upper TEPS quartile" = 17), name = NULL) +
  labs(title = "Bin 10 target pattern", subtitle = "Mean centered, unit-norm patterns; 7 SZ participants in each TEPS tail",
       x = "Anticipation target rank by HC mean FC (43 targets)", y = "Centered FC (unit norm)") +
  theme(legend.margin = margin(-2,0,-3,0))
p6 <- ((p6a | p6b) / p6c) +
  plot_layout(heights = c(1,1)) +
  tag(paste("All panels use G1_bin_current (285 voxels in Bin 10). a. Blue and dashed orange are HC and SZ means.",
            "b. Binwise estimates are exploratory apart from the original Bin 10 similarity endpoint.",
            "c. Points are descriptive quartile means for each of the 43 frozen targets.", sep = "\n"))
save_figure(p6, "Figure_06_Anticipation_FC_Profiles", 183, 159)

# Figure 7 gives both the stage matrix and the direct paired TEPS contrasts.
d7 <- source_data("03_CROSS_STAGE_ASSOCIATION_FIGURE_data.csv")
cells <- subset(d7, record_type == "cell")
stage_order <- c("TEPS_anticipatory","TEPS_consummatory","TEPS_total",
                 "ACIPS_anticipatory","ACIPS_consummatory","ACIPS_total")
labels <- sprintf("%s (n = %d)", pretty[stage_order], cells$n[match(stage_order, cells$scale)])
cells$label <- factor(cells$scale, levels = rev(stage_order), labels = rev(labels))
cells$stage <- factor(cells$outcome, levels = c("Anticipation_similarity","Outcome_similarity"),
                      labels = c("Anticipation similarity","Outcome similarity"))
cells$evidence <- factor(ifelse(cells$historical_SZ_q_BH119 < .05, "Historical BH q < 0.05", "Historical BH q >= 0.05"),
                         levels = c("Historical BH q < 0.05","Historical BH q >= 0.05"))
p7a <- ggplot(cells, aes(beta, label)) +
  geom_vline(xintercept = 0, colour = faint, linetype = "dotted", linewidth = .40) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = .13, colour = gray, linewidth = .41) +
  geom_point(aes(fill = evidence), shape = 21, colour = deep_sz, stroke = .52, size = 2.29) +
  scale_fill_manual(values = c("Historical BH q < 0.05" = sz, "Historical BH q >= 0.05" = "white"), name = NULL) +
  facet_wrap(~stage, nrow = 1) +
  coord_cartesian(xlim = c(-.89,.14)) +
  labs(x = "Adjusted beta (HC3 95% CI)", y = NULL, title = "Scale associations within SZ")
diff <- subset(d7, record_type == "paired_contrast")
contrast_labels <- c("Consummatory - anticipatory score\non Anticipation similarity",
                     "Anticipation - Outcome | TEPS anticipatory",
                     "Anticipation - Outcome | TEPS consummatory",
                     "Anticipation - Outcome | TEPS total")
diff$label_short <- factor(diff$label, levels = rev(diff$label), labels = rev(contrast_labels))
p7b <- ggplot(diff, aes(beta,label_short)) +
  geom_vline(xintercept = 0, colour = faint, linetype = "dotted", linewidth = .40) +
  geom_errorbar(aes(xmin = ci_low, xmax = ci_high), orientation = "y", width = .14, colour = gray, linewidth = .41) +
  geom_point(shape = 21, fill = sz, colour = deep_sz, stroke = .36, size = 2.2) +
  coord_cartesian(xlim = c(-.53,.47)) +
  labs(title = "Direct TEPS contrasts", subtitle = "Same 28 participants for both estimates in every comparison",
       x = "Difference in adjusted betas (paired bootstrap 95% CI)", y = NULL)
p7 <- (p7a / p7b) +
  plot_layout(heights = c(1.18,1)) +
  tag(paste("Models adjust age, sex and MeanFD. a. Filling refers to the original 119-test SZ BH family.",
            "b. Paired participant bootstraps preserve the dependence between the two effects.", sep = "\n"))
save_figure(p7, "03_CROSS_STAGE_ASSOCIATION_FIGURE", 183, 149)
