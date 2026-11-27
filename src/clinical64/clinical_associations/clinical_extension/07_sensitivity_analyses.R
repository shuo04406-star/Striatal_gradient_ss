source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1]), winslash="/")), "_common.R"))

dat <- read_tsv(file.path(INTERNAL_DIR, "clinical_imaging_analysis_dataset.tsv"))
num <- c("PANSS_negative","age_scan","mean_FD_rest_posttrim","bin10_anticipation_profile_z","bin10_outcome_profile_z",
         "SMA_anticipation_rawFC_slope","IFG_outcome_rawFC_slope","Cingulate_outcome_rawFC_slope")
for (v in num) dat[[v]] <- as_num(dat[[v]])
dat$sex_scan <- factor(dat$sex_scan, levels=c("Male","Female"))
dat$group_binary <- as.integer(dat$group == "SZ")
sz <- dat[dat$group == "SZ", ]

primary_pheno <- c("bin10_anticipation_profile_z","bin10_outcome_profile_z","SMA_anticipation_rawFC_slope")
pheno_labels <- c(bin10_anticipation_profile_z="Bin 10 anticipation similarity (Fisher z)",
                  bin10_outcome_profile_z="Bin 10 outcome similarity (Fisher z)",
                  SMA_anticipation_rawFC_slope="SMA anticipation raw-FC slope",
                  IFG_outcome_rawFC_slope="IFG outcome raw-FC slope",
                  Cingulate_outcome_rawFC_slope="Cingulate outcome raw-FC slope")
rows <- list(); diagnostics <- list()

# Covariate-adjusted partial rank regressions for the three primary clinical associations.
for (i in seq_along(primary_pheno)) {
  p <- primary_pheno[i]
  d <- sz[complete.cases(sz[c("PANSS_negative",p,"age_scan","sex_scan","mean_FD_rest_posttrim")]), ]
  mdat <- data.frame(y=rank_z(d[[p]]), x=rank_z(d$PANSS_negative), age=zscore(d$age_scan),
                     sex=d$sex_scan, fd=zscore(d$mean_FD_rest_posttrim))
  fit <- lm(y ~ x + age + sex + fd, data=mdat)
  h <- hc3_term(fit, "x")
  dg <- model_diagnostics(fit, "x")
  id <- paste0("R_ADJ_PRIMARY_", i)
  rows[[length(rows)+1L]] <- data.frame(
    analysis_id=id, analysis_tier="robustness", sensitivity_family="primary_covariate_adjustment",
    clinical_variable="PANSS_negative", phenotype=p, phenotype_label=pheno_labels[[p]], effective_n=nrow(d),
    model_type="Rank-based OLS with HC3 robust SE", effect_type="adjusted_standardized_rank_beta",
    effect=h$estimate, ci_low=h$ci_low, ci_high=h$ci_high, standard_error=h$se, p_raw=h$p, p_adjusted=NA_real_,
    correction_method="sensitivity_not_in_primary_Holm_family", covariates="age, sex, resting mean FD",
    direction_stable_vs_primary=NA, missing_data_rule="Complete observed cases for model; no imputation",
    notes="Sensitivity model; not a replacement for the prespecified Spearman test", stringsAsFactors=FALSE)
  diagnostics[[id]] <- cbind(analysis_id=id, dg)
}

# Pearson correlations are retained to disclose any Spearman/Pearson discrepancy.
primary <- read.csv(file.path(ANALYSIS_ROOT,"primary_clinical_associations.csv"), stringsAsFactors=FALSE, fileEncoding="UTF-8")
for (i in seq_along(primary_pheno)) {
  p <- primary_pheno[i]
  d <- sz[complete.cases(sz[c("PANSS_negative",p)]), ]
  ct <- cor.test(d$PANSS_negative, d[[p]], method="pearson")
  spr <- primary$effect[primary$phenotype == p]
  rows[[length(rows)+1L]] <- data.frame(
    analysis_id=paste0("R_PEARSON_PRIMARY_",i), analysis_tier="robustness", sensitivity_family="Pearson_vs_Spearman",
    clinical_variable="PANSS_negative", phenotype=p, phenotype_label=pheno_labels[[p]], effective_n=nrow(d),
    model_type="Pearson correlation", effect_type="Pearson_r", effect=unname(ct$estimate), ci_low=ct$conf.int[1], ci_high=ct$conf.int[2],
    standard_error=NA_real_, p_raw=ct$p.value, p_adjusted=NA_real_, correction_method="descriptive_sensitivity",
    covariates="none", direction_stable_vs_primary=sign(unname(ct$estimate)) == sign(spr),
    missing_data_rule="Pairwise observed SZ cases; no imputation",
    notes=if(abs(unname(ct$estimate)-spr) > .2 || sign(unname(ct$estimate)) != sign(spr)) "Material Pearson/Spearman discrepancy; inspect distribution/nonlinearity" else "No material method discrepancy by prespecified flag", stringsAsFactors=FALSE)
}

# Relationship between resting-state motion and each fixed phenotype within SZ.
motion_pheno <- c(primary_pheno,"IFG_outcome_rawFC_slope","Cingulate_outcome_rawFC_slope")
motion_start <- length(rows)
for (i in seq_along(motion_pheno)) {
  p <- motion_pheno[i]
  z <- spearman_boot(sz, "mean_FD_rest_posttrim", p, 20261500L+i, B=10000L)
  rows[[length(rows)+1L]] <- data.frame(
    analysis_id=paste0("R_MOTION_ASSOC_",i), analysis_tier="robustness", sensitivity_family="SZ_motion_phenotype_5",
    clinical_variable="mean_FD_rest_posttrim", phenotype=p, phenotype_label=pheno_labels[[p]], effective_n=z$n,
    model_type="Spearman rank correlation", effect_type="Spearman_rho", effect=z$rho, ci_low=z$ci_low, ci_high=z$ci_high,
    standard_error=NA_real_, p_raw=z$p, p_adjusted=NA_real_, correction_method="Holm across five motion-phenotype checks",
    covariates="none", direction_stable_vs_primary=NA, missing_data_rule="Observed frozen SZ32; no imputation",
    notes="Resting-state post-trim MeanFD, not reward-task motion block FD", stringsAsFactors=FALSE)
}

# Re-estimate phenotype-level group differences with and without age/sex/MeanFD.
for (i in seq_along(primary_pheno)) {
  p <- primary_pheno[i]
  d <- dat[complete.cases(dat[c(p,"group_binary","age_scan","sex_scan","mean_FD_rest_posttrim")]), ]
  mdat <- data.frame(y=zscore(d[[p]]), group=d$group_binary, age=zscore(d$age_scan), sex=d$sex_scan, fd=zscore(d$mean_FD_rest_posttrim))
  for (adj in c(FALSE,TRUE)) {
    fit <- if(adj) lm(y ~ group + age + sex + fd, data=mdat) else lm(y ~ group, data=mdat)
    h <- hc3_term(fit,"group")
    id <- paste0("R_GROUP_",i,"_",if(adj) "ADJ" else "UNADJ")
    rows[[length(rows)+1L]] <- data.frame(
      analysis_id=id, analysis_tier="robustness", sensitivity_family="phenotype_group_effect_motion_adjustment",
      clinical_variable="SZ_vs_HC", phenotype=p, phenotype_label=pheno_labels[[p]], effective_n=nrow(d),
      model_type="OLS with HC3 robust SE", effect_type="standardized_group_difference_SZ_minus_HC", effect=h$estimate,
      ci_low=h$ci_low, ci_high=h$ci_high, standard_error=h$se, p_raw=h$p, p_adjusted=NA_real_, correction_method="descriptive_sensitivity",
      covariates=if(adj) "age, sex, resting mean FD" else "none", direction_stable_vs_primary=NA,
      missing_data_rule="Frozen 64; complete covariates and phenotype", notes="Phenotype-level sensitivity; does not replace the original mixed/repeated-data main model", stringsAsFactors=FALSE)
    diagnostics[[id]] <- cbind(analysis_id=id, model_diagnostics(fit,"group"))
  }
}

# Explicitly record unavailable high-motion and CPZ sensitivities.
for (i in seq_along(primary_pheno)) {
  p <- primary_pheno[i]
  rows[[length(rows)+1L]] <- data.frame(
    analysis_id=paste0("R_HIGHMOTION_NOT_RUN_",i), analysis_tier="robustness_not_run", sensitivity_family="high_motion_exclusion",
    clinical_variable="project_QC_high_motion_flag", phenotype=p, phenotype_label=pheno_labels[[p]], effective_n=0L,
    model_type="not_run", effect_type="not_estimable", effect=NA_real_, ci_low=NA_real_, ci_high=NA_real_, standard_error=NA_real_,
    p_raw=NA_real_, p_adjusted=NA_real_, correction_method="not_applicable", covariates="not_applicable", direction_stable_vs_primary=NA,
    missing_data_rule="No ad hoc threshold introduced", notes="No traceable project-defined subject exclusion threshold was located for the frozen resting-state cohort.", stringsAsFactors=FALSE)
  rows[[length(rows)+1L]] <- data.frame(
    analysis_id=paste0("R_CPZ_NOT_RUN_",i), analysis_tier="robustness_not_run", sensitivity_family="CPZ_adjustment",
    clinical_variable="CPZ_equivalent_mg_day", phenotype=p, phenotype_label=pheno_labels[[p]], effective_n=0L,
    model_type="not_run", effect_type="not_estimable", effect=NA_real_, ci_low=NA_real_, ci_high=NA_real_, standard_error=NA_real_,
    p_raw=NA_real_, p_adjusted=NA_real_, correction_method="not_applicable", covariates="not_applicable", direction_stable_vs_primary=NA,
    missing_data_rule="No medication dose imputation or zero-filling", notes="CPZ conversion blocked by medication audit; adjustment was not performed.", stringsAsFactors=FALSE)
}

res <- do.call(rbind, rows)
motion_idx <- which(res$sensitivity_family == "SZ_motion_phenotype_5")
res$p_adjusted[motion_idx] <- safe_holm(res$p_raw[motion_idx])
for (p in primary_pheno) {
  prim_eff <- primary$effect[primary$phenotype == p]
  idx <- res$sensitivity_family == "primary_covariate_adjustment" & res$phenotype == p
  res$direction_stable_vs_primary[idx] <- sign(res$effect[idx]) == sign(prim_eff)
}
write_csv_utf8(res, file.path(ANALYSIS_ROOT,"robustness_sensitivity_results.csv"))
write_tsv_utf8(do.call(rbind, diagnostics), file.path(INTERNAL_DIR,"regression_model_diagnostics.tsv"))

# Standardized demographic differences for the actual frozen 32+32 cohort.
hedges_g <- function(x1,x0) {
  n1<-length(x1);n0<-length(x0);sp<-sqrt(((n1-1)*var(x1)+(n0-1)*var(x0))/(n1+n0-2));d<-(mean(x1)-mean(x0))/sp; d*(1-3/(4*(n1+n0)-9))
}
boot_demo <- function(type, seed, B=10000L) {
  szd<-dat[dat$group=="SZ",]; hcd<-dat[dat$group=="HC",]; set.seed(seed)
  if(type=="age") f<-function(a,b) hedges_g(a$age_scan,b$age_scan)
  if(type=="fd") f<-function(a,b) hedges_g(a$mean_FD_rest_posttrim,b$mean_FD_rest_posttrim)
  if(type=="sex") f<-function(a,b) mean(a$sex_scan=="Female")-mean(b$sex_scan=="Female")
  est<-f(szd,hcd); bs<-replicate(B,f(szd[sample.int(nrow(szd),nrow(szd),TRUE),],hcd[sample.int(nrow(hcd),nrow(hcd),TRUE),]));
  c(est=est,quantile(bs,c(.025,.975),type=6,na.rm=TRUE))
}
bd <- rbind(age=boot_demo("age",20261601L), resting_meanFD=boot_demo("fd",20261602L), female_proportion=boot_demo("sex",20261603L))
demo_effects <- data.frame(variable=rownames(bd), effect_type=c("Hedges_g_SZ_minus_HC","Hedges_g_SZ_minus_HC","risk_difference_SZ_minus_HC"),
                           effect=bd[,1],ci_low=bd[,2],ci_high=bd[,3],n_SZ=32L,n_HC=32L,
                           note="Actual frozen imaging cohort; 10,000-resample group-stratified bootstrap CI", stringsAsFactors=FALSE)
write_tsv_utf8(demo_effects,file.path(INTERNAL_DIR,"actual_imaging64_demographic_effect_sizes.tsv"))

append_provenance(data.frame(
  output_or_result=c("robustness_sensitivity_results.csv","regression_model_diagnostics.tsv","actual_imaging64_demographic_effect_sizes.tsv"),
  analysis_stage="C_robustness", input_files="clinical_imaging_analysis_dataset.tsv; primary_clinical_associations.csv",
  script="scripts/07_sensitivity_analyses.R",
  variable_source=c("Primary associations, covariates, motion and fixed phenotypes","All fitted OLS sensitivity models","Frozen current64 covariates"),
  sample_rule=c("SZ observed cases; group models on frozen 64","Model-specific complete cases","32 SZ and 32 HC"),
  method=c("Rank-based adjusted HC3, Pearson, motion correlations, group-effect comparison","Cook's distance, leverage, VIF, condition number, Shapiro","Hedges g/risk difference with stratified bootstrap CI"),
  version="clinical_extension_v1_20261029", stringsAsFactors=FALSE
))

session_record("07_sensitivity_analyses")
message("Covariate, motion, model-diagnostic, and demographic sensitivities completed.")
