"""Phases 3 and 4: dependent stage contrasts, scale overlap and clinical adjustments."""
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from followup_common import (
    ACIPS_A, ACIPS_C, ACIPS_T, ANT, BOOT_N, CLINICAL, CORE, COVARIATES, ID, OUTCOME, SEED,
    STAGE_SCALES, TEPS_A, TEPS_C, TEPS_T, bh, bootstrap_betas, cases, close, covariate_frame,
    effect, fit, model, percentile, residual, write_csv, z,
)

STAGES = (ANT, OUTCOME)


def stage_matrix(src):
    hist = src.read("within").set_index(["population", "scale_variable", "imaging_variable"])
    combined = src.read("eligible").set_index(["scale_variable", "imaging_variable"])
    rows = []
    for scale in STAGE_SCALES:
        for image in STAGES:
            d = cases(src.sz, scale, image)
            m = model(d, scale, image)
            normal = effect(m)
            t = effect(m, use_t=True)
            prev = hist.loc[("SZ", scale, image)]
            hc = hist.loc[("HC", scale, image)]
            pool = combined.loc[(scale, image)]
            assert prev.n==len(d) and close(list(normal.values())[1:],
                                            prev[["effect","se","ci_low","ci_high","p"]].to_numpy(float))
            rows.append(dict(record_type="cell", scale=scale, outcome=image, population="SZ", **t,
                             residual_df=m.df_resid, sample_ids=";".join(d.index),
                             historical_SZ_beta=prev.effect, historical_SZ_p=prev.p,
                             historical_SZ_ci_low=prev.ci_low, historical_SZ_ci_high=prev.ci_high,
                             historical_SZ_q_BH119=prev.p_adj_bh_within_population,
                             historical_HC_n=hc.n, historical_HC_beta=hc.effect, historical_HC_p=hc.p,
                             historical_HC_ci_low=hc.ci_low, historical_HC_ci_high=hc.ci_high,
                             historical_HC_q_BH70=hc.p_adj_bh_within_population,
                             historical_pooled_n=pool.n, historical_pooled_beta=pool.effect,
                             historical_pooled_p=pool.p, historical_pooled_q_BH119=pool.p_adj_bh_global,
                             historical_pooled_CI_low=pool.ci_low, historical_pooled_CI_high=pool.ci_high,
                             family="exploratory_BH12_SZ_stage_matrix",
                             method="Standardized OLS adjusted Age, Sex, MeanFD; HC3 t reference"))
    cells = pd.DataFrame(rows)
    cells["exploratory_q"] = bh(cells.p)
    return cells


def paired_contrast(src, pair_a, pair_b, label, family, seed):
    x1, y1 = pair_a
    x2, y2 = pair_b
    xs, ys = tuple(dict.fromkeys([x1,x2])), tuple(dict.fromkeys([y1,y2]))
    d = src.sz.dropna(subset=[*xs, *ys, *COVARIATES])
    a, b = effect(model(d, x1, y1))["beta"], effect(model(d, x2, y2))["beta"]
    difference = a-b
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(d), size=(BOOT_N, len(d)))
    fitted = bootstrap_betas(d, xs, ys, sample)
    deltas = fitted[:, xs.index(x1), ys.index(y1)] - fitted[:, xs.index(x2), ys.index(y2)]
    deltas = deltas[np.isfinite(deltas)]
    assert len(deltas) > .99*BOOT_N
    low, high = percentile(deltas)
    p = (1 + (np.abs(deltas-difference)>=abs(difference)).sum())/(len(deltas)+1)
    design = covariate_frame(d)
    c1, c2 = (x1,x2) if x1 != x2 else (y1,y2)
    shared_r = np.corrcoef(residual(d[c1], design), residual(d[c2], design))[0,1]
    hc3 = {}
    if x1==x2:
        design["TEPS_or_ACIPS"] = z(d[x1])
        e = effect(fit(z(d[y1])-z(d[y2]), design), use_t=True)
        assert close(e["beta"], difference)
        hc3 = dict(same_predictor_HC3_t_p=e["p"], same_predictor_HC3_t_ci_low=e["ci_low"],
                   same_predictor_HC3_t_ci_high=e["ci_high"])
    return dict(record_type="paired_contrast", label=label, scale=x1, outcome=y1, comparator_scale=x2,
                comparator_outcome=y2, population="SZ", n=len(d), beta=difference,
                beta_a=a, beta_b=b, ci_low=low, ci_high=high, p=p, n_resamples=BOOT_N, valid_boot=len(deltas),
                shared_pair_partial_r=shared_r, probability_delta_negative=(deltas<0).mean(),
                sample_ids=";".join(d.index), family=family, **hc3,
                method="Paired participant percentile bootstrap; approximate two-sided centered bootstrap p")


def stage_contrasts(src):
    specs = [((TEPS_C, ANT), (TEPS_A, ANT), "Consummatory minus anticipatory | Anticipation", "exploratory_BH4_TEPS_direct")]
    for scale in (TEPS_A, TEPS_C, TEPS_T):
        label = scale.split("_")[1].capitalize()
        specs.append(((scale, ANT), (scale, OUTCOME), f"Anticipation minus Outcome | {label}", "exploratory_BH4_TEPS_direct"))
    for x1, x2 in [(TEPS_A,ACIPS_A),(TEPS_C,ACIPS_C),(TEPS_T,ACIPS_T)]:
        for y in STAGES:
            specs.append(((x1,y),(x2,y),f"TEPS minus ACIPS | {x1.split('_')[1]} | {y.split('_')[0]}",
                          "exploratory_BH6_cross_instrument"))
    rows = [paired_contrast(src, a, b, label, family, SEED+300+i)
            for i,(a,b,label,family) in enumerate(specs)]
    result = pd.DataFrame(rows)
    result["exploratory_q"] = result.groupby("family")["p"].transform(bh)
    return result


def scale_overlap(src, score_summary):
    rows = []
    pairs = [
        (TEPS_T,TEPS_C), (TEPS_T,TEPS_A), (TEPS_A,TEPS_C),
        (ACIPS_T,ACIPS_C), (ACIPS_T,ACIPS_A), (ACIPS_A,ACIPS_C),
        (TEPS_A,ACIPS_A), (TEPS_C,ACIPS_C), (TEPS_T,ACIPS_T),
    ]
    for group in ("SZ", "HC"):
        for a, b in pairs:
            d = src.data[src.data.Group.eq(group)].dropna(subset=[a,b,*COVARIATES])
            c = covariate_frame(d)
            r, p = stats.pearsonr(d[a], d[b])
            rho, p_s = stats.spearmanr(d[a], d[b])
            rows.append(dict(record_type="pairwise_overlap", population=group, scale=a, scale_b=b, n=len(d),
                             pearson_r=r, pearson_r_squared=r**2, pearson_p_descriptive=p,
                             spearman_rho=rho, spearman_p_descriptive=p_s,
                             partial_pearson_r=np.corrcoef(residual(d[a],c),residual(d[b],c))[0,1],
                             sample_ids=";".join(d.index)))
    d = src.sz.dropna(subset=[TEPS_A,TEPS_C,TEPS_T,ANT,*COVARIATES])
    design = covariate_frame(d)
    design[TEPS_A], design[TEPS_C] = z(d[TEPS_A]), z(d[TEPS_C])
    vifs = {term: variance_inflation_factor(design.to_numpy(), i) for i,term in enumerate(design.columns) if i}
    y = z(d[ANT])
    assert len(d)==28 and design.shape[1]==6
    for term, value in vifs.items():
        rows.append(dict(record_type="joint_model_VIF", scale=term, population="SZ", n=len(d),
                         vif=value, model_parameters=design.shape[1]))
    if max(vifs.values()) < 5:
        m = fit(y, design)
        for scale in (TEPS_A, TEPS_C):
            rows.append(dict(record_type="joint_TEPS_subscales_HC3_t", scale=scale, population="SZ",
                             **effect(m, scale, use_t=True), vif=vifs[scale],
                             model_parameters=design.shape[1], residual_df=m.df_resid))
    items = src.read("items")
    q13 = items[items.Questionnaire.eq("TEPS") & items.Item_ID.eq("TEPS13")].set_index(ID).loc[d.index,"Scored_Value"]
    assert close(d[TEPS_T], d[TEPS_A]+d[TEPS_C]+q13)
    c = covariate_frame(d)
    y_resid, total_resid = residual(y,c), residual(d[TEPS_T],c)
    denominator = total_resid @ total_resid
    components = {key: residual(val,c) @ y_resid / denominator * d[TEPS_T].std()
                  for key,val in [(TEPS_A,d[TEPS_A]),(TEPS_C,d[TEPS_C]),("TEPS_item13_reversed",q13)]}
    total_beta = effect(model(d, TEPS_T, ANT))["beta"]
    assert close(sum(components.values()), total_beta)
    for scale, component in components.items():
        rows.append(dict(record_type="arithmetic_total_beta_component", scale=scale, population="SZ", n=len(d),
                         beta_component=component, signed_share_of_total_beta=component/total_beta,
                         total_beta=total_beta))
    return pd.concat([score_summary, pd.DataFrame(rows)], ignore_index=True)


def run_phase3(src, out, score_summary):
    cells, contrasts = stage_matrix(src), stage_contrasts(src)
    write_csv(pd.concat([cells,contrasts],ignore_index=True),out/"03_STAGE_SPECIFICITY_RESULTS.csv")
    write_csv(scale_overlap(src,score_summary),out/"03_TEPS_ACIPS_SCALE_INFORMATION.csv")
    teps = contrasts.family.eq("exploratory_BH4_TEPS_direct")
    write_csv(pd.concat([cells, contrasts.loc[teps]],ignore_index=True),
              out/"figures/source_data/03_CROSS_STAGE_ASSOCIATION_FIGURE_data.csv")


def run_phase4(src, out):
    rows = []
    specs = [
        ("PANSS_negative", "negative symptom burden"),
        ("PANSS_total", "general symptom burden; fitted separately from PANSS negative"),
        ("SANS", "alternative negative-symptom assessment"),
        ("SAS", "parkinsonism/motor burden"),
        ("BARS", "akathisia burden"),
        ("illness_duration", "recorded disease duration in years"),
        ("DUP", "duration of untreated psychosis"),
        ("age_at_onset", "document coverage only; age and duration already addressed"),
        ("PANSS_positive", "document coverage only; PANSS total used for overall burden"),
        ("PANSS_general", "document coverage only; PANSS total used for overall burden"),
        ("medication_text", "unharmonized free text; no verified scan-time dose or CPZ-equivalent field"),
    ]
    for scale in CORE:
        full = cases(src.sz,scale,ANT)
        m = model(full,scale,ANT)
        for term in ("TEPS_or_ACIPS","Age","MeanFD","Sex_Male_vs_Female"):
            rows.append(dict(scale=scale, record_type="baseline_covariate", clinical_variable=term,
                             status="executed", **effect(m,term,use_t=True), residual_df=m.df_resid,
                             unit="similarity SD per covariate SD" if not term.startswith("Sex") else
                             "similarity SD, male minus female",
                             sample_ids=";".join(full.index)))
        for covariate, rationale in specs:
            values = full[covariate]
            d = full.loc[values.notna()]
            n_nonzero = values.dropna().ne(0).sum() if covariate!="medication_text" else np.nan
            row = dict(scale=scale, record_type="one_at_a_time_adjustment", clinical_variable=covariate,
                       n=len(d), n_valid_SZ32=src.sz[covariate].notna().sum(), n_unavailable_in_core=len(full)-len(d),
                       n_distinct=values.nunique(), n_nonzero=n_nonzero, rationale=rationale, sample_ids=";".join(d.index))
            if covariate in ("medication_text", "age_at_onset", "PANSS_positive", "PANSS_general"):
                rows.append(dict(**row, status="coverage_only", reason=rationale))
                continue
            if len(d)<20 or values.nunique()<3 or n_nonzero<5:
                rows.append(dict(**row, status="skipped", reason="Insufficient complete N or effective variation"))
                continue
            same = effect(model(d,scale,ANT),use_t=True)
            design = covariate_frame(d, (*COVARIATES,covariate))
            design["TEPS_or_ACIPS"] = z(d[scale])
            x = design.to_numpy()
            add_vif = variance_inflation_factor(x, design.columns.get_loc(covariate))
            if not np.isfinite(add_vif) or add_vif>=5:
                rows.append(dict(**row,status="skipped",reason="Added covariate VIF >=5",added_covariate_vif=add_vif))
                continue
            adjusted = fit(z(d[ANT]),design)
            a = effect(adjusted,use_t=True)
            extra = effect(adjusted,covariate,use_t=True)
            rows.append(dict(**row,status="executed", residual_df=adjusted.df_resid, added_covariate_vif=add_vif,
                             beta_baseline_same_subset=same["beta"], baseline_ci_low=same["ci_low"], baseline_ci_high=same["ci_high"],
                             beta=a["beta"], ci_low=a["ci_low"], ci_high=a["ci_high"], p=a["p"],
                             delta_beta=a["beta"]-same["beta"], added_covariate_beta=extra["beta"],
                             added_covariate_ci_low=extra["ci_low"], added_covariate_ci_high=extra["ci_high"],
                             added_covariate_p_descriptive=extra["p"]))
            if covariate in ("PANSS_negative", "PANSS_total","SANS"):
                # Reuse the recorded numeric conflict ID; availability differences are handled by complete cases.
                disputed_id = os.environ["STRIATAL_DISPUTED_SUBJECT_ID"]
                has_disputed = disputed_id in d.index
                if has_disputed:
                    no_dispute = d.drop(index=disputed_id)
                    base = effect(model(no_dispute,scale,ANT),use_t=True)
                    sens = effect(model(no_dispute,scale,ANT,(*COVARIATES,covariate)),use_t=True)
                    rows.append(dict(scale=scale, record_type="numeric_source_conflict_sensitivity",
                                     clinical_variable=covariate,n=len(no_dispute),status="executed",
                                     sample_ids=";".join(no_dispute.index), omitted_ID=disputed_id,
                                     beta_baseline_same_subset=base["beta"], baseline_ci_low=base["ci_low"],
                                     baseline_ci_high=base["ci_high"], beta=sens["beta"],ci_low=sens["ci_low"],
                                     ci_high=sens["ci_high"],p=sens["p"],delta_beta=sens["beta"]-base["beta"]))
    write_csv(pd.DataFrame(rows), out/"04_CLINICAL_COVARIATE_SENSITIVITY.csv")
