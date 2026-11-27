"""Phase 1: raw reference positions, complete-case residuals and robustness."""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from followup_common import (
    ANT, BOOT_N, CORE, COVARIATES, ID, OUTCOME, PERM_N, SEED, SPLIT_N, TEPS_A,
    TEPS_C, TEPS_T, bootstrap_betas, cases, center_unit, close, covariate_frame,
    distribution, effect, fisher, model, percentile, residual, similarity, write_csv, z,
)

RAW = "Anticipation_raw_similarity_r"
MATCHED = "exploratory_HC31_mean_r"


def positions(values, low, high):
    return np.select([values < low, values > high], ["below", "above"], default="within")


def reference_sensitivity(src, participants):
    """Compute size-matched HC31 and common-template HC16 descriptions."""
    hc = src.data.Group.eq("HC").to_numpy()
    x = src.current_profiles[:, -1, src.stage_mask("Anticipation")]
    hx, sx = x[hc], x[~hc]
    hc_r = participants.loc[hc, RAW].to_numpy()
    sz_r = participants.loc[~hc, RAW].to_numpy()
    d31 = center_unit(sx) @ center_unit((hx.sum(axis=0)-hx)/31).T
    assert close(similarity(hx, (hx.sum(axis=0)-hx)/31), hc_r)
    participants[MATCHED] = participants[RAW]
    participants.loc[~hc, MATCHED] = d31.mean(axis=1)
    participants["exploratory_HC31_min_r"] = np.nan
    participants["exploratory_HC31_max_r"] = np.nan
    participants.loc[~hc, "exploratory_HC31_min_r"] = d31.min(axis=1)
    participants.loc[~hc, "exploratory_HC31_max_r"] = d31.max(axis=1)
    participants["exploratory_HC31_change_r"] = participants[MATCHED] - participants[RAW]
    participants["exploratory_HC31_HC_relative_z"] = (participants[MATCHED]-hc_r.mean()) / hc_r.std(ddof=1)

    sz = participants.loc[~hc]
    teps_ok = sz[TEPS_C].notna().to_numpy()
    assert np.array_equal(teps_ok, sz[TEPS_T].notna())
    reference_stats = distribution(hc_r)
    p10, p90 = reference_stats["p10"], reference_stats["p90"]
    participants["exploratory_HC31_position"] = positions(participants[MATCHED], p10, p90)
    participants["exploratory_HC31_fraction_below"] = np.nan
    participants["exploratory_HC31_fraction_above"] = np.nan
    participants.loc[~hc, "exploratory_HC31_fraction_below"] = (d31 < p10).mean(axis=1)
    participants.loc[~hc, "exploratory_HC31_fraction_above"] = (d31 > p90).mean(axis=1)
    hc31_folds = pd.DataFrame({
        "held_out_HC_ID": participants.index[hc],
        "held_out_HC_raw_r": hc_r,
        "mean_SZ32_r_same_HC31_template": d31.mean(axis=0),
        "mean_SZ28_r_same_HC31_template": d31[teps_ok].mean(axis=0),
        "mean_SZ32_minus_heldout_HC": d31.mean(axis=0)-hc_r,
        "mean_SZ28_minus_heldout_HC": d31[teps_ok].mean(axis=0)-hc_r,
    })
    pairwise31 = []
    for mask, pop in [(np.ones(len(sz_r), dtype=bool), "SZ32"), (teps_ok, "SZ28")]:
        pairwise31.append(dict(design="matched_HC31", population=pop, reference_size=31,
                               mean_raw_r_difference=(d31[mask]-hc_r).mean(),
                               sz_less_than_hc_pair_fraction=(d31[mask] < hc_r).mean(),
                               n_subjects_sz=mask.sum(), n_subjects_hc=32))

    rng = np.random.default_rng(SEED)
    indices = np.arange(32)
    hc_unit, sz_unit = center_unit(hx), center_unit(sx)
    h_ids = participants.index[hc].to_numpy()
    c = covariate_frame(sz.loc[teps_ok])
    xr = {scale: residual(z(sz.loc[teps_ok, scale]), c) for scale in CORE}
    q75 = {scale: sz.loc[teps_ok, scale].quantile(.75) for scale in CORE}
    high = {scale: (sz[scale] >= q75[scale]).to_numpy() for scale in CORE}
    fold_rows, sz_values, sz_below, sz_above = [], [], [], []
    for split in range(SPLIT_N):
        a = np.sort(rng.choice(32, 16, replace=False))
        b = np.setdiff1d(indices, a)
        for side, (train, test) in enumerate(((a, b), (b, a)), start=1):
            template = center_unit(hx[train].mean(axis=0))
            hr, sr = hc_unit[test] @ template, sz_unit @ template
            lo, hi = np.quantile(hr, [.1, .9])
            low, high_r = sr < lo, sr > hi
            sz_values.append(sr)
            sz_below.append(low)
            sz_above.append(high_r)
            yr = residual(z(fisher(sr[teps_ok])), c)
            row = dict(split=split+1, side=side, ref_n=16, held_out_HC_n=16, SZ_n=32, SZ_TEPS_n=teps_ok.sum(),
                       ref_HC_IDs=";".join(h_ids[train]), eval_HC_IDs=";".join(h_ids[test]),
                       HC_mean_r=hr.mean(), HC_p10=lo, HC_p90=hi,
                       SZ32_mean_r=sr.mean(), SZ28_mean_r=sr[teps_ok].mean(),
                       SZ32_minus_HC=sr.mean()-hr.mean(), SZ28_minus_HC=sr[teps_ok].mean()-hr.mean(),
                       SZ32_below=low.mean(), SZ28_below=low[teps_ok].mean(),
                       SZ32_within=(~(low|high_r)).mean(), SZ28_within=(~(low|high_r))[teps_ok].mean(),
                       SZ32_above=high_r.mean(), SZ28_above=high_r[teps_ok].mean())
            for scale in CORE:
                mask = high[scale]
                row[f"beta_{scale}"] = xr[scale] @ yr / (xr[scale] @ xr[scale])
                row[f"top_quartile_{scale}_n"] = mask.sum()
                row[f"top_quartile_{scale}_below"] = low[mask].mean()
                row[f"top_quartile_{scale}_within"] = (~(low|high_r))[mask].mean()
                row[f"top_quartile_{scale}_above"] = high_r[mask].mean()
            fold_rows.append(row)
    splits = pd.DataFrame(fold_rows)
    for name, values in [("mean_r", sz_values), ("fraction_below", sz_below), ("fraction_above", sz_above)]:
        col = f"exploratory_HC16_{name}"
        participants[col] = np.nan
        participants.loc[~hc, col] = np.mean(values, axis=0)
    participants["exploratory_HC16_fraction_within"] = 1 - participants.exploratory_HC16_fraction_below - participants.exploratory_HC16_fraction_above
    return participants, hc31_folds, splits, pairwise31


def build_subject_table(src):
    data = src.data[[ "Group", *COVARIATES, TEPS_A, TEPS_C, TEPS_T, ANT, OUTCOME]].copy()
    data = data.rename(columns={ANT: "Anticipation_Fisher_z_similarity", OUTCOME: "Outcome_Fisher_z_similarity"})
    for stage in ("Anticipation", "Outcome"):
        d = src.similarity_long.query("Stage == @stage").set_index(ID).reindex(data.index)
        data[f"{stage}_raw_similarity_r"] = d.Similarity_Raw_r
        data[f"{stage}_formal_template"] = d.Reference_Pattern_ID
    h = data.loc[data.Group.eq("HC")]
    raw_ref = distribution(h[RAW])
    for name, field in [("HC_relative_similarity_z", RAW),
                        ("HC_relative_Fisher_similarity_z", "Anticipation_Fisher_z_similarity")]:
        data[name] = (data[field] - h[field].mean()) / h[field].std()
    data["absolute_HC_deviation"] = data.HC_relative_similarity_z.abs()
    data["absolute_HC_Fisher_deviation"] = data.HC_relative_Fisher_similarity_z.abs()
    data["position_vs_HC_sample_P10_P90"] = positions(data[RAW], raw_ref["p10"], raw_ref["p90"])
    data["SZ_complete_core"] = data.Group.eq("SZ") & data[TEPS_C].notna()
    for scale in (TEPS_A, *CORE):
        d = cases(src.sz, scale, ANT)
        cov = covariate_frame(d)
        for prefix, raw_field in [(scale, scale), (f"{scale}_Anticipation", ANT)]:
            v = d[raw_field]
            data.loc[d.index, f"{prefix}_raw_residual"] = residual(v, cov)
            data.loc[d.index, f"{prefix}_PPT_residual"] = residual(z(v), cov)
            data.loc[d.index, f"{prefix}_standardized_residual"] = z(residual(v, cov))
        data.loc[d.index, f"{scale}_Anticipation_Pearson_r_raw_residual"] = residual(data.loc[d.index, RAW], cov)
        data.loc[d.index, f"{scale}_Anticipation_Pearson_r_standardized_residual"] = z(
            data.loc[d.index, f"{scale}_Anticipation_Pearson_r_raw_residual"])
        x = data.loc[d.index, f"{scale}_PPT_residual"]
        y = data.loc[d.index, f"{scale}_Anticipation_PPT_residual"]
        x_sign = np.where(x > 0, "+", np.where(x < 0, "-", "0"))
        y_sign = np.where(y > 0, "+", np.where(y < 0, "-", "0"))
        data.loc[d.index, f"{scale}_quadrant"] = pd.Series(
            np.char.add(np.char.add("x", x_sign), np.char.add(" y", y_sign)), index=d.index)
    return data


def distribution_tables(data, splits, pairwise31):
    summaries = []
    tail_rows = []
    for field in [RAW, "Anticipation_Fisher_z_similarity", MATCHED]:
        for label, mask in [("HC32", data.Group.eq("HC")), ("SZ32", data.Group.eq("SZ")),
                            ("SZ28", data.SZ_complete_core), ("SZ4_TEPS_missing", data.Group.eq("SZ") & ~data.SZ_complete_core)]:
            d = data.loc[mask]
            h = data.loc[data.Group.eq("HC"), field]
            lo, hi = h.quantile([.1, .9])
            pos = positions(d[field], lo, hi)
            summaries.append(dict(population=label, similarity=field, **distribution(d[field]),
                                  n_below_HC_mean=(d[field]<h.mean()).sum(), n_below_HC_p10=(pos=="below").sum(),
                                  n_within_HC_p10_p90=(pos=="within").sum(), n_above_HC_p90=(pos=="above").sum()))
    for scale in CORE:
        s = data.loc[data.SZ_complete_core]
        q25, q75 = s[scale].quantile([.25, .75])
        for name, selection, cut in [("bottom_quartile", s[scale]<=q25, q25), ("top_quartile", s[scale]>=q75, q75)]:
            d = s.loc[selection]
            for field, pos_col in [(RAW, "position_vs_HC_sample_P10_P90"), (MATCHED, "exploratory_HC31_position")]:
                ref = data.loc[data.Group.eq("HC"), field]
                tail_rows.append(dict(scale=scale, tail=name, cutoff=cut, tie_rule="include ties",
                                      reference_metric=field, raw_score_min=d[scale].min(), raw_score_max=d[scale].max(),
                                      ids=";".join(d.index), **distribution(d[field]),
                                      median_HC_relative_z=((d[field]-ref.mean())/ref.std()).median(),
                                      n_below_HC_mean=(d[field]<ref.mean()).sum(),
                                      n_below_HC_p10=d[pos_col].eq("below").sum(),
                                      n_within_HC_p10_p90=d[pos_col].eq("within").sum(),
                                      n_above_HC_p90=d[pos_col].eq("above").sum(),
                                      shared_HC16_below_fraction=d.exploratory_HC16_fraction_below.mean(),
                                      shared_HC16_within_fraction=d.exploratory_HC16_fraction_within.mean(),
                                      shared_HC16_above_fraction=d.exploratory_HC16_fraction_above.mean()))
    result_rows = [dict(design="shared_HC16", metric=metric, reference_size=16,
                        **distribution(splits[metric])) for metric in splits.columns
                   if metric in ["SZ32_minus_HC", "SZ28_minus_HC", *[f"beta_{s}" for s in CORE]]
                   or metric.startswith("top_quartile_") and not metric.endswith("_n")]
    for pop, mask in [("SZ32", data.Group.eq("SZ")), ("SZ28", data.SZ_complete_core)]:
        h = data.loc[data.Group.eq("HC"), RAW]
        d = data.loc[mask]
        result_rows.append(dict(design="frozen_HC31_vs_HC32", population=pop,
                                mean_raw_r_difference=d[RAW].mean()-h.mean(), n_subjects_hc=32,
                                n_subjects_sz=len(d)))
        result_rows.append(dict(design="matched_HC31", population=pop, metric="SZ_r_change_vs_HC32",
                                reference_size=31, **distribution(d[MATCHED]-d[RAW])))
    result_rows.extend(pairwise31)
    return pd.DataFrame(summaries), pd.DataFrame(tail_rows), pd.DataFrame(result_rows)


def permute_freedman_lane(d, fitted, rng):
    """Permute reduced-model residuals, keeping the full design fixed."""
    x = np.asarray(fitted.model.exog)
    y = fitted.model.endog
    nuisance = covariate_frame(d)
    e = residual(y, nuisance)
    order = np.argsort(rng.random((PERM_N, len(d))), axis=1)
    y_star = (y-e) + e[order]
    inv = np.linalg.pinv(x)
    betas = y_star @ inv.T
    target = fitted.model.exog_names.index("TEPS_or_ACIPS")
    hat = fitted.get_influence().hat_matrix_diag
    residuals = y_star - betas @ x.T
    se = np.sqrt((residuals**2) @ ((inv[target]/(1-hat))**2))
    empirical = np.abs(betas[:, target] / se)
    observed = abs(fitted.params.iloc[target] / effect(fitted)["se"])
    exceed = int(np.count_nonzero(empirical >= observed))
    return dict(p=(exceed+1)/(PERM_N+1), n_resamples=PERM_N, n_valid=PERM_N,
                observed_abs_stat=observed, null_exceedances=exceed)


def model_robustness(src, data):
    cov_rows, all_rows, inf_rows, loo_rows, boot_rows, line_rows = [], [], [], [], [], []
    historical = src.read("within").query("population == 'SZ' and imaging_variable == @ANT").set_index("scale_variable")
    cov_models = [
        ("Unadjusted", ()), ("Age only", ("Age",)), ("Sex only", ("Sex",)),
        ("MeanFD only", ("MeanFD",)), ("Age + Sex", ("Age", "Sex")),
        ("Age + MeanFD", ("Age", "MeanFD")), ("Sex + MeanFD", ("Sex", "MeanFD")),
        ("Age + Sex + MeanFD", COVARIATES),
    ]
    for scale_index, scale in enumerate(CORE):
        rng = np.random.default_rng(SEED + 100 + scale_index)
        d = cases(src.sz, scale, ANT).copy()
        d[RAW] = data.loc[d.index, RAW]
        d["exploratory_matched_HC31_Fisher_z"] = fisher(data.loc[d.index, MATCHED])
        y = d[RAW].to_numpy()
        x = d[scale].to_numpy()
        full = model(d, scale, ANT)
        main = effect(full)
        main_t = effect(full, use_t=True)
        h = historical.loc[scale]
        assert close(main["beta"], h.effect) and len(d)==28
        base = dict(scale=scale, outcome=ANT, estimate_type="standardized_beta")
        all_rows.extend([
            dict(**base, method="Adjusted OLS HC3 (historical Wald)", **main, historical_q=h.p_adj_bh_within_population),
            dict(**base, method="Adjusted OLS HC3 (t)", **main_t, residual_df=full.df_resid),
        ])
        samples = rng.integers(0, len(d), size=(BOOT_N, len(d)))
        bs = bootstrap_betas(d, (scale,), (ANT,), samples)[:, 0, 0]
        low, high = percentile(bs)
        assert np.isfinite(bs).mean() > .99
        all_rows.append(dict(**base, method="Adjusted OLS participant bootstrap", n=len(d), beta=main["beta"],
                             ci_low=low, ci_high=high, n_resamples=BOOT_N,
                             n_valid=np.isfinite(bs).sum(), negative_fraction=np.nanmean(bs[np.isfinite(bs)]<0)))
        boot_rows.append(pd.DataFrame(dict(scale=scale, draw=np.arange(1, BOOT_N+1), beta=bs)))
        all_rows.append(dict(**base, method="Adjusted OLS Freedman-Lane", n=len(d), beta=main["beta"],
                             **permute_freedman_lane(d, full, rng)))
        pr = stats.pearsonr(x, y)
        sp = stats.spearmanr(x, y)
        sp_samples = (z(stats.rankdata(x[samples], axis=1), axis=1) *
                      z(stats.rankdata(y[samples], axis=1), axis=1)).sum(axis=1)/(len(d)-1)
        sp_lo, sp_hi = percentile(sp_samples)
        all_rows.extend([
            dict(scale=scale, outcome=RAW, method="Raw Pearson", estimate_type="r", n=len(d), beta=pr.statistic,
                 ci_low=pr.confidence_interval().low, ci_high=pr.confidence_interval().high, p=pr.pvalue),
            dict(scale=scale, outcome=RAW, method="Raw Spearman", estimate_type="rho", n=len(d), beta=sp.statistic,
                 ci_low=sp_lo, ci_high=sp_hi, p=sp.pvalue, n_resamples=BOOT_N, n_valid=np.isfinite(sp_samples).sum()),
        ])
        for metric in [RAW, ANT]:
            for label, terms in [("Unadjusted OLS HC3 in observed units", ()),
                                 ("Adjusted OLS HC3 in observed units", COVARIATES)]:
                res = effect(model(d, scale, metric, terms, standardized=False), use_t=True)
                all_rows.append(dict(scale=scale, outcome=metric, method=label,
                                     estimate_type="similarity_per_TEPS_point", **res))

        raw_fit = model(d, scale, RAW, (), standardized=False)
        grid = np.linspace(x.min(), x.max(), 101)
        design = pd.DataFrame({"Intercept": 1., "TEPS_or_ACIPS": grid})
        pred = raw_fit.get_robustcov_results(cov_type="HC3", use_t=True).get_prediction(design).summary_frame()
        line_rows.append(pd.DataFrame(dict(scale=scale, score=grid, line_r=pred["mean"],
                                          line_ci_low=pred.mean_ci_lower, line_ci_high=pred.mean_ci_upper)))

        rlm = sm.RLM(full.model.endog, full.model.exog, M=sm.robust.norms.HuberT()).fit(cov="H2")
        ix = full.model.exog_names.index("TEPS_or_ACIPS")
        b, se = rlm.params[ix], rlm.bse[ix]
        all_rows.append(dict(**base, method="Huber M regression (H2)", n=len(d), beta=b, se=se,
                             ci_low=b-1.96*se, ci_high=b+1.96*se, p=2*stats.norm.sf(abs(b/se))))
        matched = effect(model(d, scale, "exploratory_matched_HC31_Fisher_z"))
        all_rows.append(dict(scale=scale, outcome="exploratory_matched_HC31_Fisher_z",
                             estimate_type="standardized_beta", method="Matched HC31 template sensitivity", **matched))

        diagnostics = full.get_influence()
        n, parameters = len(d), len(full.params)
        influence = pd.DataFrame({ID: d.index, "scale": scale, "score": x,
                                  "raw_similarity_r": y, "PPT_residual_x": data.loc[d.index, f"{scale}_PPT_residual"],
                                  "PPT_residual_y": data.loc[d.index, f"{scale}_Anticipation_PPT_residual"],
                                  "cook_distance": diagnostics.cooks_distance[0],
                                  "leverage": diagnostics.hat_matrix_diag,
                                  "studentized_residual_external": diagnostics.resid_studentized_external,
                                  "dfbetas_TEPS": diagnostics.dfbetas[:, ix], "huber_weight": rlm.weights})
        thresholds = {"cook_distance": 4/n, "leverage": 2*parameters/n,
                      "studentized_residual_external": 3., "dfbetas_TEPS": 2/np.sqrt(n)}
        for field, threshold in thresholds.items():
            influence[f"threshold_{field}"] = threshold
            influence[f"flag_{field}"] = influence[field].abs()>threshold
        influence["flag_any"] = influence[[f"flag_{f}" for f in thresholds]].any(axis=1)
        flags = influence.loc[influence.flag_any, ID].tolist()
        reduced = effect(model(d.drop(index=flags), scale, ANT))
        assert len(flags)==h.n_influence_flags and close(reduced["beta"], h.flagged_excluded_effect)
        all_rows.append(dict(**base, method="Exclude all influence flags", **reduced,
                             n_excluded=len(flags), excluded_IDs=";".join(flags)))
        rows = []
        for subject in d.index:
            leave = effect(model(d.drop(index=subject), scale, ANT))
            rows.append(dict(scale=scale, omitted_ID=subject, **leave, full_beta=main["beta"],
                             full_ci_low=main["ci_low"], full_ci_high=main["ci_high"],
                             delta=leave["beta"]-main["beta"], direction_reversed=leave["beta"]*main["beta"]<0))
        loo = pd.DataFrame(rows).sort_values("beta")
        assert close([loo.beta.min(), loo.beta.max()], [h.loo_min, h.loo_max])
        loo["effect_order"] = np.arange(1, len(loo)+1)
        all_rows.append(dict(**base, method="Leave-one-out summary", n=n-1, beta=main["beta"],
                             loo_beta_min=loo.beta.min(), loo_beta_max=loo.beta.max(),
                             loo_ci_low_min=loo.ci_low.min(), loo_ci_low_max=loo.ci_low.max(),
                             loo_ci_high_min=loo.ci_high.min(), loo_ci_high_max=loo.ci_high.max(),
                             n_loo=len(loo), n_reversed=loo.direction_reversed.sum(), n_loo_ci_excludes_zero=(loo.ci_high<0).sum(),
                             max_delta_ID=loo.loc[loo.delta.abs().idxmax(), "omitted_ID"], max_abs_delta=loo.delta.abs().max()))
        inf_rows.append(influence.reset_index(drop=True).merge(loo[[ "omitted_ID", "beta", "delta"]].rename(
            columns={"omitted_ID": ID, "beta": "beta_if_omitted", "delta": "delta_if_omitted"}), on=ID, validate="one_to_one"))
        loo_rows.append(loo)

        for label, terms in cov_models:
            reg = model(d, scale, ANT, terms)
            primary = effect(reg)
            for term in reg.params.index.drop("Intercept"):
                res = effect(reg, term)
                cov_rows.append(dict(scale=scale, model=label, term=term, **res,
                                     delta_scale_beta_vs_full=primary["beta"]-main["beta"],
                                     coefficient_unit="Y SD, male vs female" if term.startswith("Sex_") else "Y SD / predictor SD"))
            if not terms:
                all_rows.append(dict(**base, method="Unadjusted OLS on Fisher z", **primary))
    return (pd.DataFrame(all_rows), pd.concat(inf_rows), pd.DataFrame(cov_rows),
            pd.concat(loo_rows), pd.concat(boot_rows), pd.concat(line_rows))


def quadrant_tables(data):
    rows = []
    for scale in CORE:
        d = data.loc[data.SZ_complete_core]
        for quad in ["x+ y+", "x- y+", "x- y-", "x+ y-"]:
            dq = d[d[f"{scale}_quadrant"].eq(quad)]
            fields = {"TEPS": scale, "raw_r": RAW, "HC_relative_z": "HC_relative_similarity_z"}
            summary = {f"{name}_{k}": v for name, col in fields.items() for k, v in distribution(dq[col]).items()}
            rows.append(dict(scale=scale, quadrant=quad, n=len(dq), **summary,
                             ids=";".join(dq.index),
                             below_HC_p10=dq.position_vs_HC_sample_P10_P90.eq("below").sum(),
                             within_HC_p10_p90=dq.position_vs_HC_sample_P10_P90.eq("within").sum(),
                             above_HC_p90=dq.position_vs_HC_sample_P10_P90.eq("above").sum()))
    return pd.DataFrame(rows)


def phase1_plot_data(out, data, distributions, robustness, quadrants, loo, boot, lines, splits):
    source = out / "figures/source_data"
    source.mkdir(parents=True, exist_ok=True)
    hc = distributions.query("population == 'HC32' and similarity == @RAW").iloc[0]
    group_1 = []
    group_3 = []
    for scale in CORE:
        d = data.loc[data.SZ_complete_core].reset_index()
        raw = d[[ID, RAW, scale]].rename(columns={scale: "score", RAW: "raw_r"}).assign(kind="point", scale=scale)
        line = lines[lines.scale.eq(scale)].assign(kind="line")
        values = robustness.loc[robustness.scale.eq(scale)].set_index("method")
        a = values.loc["Adjusted OLS HC3 (historical Wald)"]
        pr = values.loc["Raw Pearson"]
        annotation = dict(kind="annotation", scale=scale, n=a.n, beta=a.beta, ci_low=a.ci_low,
                          ci_high=a.ci_high, p_adjusted_model=a.p, historical_q=a.historical_q, pearson_r=pr.beta,
                          pearson_p=pr.p, HC_mean=hc["mean"], HC_p10=hc.p10, HC_p90=hc.p90)
        group_1.append(pd.concat([raw, line, pd.DataFrame([annotation])], ignore_index=True))
        points = raw.assign(kind="point", x_resid=d[f"{scale}_PPT_residual"], y_resid=d[f"{scale}_Anticipation_PPT_residual"],
                            quadrant=d[f"{scale}_quadrant"], HC_relative_z=d.HC_relative_similarity_z)
        counts = quadrants.loc[quadrants.scale.eq(scale), ["scale", "quadrant", "n"]].assign(kind="count")
        group_3.append(pd.concat([points, counts, pd.DataFrame([annotation])], ignore_index=True))
    write_csv(pd.concat(group_1), source / "Figure_01_data.csv")
    write_csv(pd.concat(group_3), source / "Figure_03_data.csv")
    fig2 = data.reset_index()[[ID, "Group", TEPS_C, RAW, MATCHED]]
    frozen_gap = data.loc[data.Group.eq("SZ"), RAW].mean()-hc["mean"]
    hc31_gap = data.loc[data.Group.eq("SZ"), MATCHED].mean()-hc["mean"]
    fig2 = fig2.rename(columns={RAW: "raw_r"}).assign(kind="point")
    rng = np.random.default_rng(SEED+200)
    jitter = pd.Series(rng.uniform(-.115, .115, len(data)), index=data.index)
    fig2["display_x"] = np.where(fig2.Group.eq("HC"), 1, 2) + fig2[ID].map(jitter)
    fig2 = fig2.assign(HC_mean=hc["mean"], HC_p10=hc.p10, HC_p90=hc.p90)
    fold_data = splits[["split", "side", "ref_n", "held_out_HC_n", "SZ_n", "SZ32_minus_HC"]].copy()
    fold_data["kind"] = "shared_HC16_fold"
    annotations = pd.DataFrame([dict(kind="group_gap_annotation", frozen_group_gap=frozen_gap,
                                     matched_HC31_group_gap=hc31_gap)])
    write_csv(pd.concat([fig2, fold_data, annotations], ignore_index=True), source / "Figure_02_data.csv")
    fig4 = pd.concat([loo.assign(kind="leave_one_out"), boot.assign(kind="bootstrap"),
                      robustness[robustness.method.eq("Adjusted OLS participant bootstrap")].assign(kind="summary")],
                     ignore_index=True)
    write_csv(fig4, source / "Figure_04_data.csv")
    fig5_methods = ["Unadjusted OLS on Fisher z", "Adjusted OLS HC3 (historical Wald)", "Adjusted OLS HC3 (t)",
                    "Adjusted OLS participant bootstrap", "Huber M regression (H2)", "Exclude all influence flags",
                    "Matched HC31 template sensitivity"]
    write_csv(robustness[robustness.method.isin(fig5_methods)], source / "Figure_05_data.csv")


def run_phase1(src, out):
    data, folds31, splits, pairs31 = reference_sensitivity(src, build_subject_table(src))
    distributions, tails, references = distribution_tables(data, splits, pairs31)
    robustness, influence, covariates, loo, bootstrap, lines = model_robustness(src, data)
    quadrants = quadrant_tables(data)
    exports = {
        "SUBJECT_RAW_AND_HC_RELATIVE_VALUES": data.reset_index(),
        "RAW_SIMILARITY_DISTRIBUTIONS": distributions,
        "TEPS_TAIL_REFERENCE_POSITIONS": tails,
        "REFERENCE_SENSITIVITY": references,
        "MATCHED_HC31_FOLDS": folds31,
        "SHARED_HC16_FOLDS": splits,
        "QUADRANT_SUMMARY": quadrants,
        "ROBUSTNESS_RESULTS": robustness,
        "INFLUENCE_DIAGNOSTICS": influence,
        "COVARIATE_SENSITIVITY": covariates,
        "LEAVE_ONE_OUT_RESULTS": loo,
    }
    for name, table in exports.items():
        write_csv(table, out / f"01_{name}.csv")
    phase1_plot_data(out, data, distributions, robustness, quadrants, loo, bootstrap, lines, splits)
