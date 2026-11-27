"""Phase 2: trace the association to frozen voxelwise FC and target profiles."""
import numpy as np
import pandas as pd

from followup_common import (
    ANT, CORE, CURRENT_BINS, COVARIATES, ID, TEPS_C, TEPS_T, bh, cases, center_unit,
    close, covariate_frame, distribution, effect, fisher, fit, model, residual, similarity, write_csv, z,
)

REGIONS = ("SMA", "IFG", "Cingulate")
BIN_NAMES = tuple(range(1, 11))


def statistics(data, outcome, scale, **labels):
    d = cases(data, scale, outcome)
    return dict(scale=scale, metric=outcome, **labels, **effect(model(d, scale, outcome), use_t=True))


def phase2_bin_profiles(src):
    is_hc = src.data.Group.eq("HC").to_numpy()
    mask = src.stage_mask("Anticipation")
    fc = src.current_profiles[..., mask]
    ref = src.templates[..., mask]
    ufc, uref = center_unit(fc), center_unit(ref)
    r = (ufc * uref).sum(axis=-1)
    stage_mean = fc.mean(axis=-1)
    offset = stage_mean - ref.mean(axis=-1)
    fc_centered = fc - stage_mean[..., None]
    ref_centered = ref - ref.mean(axis=-1, keepdims=True)
    centered_rmse = np.sqrt(np.mean((fc_centered-ref_centered)**2, axis=-1))
    rmse = np.sqrt(np.mean((fc-ref)**2, axis=-1))
    assert close(rmse**2, centered_rmse**2+offset**2)
    a = src.profiles[..., mask]
    a_ref = np.broadcast_to(a[is_hc].mean(axis=0), a.shape).copy()
    a_ref[is_hc] = (a[is_hc].sum(axis=0)-a[is_hc])/31
    a_r = similarity(a, a_ref)

    df = src.data[["Group", *COVARIATES, *CORE, "Anticipation_slope", ANT]].copy()
    for j, b in enumerate(BIN_NAMES):
        df[f"current_bin{b:02d}_mean_fc"] = stage_mean[:, j]
        df[f"current_bin{b:02d}_similarity"] = fisher(r[:, j])
    df["current_bin10_centered_RMSE"] = centered_rmse[:, -1]
    df["current_bin10_target_SD"] = fc[:, -1].std(axis=-1, ddof=1)
    df["current_across_bins_mean_fc"] = stage_mean.mean(axis=1)
    df["exploratory_Definition_A_bin10_similarity"] = fisher(a_r[:, -1])
    assert close(df.current_bin10_similarity, df[ANT])
    models = []
    metrics = [
        ("current_bin10_centered_RMSE", "G1_bin_current", "Centered FC RMSE"),
        ("current_bin10_target_SD", "G1_bin_current", "Across-target FC SD"),
        ("current_across_bins_mean_fc", "G1_bin_current", "Mean FC across all 10 bins"),
        ("Anticipation_slope", "Definition_A", "Frozen overall FC slope"),
        ("exploratory_Definition_A_bin10_similarity", "Definition_A", "Alternative Bin 10 similarity"),
    ]
    sz = df.loc[~is_hc].copy()
    for scale in CORE:
        for b in BIN_NAMES:
            for name, label in [("mean_fc", "Mean FC"), ("similarity", "Profile similarity")]:
                models.append(statistics(sz, f"current_bin{b:02d}_{name}", scale, bin=b,
                                         bin_scheme=CURRENT_BINS, family="exploratory_42_bin_and_localization_tests",
                                         category=label, status="formal endpoint refit" if b==10 and name=="similarity" else "exploratory"))
        d = cases(sz, scale, ANT)
        y = z(d[[f"current_bin{b:02d}_similarity" for b in BIN_NAMES]])
        des = covariate_frame(d)
        des["TEPS_or_ACIPS"] = z(d[scale])
        contrast = effect(fit(y[:, -1]-y[:, :-1].mean(axis=1), des), use_t=True)
        models.append(dict(scale=scale, metric="Bin10_minus_mean_of_Bin01_to_Bin09_similarity_betas", bin=10,
                           bin_scheme=CURRENT_BINS, family="exploratory_42_bin_and_localization_tests",
                           category="Bin 10 specificity contrast", status="exploratory", **contrast))
        for metric, scheme, category in metrics:
            models.append(statistics(sz, metric, scale, bin=10 if "bin10" in metric else np.nan,
                                     bin_scheme=scheme, family="exploratory_10_FC_descriptors",
                                     category=category, status="exploratory"))
    models = pd.DataFrame(models)
    models["exploratory_q"] = models.groupby("family")["p"].transform(bh)
    assert models.groupby("family").size().to_dict() == {
        "exploratory_42_bin_and_localization_tests": 42, "exploratory_10_FC_descriptors": 10}

    long = pd.DataFrame({
        ID: np.repeat(df.index.to_numpy(), 10), "Group": np.repeat(df.Group.to_numpy(), 10),
        "bin": np.tile(BIN_NAMES, len(df)), "bin_n_voxels": np.tile(np.bincount(src.current_bins)[1:], len(df)),
        "current_mean_FC": stage_mean.ravel(), "reference_mean_FC": ref.mean(axis=-1).ravel(),
        "HC32_mean_FC": np.tile(stage_mean[is_hc].mean(axis=0), len(df)),
        "mean_FC_minus_own_reference": offset.ravel(), "centered_RMSE_FC": centered_rmse.ravel(),
        "RMSE_FC": rmse.ravel(), "across_target_FC_SD": fc.std(axis=-1, ddof=1).ravel(),
        "current_profile_raw_r": r.ravel(), "current_profile_Fisher_z": fisher(r).ravel(),
        "Definition_A_mean_FC": a.mean(axis=-1).ravel(), "frozen_Definition_A_slope": np.repeat(df.Anticipation_slope,10),
        TEPS_C: np.repeat(df[TEPS_C], 10), TEPS_T: np.repeat(df[TEPS_T], 10),
        "template": np.repeat(np.where(is_hc, "HC31_LOO", "HC32"), 10),
    })
    summary = []
    for pop, d in [("HC32", long[long.Group.eq("HC")]), ("SZ32", long[long.Group.eq("SZ")]),
                   ("SZ28", long[long.Group.eq("SZ") & long[TEPS_C].notna()])]:
        for b, rows in d.groupby("bin"):
            for metric in ["current_mean_FC", "current_profile_raw_r", "Definition_A_mean_FC",
                           "centered_RMSE_FC", "across_target_FC_SD"]:
                summary.append(dict(population=pop, bin=b, metric=metric, **distribution(rows[metric])))
    return models, long, pd.DataFrame(summary)


def target_profiles(src):
    is_hc = src.data.Group.eq("HC").to_numpy()
    mask = src.stage_mask("Anticipation")
    x, ref = src.current_profiles[:, -1, mask], src.templates[:, -1, mask]
    hc32 = x[is_hc].mean(axis=0)
    unit = center_unit(x)
    contribution = unit * center_unit(ref)
    meta = src.mapping.merge(src.read("subfamily_map"), on=["roi_index","roi_name"],
                             validate="one_to_one").loc[mask].reset_index(drop=True)
    rank = pd.Series(hc32).rank(method="first").astype(int).to_numpy()
    assert close(contribution.sum(axis=1), similarity(x, ref))
    long = pd.DataFrame({
        ID: np.repeat(src.data.index.to_numpy(), 43), "Group": np.repeat(src.data.Group.to_numpy(), 43),
        "target_ROI": np.tile(meta.roi_name, 64), "subfamily": np.tile(meta.subfamily, 64),
        "ROI_index0": np.tile(meta.roi_index, 64), "HC_mean_order": np.tile(rank, 64),
        "FC_Fisher_z": x.ravel(), "HC32_FC_Fisher_z": np.tile(hc32, 64),
        "own_template_FC_Fisher_z": ref.ravel(),
        "own_template": np.repeat(np.where(is_hc, "HC31_LOO", "HC32"), 43),
        "centered_unit_subject_FC": unit.ravel(), "centered_unit_HC32_FC": np.tile(center_unit(hc32), 64),
        "raw_r_target_contribution_43": contribution.ravel(),
        TEPS_C: np.repeat(src.data[TEPS_C], 43), TEPS_T: np.repeat(src.data[TEPS_T], 43),
    })

    estimates, subfamilies, tail_profiles = [], [], []
    for scale in CORE:
        d = cases(src.sz, scale, ANT)
        idx = src.data.index.get_indexer(d.index)
        cov = covariate_frame(d)
        x_resid = residual(z(d[scale]), cov)
        weights = x_resid / (x_resid @ x_resid)
        additive = weights @ contribution[idx]
        assert close(additive.sum(), weights @ contribution[idx].sum(axis=-1))
        full = effect(model(d, scale, ANT))["beta"]
        for j, roi in meta.iterrows():
            keep = np.arange(43) != j
            dropped = similarity(x[idx][:, keep], hc32[keep])
            beta_drop = weights @ z(fisher(dropped))
            estimates.append(dict(scale=scale, n=len(d), target_ROI=roi.roi_name, subfamily=roi.subfamily,
                                  ROI_index0=roi.roi_index, effect_full_Fisher_z=full,
                                  effect_drop_target_Fisher_z=beta_drop, delta_drop=beta_drop-full,
                                  contribution_to_beta_in_raw_r_units=additive[j],
                                  inferential_status="Descriptive leave-one-target-out; no tests"))
        for family, members in meta.groupby("subfamily", sort=True):
            keep = ~meta.subfamily.eq(family).to_numpy()
            drop_r = similarity(x[idx][:, keep], hc32[keep])
            beta_drop = weights @ z(fisher(drop_r))
            subfamilies.append(dict(scale=scale, n=len(d), removed_subfamily=family, removed_targets=len(members),
                                    effect_full_Fisher_z=full, effect_drop_subfamily_Fisher_z=beta_drop,
                                    delta_drop=beta_drop-full, contribution_to_beta_in_raw_r_units=additive[~keep].sum(),
                                    inferential_status="Descriptive leave-one-subfamily-out; no tests"))
        for tail, test in [("Lower TEPS quartile", d[scale]<=d[scale].quantile(.25)),
                           ("Upper TEPS quartile", d[scale]>=d[scale].quantile(.75))]:
            mask_ids = idx[test.to_numpy()]
            for j, roi in meta.iterrows():
                tail_profiles.append(dict(scale=scale, group=tail, n=len(mask_ids), target_ROI=roi.roi_name,
                                          subfamily=roi.subfamily, HC_mean_order=rank[j],
                                          HC32_FC_Fisher_z=hc32[j], mean_FC_Fisher_z=x[mask_ids,j].mean(),
                                          mean_centered_unit_FC=unit[mask_ids,j].mean(),
                                          HC32_centered_unit_FC=center_unit(hc32)[j],
                                          mean_raw_r_target_contribution=contribution[mask_ids,j].mean()))
    return long, pd.DataFrame(estimates), pd.DataFrame(subfamilies), pd.DataFrame(tail_profiles)


def regional_analyses(src):
    roi_map = src.read("roi_map_3")
    hist_con = src.read("historical_contribution").set_index("subfamily")
    hist_raw = src.read("historical_raw_fc_roi").query("Test == 'Linear slope'").set_index("Region")
    original = src.read("historical_individual_contribution")
    expected = original.pivot(index="subject", columns="roi_index0", values="r_contribution").reindex(src.data.index)
    contrib = center_unit(src.current_profiles[:, -1, :]) * center_unit(src.templates[:, -1, :])
    assert close(contrib, expected)
    assert close(src.current_profiles[:, -1, :], original.pivot(index="subject", columns="roi_index0",
                                                               values="subject_FC").reindex(src.data.index))
    table, subjects = [], []
    k = np.arange(1,11)-5.5
    for region in REGIONS:
        m = roi_map[roi_map.Region.eq(region)]
        indices = m.ROI_array_column_0based.to_numpy()
        assert np.array_equal(src.mapping.iloc[indices].roi_name, m.ROI)
        assert set(m.Stage) == set(src.mapping.iloc[indices].reward_stage)
        original_slopes = (src.profiles[..., indices].mean(axis=-1) @ k)/(k @ k)
        assert close(original_slopes, src.data[f"{region}_slope"])
        d = src.data[["Group", *COVARIATES, *CORE, f"{region}_slope"]].copy()
        d[f"{region}_current_Bin10_FC"] = src.current_profiles[:, -1, indices].mean(axis=-1)
        d[f"{region}_current_Bin10_65_target_contribution"] = contrib[:, indices].sum(axis=-1)
        subjects.append(d.reset_index().rename(columns={
            f"{region}_slope": "Definition_A_slope", f"{region}_current_Bin10_FC": "current_Bin10_FC",
            f"{region}_current_Bin10_65_target_contribution": "current_Bin10_65_target_contribution"
        }).assign(region=region, stage=m.Stage.iloc[0], target_n=len(m)))
        raw = hist_raw.loc[region]
        h = hist_con.loc[region]
        for scale in CORE:
            for key, bin_scheme, measure in [
                (f"{region}_slope", "Definition_A", "Frozen raw-FC slope"),
                (f"{region}_current_Bin10_FC", CURRENT_BINS, "Bin 10 mean raw FC"),
                (f"{region}_current_Bin10_65_target_contribution", CURRENT_BINS, "Frozen 65-target Pearson-r contribution"),
            ]:
                table.append(statistics(d[d.Group.eq("SZ")], key, scale, region=region, stage=m.Stage.iloc[0],
                                        bin_scheme=bin_scheme, target_n=len(m), measure=measure,
                                        historical_SZ_minus_HC_65_contribution=h.beta_SZ_minus_HC,
                                        historical_65_contribution_q=h.q_FDR_13_assigned,
                                        historical_rawFC_slope_group_effect=raw.Beta_or_F,
                                        historical_rawFC_slope_group_q_BH9=raw.BH_q_9_tests,
                                        family="exploratory_18_regional_TEPS_tests"))
    results = pd.DataFrame(table)
    assert len(results)==18
    results["exploratory_q"] = bh(results.p)
    return results, pd.concat(subjects)


def run_phase2(src, out):
    models, profiles, summaries = phase2_bin_profiles(src)
    targets, loto, subfamilies, tails = target_profiles(src)
    regions, regional_data = regional_analyses(src)
    exports = {
        "G1_FC_PROFILE_RESULTS": models,
        "G1_FC_PROFILE_SUMMARIES": summaries,
        "ANTICIPATION_FC_PROFILE_SUBJECTS": profiles,
        "BIN10_ANTICIPATION_TARGET_FC": targets,
        "BIN10_LEAVE_ONE_TARGET_OUT": loto,
        "BIN10_LEAVE_ONE_SUBFAMILY_OUT": subfamilies,
        "BIN10_TARGET_PROFILES_BY_TEPS_TAIL": tails,
        "ROI_TEPS_ASSOCIATIONS": regions,
        "ROI_SUBJECT_VALUES": regional_data,
    }
    for name, table in exports.items():
        write_csv(table, out / f"02_{name}.csv")
    fig = pd.concat([
        profiles[profiles.Group.eq("SZ") & profiles[TEPS_C].notna()].assign(kind="profile"),
        summaries[summaries.metric.eq("current_mean_FC") & summaries.population.isin(["HC32","SZ28"])].assign(kind="profile_mean"),
        models[models.scale.eq(TEPS_C) & models.category.isin(["Mean FC","Profile similarity"])].assign(kind="bin_effect"),
        tails[tails.scale.eq(TEPS_C)].assign(kind="tail_pattern"),
    ], ignore_index=True)
    write_csv(fig, out / "figures/source_data/Figure_06_data.csv")
