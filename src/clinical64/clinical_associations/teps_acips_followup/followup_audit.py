"""Phase 0: check sources, scoring, historical estimates, and figure axes."""
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from openpyxl import load_workbook

from followup_common import (
    ACIPS_A, ACIPS_C, ACIPS_T, ANT, CLINICAL, CLINICAL_HISTORY, CORE, COVARIATES,
    G1_VERSION, ID, KEYS, OUTCOME, REVERSE, SIM_TOL, TEPS_A, TEPS_C, TEPS_T,
    bh, cases, close, covariate_frame, distribution, effect, fisher, fit, model,
    residual, sha256, similarity, write_csv, z,
)


def audit_sources(src, out, qc):
    records = []
    for role, path in src.paths.items():
        records.append(dict(role=role, full_path=path.as_posix(), bytes=path.stat().st_size,
                            sha256=sha256(path)))
    provenance = pd.DataFrame(records)
    by_role = provenance.set_index("role")
    historical_hashes = src.read("historical_fingerprint_hashes").set_index("subject")
    assert set(historical_hashes.index) == set(src.data.index)
    for subject in src.data.index:
        path = src.fingerprint_path(subject)
        digest = sha256(path)
        assert digest == historical_hashes.at[subject, "sha256"], subject
        records.append(dict(role="voxel_FC_fingerprint", full_path=path.as_posix(),
                            bytes=path.stat().st_size, sha256=digest))
    provenance = pd.DataFrame(records)
    for row in src.read("manifest").itertuples():
        frozen = src.frozen / row.frozen_path.replace("\\", "/").rsplit("/", 1)[-1]
        assert sha256(frozen) == row.frozen_sha256, f"Frozen SHA256 differs for {frozen.name}"
    canonical_roles = {
        "g1": "canonical_G1_npy", "bins": "canonical_bins_definition_A_npy",
        "map": "canonical_reward_stage_mapping", "subfamily_map": "canonical_roi_subfamily_mapping",
        "canonical_covariates": "canonical_covariates",
    }
    can = src.read("can_manifest").set_index("input_role")
    for role, key in canonical_roles.items():
        assert by_role.at[role, "sha256"] == can.at[key, "frozen_sha256"], role
    for row in src.read("historical_provenance").itertuples():
        role = "sample" if row.input_name == "matrix" else row.input_name
        with src.paths[role].open("rb") as handle:
            assert hashlib.file_digest(handle, "md5").hexdigest() == row.md5, role
    assert list(np.bincount(np.load(src.paths["bins"]))[1:]) == [260, 260] + [261] * 8
    assert np.load(src.paths["g1"]).shape == (2608,)
    write_csv(provenance, out / "00_INPUT_PROVENANCE.csv")

    cross = src.read("crosswalk")
    included = cross[cross.Final64_Included.eq("YES")].set_index(ID, verify_integrity=True)
    assert len(cross) == 70 and len(src.data) == len(included) == 64
    assert included.Group.reindex(src.data.index).equals(src.data.Group)
    assert included.Final64_Analysis_ID.eq(included.index).all()
    assert included.Duplicate_ID_Flag.eq("NONE").all() and included.Unmatched_ID_Flag.eq("NONE").all()
    assert src.data.Group.value_counts().to_dict() == {"HC": 32, "SZ": 32}
    can_cov = src.read("canonical_covariates").set_index("Subject").reindex(src.data.index)
    assert can_cov.Sex.equals(src.data.Sex) and set(src.data.Sex.unique()) == {"Female", "Male"}
    assert all(close(can_cov[k], src.data[k]) for k in ["Age", "MeanFD"])

    workbook = load_workbook(src.paths["review_xlsx"], data_only=True, read_only=True)
    try:
        sheets = workbook.sheetnames
        assert {"Brain_Behaviour_Master", "Imaging_Similarity", "Questionnaire_Items", "Subject_Crosswalk"} <= set(sheets)
    finally:
        workbook.close()
    qc.extend([
        dict(test="frozen_input_hashes", status="PASS", details="Final64 lock SHA-256, canonical SHA-256, historical R input MD5"),
        dict(test="crosswalk_and_covariates", status="PASS", n=64, details="Unique explicit IDs; 32 HC, 32 SZ; Age/Sex/posttrim MeanFD all complete"),
        dict(test="review_workbook", status="PASS_REVIEW_COPY", details="; ".join(sheets)),
        dict(test="analysis_maps", status="READ_ONLY_CONTEXT",
             details=f"Clinical rows={len(src.read('clinical_map'))}; task rows={len(src.read('task_map'))}; task trials out of scope"),
    ])


def audit_imaging(src, out, qc):
    a, b = np.load(src.paths["bins"]), src.current_bins
    assert (a == 10).sum() == 261 and (b == 10).sum() == 285
    cross = pd.crosstab(pd.Series(a, name="Definition_A_bin"), pd.Series(b, name="G1_bin_current"))
    write_csv(cross.reset_index(), out / "00_BIN_DEFINITION_CROSSWALK.csv")
    qc.append(dict(test="bin_definition_conflict", status="DEFINITION_CONFLICT_ISOLATED", n=2608,
                   details=f"Same G1 and voxel order; different bins for {(a != b).sum()} rows; "
                           f"Bin 10 intersection={((a == 10) & (b == 10)).sum()}; "
                           "locked similarity uses G1_bin_current; canonical FC long/overall slopes use Definition A"))
    assert src.mapping.reward_stage.value_counts().to_dict() == {"Anticipation": 43, "Outcome": 19, "Not_assigned": 3}
    assert set(src.mapping.loc[src.mapping.reward_stage.eq("Not_assigned"), "roi_name"]) == {
        "VTA_bi", "SNr_bi", "VP_VeP_bi"}
    assert src.target_fc.Reference_G1_Version.eq(G1_VERSION).all()
    assert len(src.target_fc) == 41600
    stage = src.read("fc")
    assert len(stage) == 1280 and stage.Reference_G1_Version.eq(G1_VERSION).all()
    derived = src.read("derived").pivot(index=ID, columns="Metric_Name", values="Value").reindex(src.data.index)
    can_metrics = {
        ANT: "Similarity (Stage=Anticipation)", OUTCOME: "Similarity (Stage=Outcome)",
        "Anticipation_slope": "Overall_Anticipation_FC_slope", "Outcome_slope": "Overall_Outcome_FC_slope",
        "SMA_slope": "SMA_Anticipation_rawFC_slope", "IFG_slope": "IFG_Outcome_rawFC_slope",
        "Cingulate_slope": "Cingulate_Outcome_rawFC_slope",
    }
    for metric, frozen_metric in can_metrics.items():
        if metric not in (ANT, OUTCOME):
            assert close(derived[frozen_metric], src.data[metric]), metric
    old_all_bins = src.read("prior_all_bin_similarity")
    assert len(old_all_bins) == 1280
    for stage_name in ("Anticipation", "Outcome"):
        sim = src.similarity_long.query("Stage == @stage_name").set_index(ID, verify_integrity=True).reindex(src.data.index)
        assert len(sim) == 64 and sim.Bin.eq(10).all()
        assert sim.Group.equals(src.data.Group)
        assert sim.Reference_Pattern_ID.eq(np.where(sim.Group.eq("HC"), "HC_LOO", "HC32")).all()
        mask = src.stage_mask(stage_name)
        assert sim.N_Target_ROIs.eq(mask.sum()).all()
        assert close(sim.Similarity, src.data[f"{stage_name}_similarity"])
        assert close(fisher(sim.Similarity_Raw_r), sim.Similarity)
        reconstructed = similarity(src.current_profiles[..., mask], src.templates[..., mask])
        wrong_a = src.profiles[..., mask]
        is_hc = src.data.Group.eq("HC").to_numpy()
        a_sum = wrong_a[is_hc].sum(axis=0)
        a_templates = np.broadcast_to(a_sum/32, wrong_a.shape).copy()
        a_templates[is_hc] = (a_sum-wrong_a[is_hc])/31
        a_r = similarity(wrong_a[:, -1], a_templates[:, -1])
        a_err = np.max(np.abs(a_r - sim.Similarity_Raw_r))
        qc.append(dict(test=f"wrong_bin_definition_{stage_name}", status="ISOLATED", n=64,
                       max_abs_diff_r=a_err, max_abs_diff_Fisher_z=np.max(np.abs(fisher(a_r)-sim.Similarity)),
                       details="Diagnostic only; Definition-A FC cannot reproduce the frozen similarity."))
        error_r = np.max(np.abs(reconstructed[:, -1] - sim.Similarity_Raw_r))
        error_z = np.max(np.abs(fisher(reconstructed[:, -1]) - sim.Similarity))
        assert max(error_r, error_z) < SIM_TOL
        old = old_all_bins.query("stage == @stage_name").pivot(index="subject", columns="bin", values="profile_r")
        assert close(reconstructed, old.reindex(src.data.index), SIM_TOL)
        mean_fc = src.profiles[..., mask].mean(axis=-1)
        saved = stage.query("Stage == @stage_name").pivot(index=ID, columns="Bin", values="FC").reindex(src.data.index)
        assert close(mean_fc, saved, SIM_TOL)
        centered_bins = np.arange(1, 11) - 5.5
        slope = mean_fc @ centered_bins / (centered_bins @ centered_bins)
        assert close(slope, src.data[f"{stage_name}_slope"], SIM_TOL)
        qc.append(dict(test=f"reconstruction_{stage_name}", status="PASS", n=64,
                       max_abs_diff_r=error_r, max_abs_diff_Fisher_z=error_z,
                       details=f"G1_bin_current Bin 10 uses 285 voxels and {mask.sum()} targets; HC31-LOO and SZ HC32. Definition-A stage FC and slope separately verified"))
    return can_metrics


def audit_scoring(src, qc):
    items = src.read("items")
    items = items[items[ID].isin(src.data.index) & items.Questionnaire.isin(["TEPS", "ACIPS"])].copy()
    assert not items.duplicated([ID, "Questionnaire", "Item_ID"]).any()
    assert items.Raw_Value.dropna().between(1, 6).all()
    assert np.all(np.equal(np.mod(items.Raw_Value.dropna(), 1), 0))
    reverse = items.Item_ID.isin(["TEPS13", "ACIPS3"]) & items.Raw_Value.notna()
    assert close(items.loc[reverse, "Scored_Value"], 7 - items.loc[reverse, "Raw_Value"])
    assert items.loc[items.Item_ID.eq("TEPS13") & items.Raw_Value.notna(), "Reverse_Scored"].eq("YES_FOR_TEPS_TOTAL_ONLY").all()
    assert items.loc[items.Item_ID.eq("ACIPS3") & items.Raw_Value.notna(), "Reverse_Scored"].eq("YES").all()

    scores = items.pivot(index=ID, columns="Item_ID", values="Raw_Value").reindex(src.data.index)
    scale_rows = []
    for name, key in KEYS.items():
        scale = name.split("_")[0]
        values = scores[[f"{scale}{i}" for i in key]].copy()
        if name in REVERSE:
            col = f"{scale}{REVERSE[name]}"
            values[col] = 7 - values[col]
        calc = values.sum(axis=1, min_count=len(key))
        assert close(calc, src.data[name]), f"Local scoring does not reproduce frozen {name}"
        qc.append(dict(test="item_score_reproduction", scale=name, status="PASS", n=calc.notna().sum(),
                       details="Strict complete-item sums, identical scores and missingness"))
        for group in ("HC", "SZ", "ALL"):
            index = src.data.index if group == "ALL" else src.data.index[src.data.Group.eq(group)]
            d = calc.loc[index]
            valid = d.dropna()
            lo, hi = len(key), 6 * len(key)
            matrix = values.loc[valid.index]
            alpha = len(key) / (len(key) - 1) * (1 - matrix.var().sum() / valid.var())
            scale_rows.append(dict(record_type="distribution", scale=name, population=group,
                                   n_total=len(d), n_missing=d.isna().sum(), missing_pct=100*d.isna().mean(),
                                   missing_IDs=";".join(d.index[d.isna()]), **distribution(valid),
                                   variance=valid.var(), unique_scores=valid.nunique(),
                                   theoretical_min=lo, theoretical_max=hi,
                                   floor_n=valid.eq(lo).sum(), ceiling_n=valid.eq(hi).sum(),
                                   floor_pct=100*valid.eq(lo).mean(), ceiling_pct=100*valid.eq(hi).mean(),
                                   span_pct=100*(valid.max()-valid.min())/(hi-lo),
                                   sd_pct_of_possible_span=100*valid.std()/(hi-lo),
                                   alpha=alpha, item_n=len(key), scored_items=";".join(map(str, key)),
                                   reverse_item=REVERSE.get(name), score_direction="higher endorsement"))
    reliab = src.read("reliability").set_index("variable")
    historical_info = src.read("historical_scale_info").set_index(["variable", "population"])
    for r in scale_rows:
        row = historical_info.loc[(r["scale"], r["population"])]
        assert (r["n"], r["n_missing"]) == (row.n_valid, row.n_missing)
        assert close([r["mean"], r["sd"], r["floor_pct"], r["ceiling_pct"]],
                     [row["mean"], row.sd, row.floor_pct, row.ceiling_pct])
        if r["population"] == "ALL":
            assert close(r["alpha"], reliab.at[r["scale"], "alpha"])
    return pd.DataFrame(scale_rows)


def audit_historical_models(src, qc):
    within = src.read("within")
    orig = src.read("eligible")
    assert len(orig) == 119 and within.population.value_counts().to_dict() == {"SZ": 119, "HC": 70}
    sz_old = within.query("population == 'SZ'").copy()
    assert sz_old.scale_variable.nunique() == 17 and sz_old.imaging_variable.nunique() == 7
    check = []
    for row in sz_old.itertuples():
        scale, image = row.scale_variable, row.imaging_variable
        is_ols = row.scale_population == "BOTH"
        d = cases(src.sz, scale, image, COVARIATES if is_ols else ())
        if is_ols:
            refit = effect(model(d, scale, image))
            assert close([refit["se"], refit["ci_low"], refit["ci_high"]], [row.se, row.ci_low, row.ci_high])
        elif len(d) >= 8 and d[scale].nunique() >= 3 and d[image].nunique() >= 3:
            test = stats.spearmanr(d[scale], d[image])
            refit = dict(n=len(d), beta=test.statistic, p=test.pvalue)
        else:
            refit = dict(n=len(d), beta=np.nan, p=np.nan)
        assert refit["n"] == row.n and close(refit["beta"], row.effect) and close(refit["p"], row.p), (scale, image, refit, row)
        check.append(dict(test="historical_SZ_119_refit", status="PASS", scale=scale, outcome=image,
                          method=row.method, population="SZ", **refit, saved_beta=row.effect,
                          saved_ci_low=row.ci_low, saved_ci_high=row.ci_high, saved_p=row.p,
                          saved_q=row.p_adj_bh_within_population))
    q = bh([r["p"] for r in check])
    assert close(q, [r["saved_q"] for r in check])
    for row, q_new in zip(check, q):
        row["q"] = q_new
    qc.extend(check)

    interactions = src.read("interaction")
    assert len(interactions) == 70 and close(bh(interactions.p), interactions.p_adj_bh_70_interactions)
    ppt = src.read("ppt_points")
    ppt_stats = src.read("ppt_stats")
    ppt_bands = src.read("ppt_bands")
    for scale in CORE:
        for group in ["HC", "HC+SZ"]:
            inter = group == "HC+SZ"
            data = src.data if inter else src.data[src.data.Group.eq(group)]
            d = cases(data, scale, ANT)
            old = (interactions.query("scale_variable == @scale and imaging_variable == @ANT") if inter
                   else within.query("scale_variable == @scale and imaging_variable == @ANT and population == 'HC'")).iloc[0]
            res = effect(model(d, scale, ANT, interaction=inter),
                         "Group_SZ_x_scale" if inter else "TEPS_or_ACIPS")
            assert res["n"] == old.n and close(list(res.values())[1:],
                                              old[["effect", "se", "ci_low", "ci_high", "p"]].to_numpy(float))
            saved_q = old["p_adj_bh_70_interactions" if inter else "p_adj_bh_within_population"]
            qc.append(dict(test="historical_interaction_refit" if inter else "historical_HC_refit",
                           status="PASS", scale=scale, outcome=ANT, population=group,
                           method=old.method, **res, saved_q=saved_q, saved_beta=old.effect, saved_p=old.p,
                           ids=";".join(d.index)))

        d = cases(src.sz, scale, ANT)
        cov = covariate_frame(d)
        xr, yr = residual(z(d[scale]), cov), residual(z(d[ANT]), cov)
        p = ppt[ppt.scale_variable.eq(scale)].set_index(ID).reindex(d.index)
        assert set(d.index) == set(ppt.loc[ppt.scale_variable.eq(scale), ID])
        assert close(xr, p.scale_residual) and close(yr, p.imaging_residual)
        assert close(d[scale], p.scale_value) and close(d[ANT], p.imaging_value)
        assert close(z(d[scale]), p.scale_z) and close(z(d[ANT]), p.imaging_z)
        partial = fit(yr, pd.DataFrame({"Intercept": 1., "TEPS_or_ACIPS": xr}))
        res = effect(partial)
        full = effect(model(d, scale, ANT))
        assert close(full["beta"], res["beta"])
        grid = ppt_bands[ppt_bands.scale_variable.eq(scale)]
        des = np.column_stack([np.ones(len(grid)), grid.scale_residual])
        vc = partial.get_robustcov_results(cov_type="HC3").cov_params()
        means = des @ partial.params.to_numpy()
        half_width = 1.96*np.sqrt(np.einsum("bi,ij,bj->b", des, vc, des))
        assert close(grid.predicted, means) and close(grid.ci_low, means-half_width) and close(grid.ci_high, means+half_width)
        saved = ppt_stats[ppt_stats.scale_variable.eq(scale)].iloc[0]
        assert close([full["beta"], full["ci_low"], full["ci_high"], full["p"]],
                     saved[["effect", "ci_low", "ci_high", "p"]].to_numpy(float))
        qc.extend([
            dict(test="PPT_points_and_main_statistics", status="PASS", scale=scale, outcome=ANT, population="SZ",
                 **full, q=saved.p_adj_bh_within_population, ids=";".join(d.index),
                 residual_sd_x=np.std(xr, ddof=1), residual_sd_y=np.std(yr, ddof=1),
                 details="Residuals of complete-case z scores; no standardization after residualizing"),
            dict(test="PPT_band_inference", status="INTERPRETATION_WARNING", scale=scale, outcome=ANT,
                 method="HC3 on two-column added-variable regression", **res, saved_se=full["se"],
                 details="Slope agrees. Original ribbon uses residual-only leverage, distinct from full-model HC3 leverage."),
        ])
    return pd.DataFrame(check)


def dictionary(src, can_metrics):
    fields = []
    clinical_dict = src.read("clinical_dict").set_index("Variable_Name")
    for score, key in KEYS.items():
        r = src.registry.loc[score]
        fields.append(dict(variable=score, scope="frozen", unit="score points", source_field=r.source_col,
                           source=src.paths["master"].as_posix(),
                           definition=f"Sum of items {', '.join(map(str, key))} with complete items only",
                           direction="Higher pleasure endorsement",
                           theoretical_range=f"{len(key)} to {6*len(key)}",
                           reverse=f"Item {REVERSE[score]}: 7 - raw" if score in REVERSE else "none",
                           audit_note=clinical_dict.at[r.source_col, "Definition_Status"]))
    for legacy in ("ACIPS1", "ACIPS2", "ACIPS3"):
        fields.append(dict(variable=legacy, scope="legacy_wide_field", source=src.paths["clinical_wide"].as_posix(),
                           source_field=legacy, definition="Unverified historical summary; separate from item IDs in long table",
                           audit_note="Excluded from inference per lock dictionary"))
    for variable, v in can_metrics.items():
        is_sim = variable.endswith("_similarity")
        fields.append(dict(variable=variable, scope="frozen", unit="Fisher z of profile r" if is_sim else "Fisher-z FC per bin",
                           source_field=v, source=src.paths["sim" if is_sim else "derived"].as_posix(),
                           definition=("G1_bin_current Bin 10 (285 voxels) Pearson r of stage target vectors against HC template, followed by atanh"
                                       if is_sim else "OLS across all 10 Definition-A bins of raw mean FC on Bin - 5.5")))
    extra = [
        ("Anticipation_raw_similarity_r", "frozen", "Pearson r", "Similarity_Raw_r", "Across 43 Bin 10 target FC values"),
        ("Outcome_raw_similarity_r", "frozen", "Pearson r", "Similarity_Raw_r", "Across 19 Bin 10 target FC values"),
        ("Raw_FC", "frozen", "Fisher-z FC", "FC", "Mean of voxelwise Fisher-z rsFC in Definition-A bins"),
        ("Current_bin_Raw_FC", "reconstructed formal input", "Fisher-z FC", "", "Mean of frozen voxelwise Fisher-z rsFC in G1_bin_current bins"),
        ("Age", "frozen", "years", "Age", "Age at scan"),
        ("Sex", "frozen", "category", "Sex", "Female reference level in regression"),
        ("MeanFD", "frozen", "mm", "MeanFD", "Resting-state post-trim mean framewise displacement"),
        ("*_raw_residual", "follow-up", "original units", "", "Residual from raw complete-case value on Age, Sex and MeanFD"),
        ("*_PPT_residual", "historical recomputation", "original SD units", "", "Residual from z(raw) on Age, Sex and MeanFD, with no second scaling"),
        ("*_standardized_residual", "follow-up", "residual SD units", "", "Sample z of residual after covariate adjustment"),
        ("HC_relative_similarity_z", "descriptive", "HC sample SD", "", "(raw Pearson r - HC mean raw r) / HC sample SD"),
        ("absolute_HC_deviation", "descriptive", "HC sample SD", "", "Absolute HC-relative z; the sign is retained separately"),
        ("HC_relative_Fisher_similarity_z", "descriptive", "HC sample SD", "", "Same contrast using unresidualized frozen Fisher-z similarity"),
        ("exploratory_HC31_mean_r", "exploratory", "Pearson r", "", "Mean SZ r over 32 held-out-HC templates of size 31; HC keeps its own LOO r"),
        ("exploratory_shared_HC16", "exploratory", "Pearson r", "", "Per split both evaluation groups use the same independent 16-HC template"),
    ]
    for variable, scope, unit, field, definition in extra:
        role = {"Raw_FC": "fc", "Current_bin_Raw_FC": "current_bins_2847"}.get(
            variable, "sample" if variable in COVARIATES else "sim")
        fields.append(dict(variable=variable, scope=scope, unit=unit, source_field=field,
                           source=src.paths[role].as_posix(), definition=definition))
    for variable in (*CLINICAL, *CLINICAL_HISTORY):
        fields.append(dict(variable=variable, scope="frozen", source=src.paths["master"].as_posix(),
                           source_field=f"final64_v3__{variable}", unit="years" if variable in ("illness_duration","DUP","age_at_onset") else
                           "text" if variable == "medication_text" else "score points",
                           definition="As stored in the locked clinical table"))
    return pd.DataFrame(fields)


def run_audit(src, out):
    qc = []
    audit_sources(src, out, qc)
    can_metrics = audit_imaging(src, out, qc)
    scale_info = audit_scoring(src, qc)
    historical = audit_historical_models(src, qc)
    write_csv(pd.DataFrame(qc), out / "00_SAMPLE_AND_MODEL_QC.csv")
    write_csv(dictionary(src, can_metrics), out / "00_VARIABLE_DICTIONARY.csv")
    write_csv(scale_info, out / "00_SCALE_SCORING_QC.csv")
    focus = historical[historical.scale.isin(CORE) & historical.outcome.eq(ANT)]
    model_rows = "\n".join(f"| {r.scale} | {r.n} | {r.beta:.6f} | [{r.ci_low:.6f}, {r.ci_high:.6f}] | {r.p:.8g} | {r.q:.8f} |"
                           for r in focus.itertuples())
    check = pd.DataFrame(qc)
    ppt = check[check.test.eq("PPT_band_inference")]
    band_rows = "\n".join(f"| {r.scale} | {r.beta:.6f} | {r.se:.6f} | {r.saved_se:.6f} |"
                          for r in ppt.itertuples())
    sex_rows = ", ".join(f"{g} {sex}={n}" for (g,sex),n in src.data.groupby(["Group","Sex"]).size().items())
    ids = "; ".join(cases(src.sz, TEPS_C, ANT).index)
    missing = "; ".join(src.sz.index[src.sz[TEPS_C].isna()])
    distributions = scale_info[scale_info.population.ne("ALL")]
    scale_rows = "\n".join(f"| {r.scale} | {r.population} | {r.n}/32 | {r.min:.0f}–{r.max:.0f} | {r.mean:.2f} ± {r.sd:.2f} | {r.floor_n}/{r.ceiling_n} |"
                           for r in distributions.itertuples())
    overlap = src.sz[[TEPS_A, TEPS_C, TEPS_T]].dropna()
    overlap_r = overlap[TEPS_C].corr(overlap[TEPS_T])
    overlap_rho = overlap[TEPS_C].corr(overlap[TEPS_T], method="spearman")
    text = f"""# Final64 数据与定义审计

## 冻结来源与关联单位

- 正式版本为 Final64_AnalysisReady_v1.0，位置为 {src.lock.as_posix()}。以冻结 TSV 的 Subject_ID 合并样本、影像、逐题与临床记录。70 人登记表中只有冻结的 64 人进入分析。HC、SZ 各 32 名。年龄、性别及扫描后修剪的静息态 MeanFD 均完整。性别计数为 {sex_rows}。
- 审阅工作簿为 {src.paths['review_xlsx'].as_posix()}。正式计算使用锁定主表、分析样本矩阵、Similarity/FC 长表、衍生指标、逐题表及 Crosswalk。已核查两个关联地图。00_INPUT_PROVENANCE.csv 提供实际绝对路径及哈希。锁定数据、375 人固定 G1、Definition-A bin 数组及靶区映射的哈希均与相应清单一致。
- 65 个固定靶区包括 43 个 Anticipation、19 个 Outcome，以及未归类的 VTA_bi、SNr_bi、VP_VeP_bi。固定 G1 的 2,608 个纹状体体素映射到原 10 个 bins。低 G1 富含腹侧纹状体和尾状核，高 G1 富含壳核。既有审计注明最早空间掩膜的原件缺失，冻结衍生数组保留精确重建的行映射。

## 重要分箱冲突

锁库 FIXED_G1_REFERENCE_AUDIT 将阶段斜率和 Similarity 并列描述在同一套分箱下。冻结 FC 长表和整体/区域 slope 用 Definition A（Bin 10 为 261 体素）。正式 Similarity 使用从 2,847 体素表映射至同一 2,608 行的 G1_bin_current（Bin 10 为 285 体素）。两套都使用逐值相同的连续 G1 与相同体素顺序。529 行 bin 标签不同，Bin 10 共同体素 261 个。原始模板和所有 128 个冻结阶段 Similarity 从 285 体素路径精确复现，数值核对见 QC 表。交叉表与隔离测试见 00_BIN_DEFINITION_CROSSWALK.csv 和 00_SAMPLE_AND_MODEL_QC.csv。Phase 2 使用与 Similarity 对应的原始 FC 轮廓，引用已存 slope 时明示 Definition A。

## FC 与相似性量纲

1. **原始 FC。** 从静息态时序的相关系数做 Fisher 变换，再按固定 Bin 对体素求均值，得到每名被试的 Bin × ROI FC。阶段平均 FC 取所属靶区算术平均。冻结的整体阶段 slope 使用 Definition A 的 10 个 bins 上的平均 FC 对 Bin − 5.5 拟合。
2. **原始相似性 r。** G1_bin_current Bin 10 的 43 个 Anticipation 靶区 FC 与 HC 模板逐目标作 Pearson 相关。Outcome 同理，使用 19 个靶区。HC 模板为其余 31 个 HC 的逐特征均值，SZ 模板为 HC32 均值。
3. **正式未调整相似性。** 对轮廓 Pearson r 截取至 ±0.999999 后作 atanh，得到冻结表的 Similarity。对两阶段 64 人重建了 r、Fisher z，逐值核对正式表。其他 bins 的重建也与既有阶段特异性源表一致。旧的 65 靶区轮廓相似性与此端点分别存档。
4. **协变量残差。** 原单位残差等于实际分数减去年龄、性别、MeanFD 模型预测。PPT 坐标先在有效样本内对各轴原值做样本 SD 标准化，再对标准化值分别回归协变量并取残差。原 PPT 没有对残差再次除以残差 SD。逐例复算坐标与历史图数据一致。本轮另保留二次标准化的残差，用于明确量纲。
5. **HC 相对 z。** 将 SZ 的未调整 Pearson r 减 HC32 个 LOO r 的均值，再除以该 32 人的样本 SD。另报告未调整 Fisher z 的同类描述性指标。此处正值表示数值高于 HC 样本均值。匹配模板敏感性在 Phase 1 单列。

## 冻结本地计分

- TEPS 预期性取 9 题，消费性取 10 题。20 题总分等于预期性 + 消费性 + (7 − 原始第 13 题)，第 13 题只在总分反向。四个抽象/具体维度、题号及范围见变量字典。较高分表示更认可相应愉悦体验。
- ACIPS 预期性取 7 题，第 3 题以 7 − 原始分计入。消费性取 10 题，直接加总。17 题总分为两分量表之和。较高分表示更认可人际愉悦体验。宽表同名的旧字段 ACIPS1/2/3 缺乏已确认的定义，保留原记录。长表的 ACIPS1–17 为逐题 ID。
- 逐题重算十个正式分数。分值和缺失状态均逐例复现，无插补。旧字典的总分定义状态字段仍标待核，但记录了运算公式，正式 registry 已纳入总分，且与逐题完全相符。论文引用版本仍应以本地实际发放题干和锚点核实。中文版 TEPS 20 题和 ACIPS 的构念背景分别见 [Chan 等，2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3329425/) 和 [中文版 ACIPS 验证研究](https://onlinelibrary.wiley.com/doi/full/10.1002/pchj.139)。
- 两个核心模型的 SZ 完整样本一致。有效 ID 为 {ids}。缺失为 {missing}，均为量表缺失。
- 28 名完整 SZ 中 TEPS 消费性与总分 Pearson r={overlap_r:.3f}、Spearman ρ={overlap_rho:.3f}，存在计分嵌套。

| 冻结分数 | 组别 | 有效 N | 观察范围 | 均值 ± SD | 理论地板/天花板人数 |
| --- | --- | --- | --- | --- | --- |
{scale_rows}

## 历史统计复核

完整病例内分别以样本 SD 标准化分数、Fisher-z 相似性、年龄及 MeanFD，性别采用分类变量（Female 为参考），OLS 使用 HC3 协方差、正态尾双侧 p 和 β ± 1.96×SE。SZ 校正家族为 17 个端点 × 7 个影像表型，共 119 项，包含 70 个愉悦量表 HC3 检验和 49 个仅 SZ 临床/运动 Spearman 检验。重新计算全部历史 p 和 BH q 以验证原家族，两个核心 β、CI、p、q 与表一致。HC 另有 70 项家族，合并样本也有独立 119 项家族。原 49 个 Spearman 历史 CI 作为原表记录，未重复抽样。

| 核心关联 | SZ N | β | 95% CI | 双侧 p | 历史 BH q（119） |
| --- | --- | --- | --- | --- | --- |
{model_rows}

70 个 Group×量表交互的 BH 列重核通过。两项核心交互从冻结数据重拟合，结果见 QC 表。参与者级残差和画图模型亦逐值复现。

## 历史残差图推断说明

历史图文字报告的 β 与 CI 来自完整协变量模型。其拟合线斜率相同，但阴影带对两列残差回归另估 HC3，使用的杠杆值与完整模型不同。它对应残差回归线的均值带。作为完整模型 HC3 不确定性读取时会产生误差。新残差图保留原坐标及斜率，直接显示完整模型 CI 并省略原阴影带。

| 量表 | 残差图斜率 | 残差回归 HC3 SE | 正式完整模型 HC3 SE |
| --- | --- | --- | --- |
{band_rows}
"""
    (out / "00_DATA_AND_DEFINITION_AUDIT.md").write_text(text, encoding="utf-8")
    return scale_info
