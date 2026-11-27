"""Read-only inputs and numerical routines for the Final64 follow-up."""
from functools import cached_property
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ID = "Subject_ID"
ANT = "Anticipation_similarity"
OUTCOME = "Outcome_similarity"
TEPS_A = "TEPS_anticipatory"
TEPS_C = "TEPS_consummatory"
TEPS_T = "TEPS_total"
ACIPS_A = "ACIPS_anticipatory"
ACIPS_C = "ACIPS_consummatory"
ACIPS_T = "ACIPS_total"
CORE = (TEPS_C, TEPS_T)
STAGE_SCALES = (TEPS_A, TEPS_C, TEPS_T, ACIPS_A, ACIPS_C, ACIPS_T)
COVARIATES = ("Age", "Sex", "MeanFD")
CLINICAL = ("PANSS_positive", "PANSS_negative", "PANSS_general", "PANSS_total", "SANS", "SAS", "BARS")
CLINICAL_HISTORY = ("illness_duration", "age_at_onset", "DUP", "medication_text")
SEED = 20261111
BOOT_N = 20000
PERM_N = 49999
SPLIT_N = 512
SIM_TOL = 1e-7
CURRENT_BINS = "G1_bin_current_mapped_2847_to_2608"
G1_VERSION = "fixed_375_person_reference_G1_10bins_2608voxels"
KEYS = {
    "TEPS_abstract_anticipatory": (4, 6, 18, 20),
    "TEPS_concrete_anticipatory": (1, 5, 8, 10, 11),
    "TEPS_abstract_consummatory": (2, 7, 3, 9, 14, 19),
    "TEPS_concrete_consummatory": (12, 15, 16, 17),
    TEPS_A: (4, 6, 18, 20, 1, 5, 8, 10, 11),
    TEPS_C: (2, 7, 3, 9, 14, 19, 12, 15, 16, 17),
    TEPS_T: tuple(range(1, 21)),
    ACIPS_A: (1, 3, 7, 8, 10, 14, 15),
    ACIPS_C: (2, 4, 5, 6, 9, 11, 12, 13, 16, 17),
    ACIPS_T: tuple(range(1, 18)),
}
REVERSE = {TEPS_T: 13, ACIPS_A: 3, ACIPS_T: 3}


def read_tsv(path, **kwargs):
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig", **kwargs)


def write_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def close(a, b, tol=1e-11):
    return np.allclose(a, b, atol=tol, rtol=0, equal_nan=True)


def z(x, axis=0):
    x = np.asarray(x, dtype=float)
    return (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, ddof=1, keepdims=True)


def center_unit(x):
    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.linalg.norm(centered, axis=-1, keepdims=True)


def similarity(x, template):
    return np.sum(center_unit(x) * center_unit(template), axis=-1)


def fisher(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def bh(p):
    p = np.asarray(p, dtype=float)
    keep = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if keep.any():
        q[keep] = multipletests(p[keep], method="fdr_bh")[1]
    return q


def covariate_frame(d, terms=COVARIATES, standardize=True):
    c = pd.DataFrame({"Intercept": np.ones(len(d))}, index=d.index)
    for term in terms:
        if term == "Sex":
            if d[term].nunique() == 2:
                c["Sex_Male_vs_Female"] = d[term].eq("Male").astype(float)
        else:
            c[term] = z(d[term]) if standardize else d[term].astype(float)
    return c


def model(d, score, outcome, terms=COVARIATES, standardized=True, interaction=False):
    design = covariate_frame(d, terms, standardize=standardized)
    design["TEPS_or_ACIPS"] = z(d[score]) if standardized else d[score].astype(float)
    if interaction:
        design["Group_SZ"] = d.Group.eq("SZ").astype(float)
        design["Group_SZ_x_scale"] = design.Group_SZ * design.TEPS_or_ACIPS
    y = z(d[outcome]) if standardized else d[outcome].to_numpy(dtype=float)
    return fit(y, design)


def fit(y, design):
    if not np.isfinite(design).all().all() or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f"Invalid or rank-deficient design {list(design.columns)}")
    return sm.OLS(y, design).fit()


def effect(model_fit, term="TEPS_or_ACIPS", use_t=False):
    hc3 = model_fit.get_robustcov_results(cov_type="HC3", use_t=use_t)
    index = model_fit.model.exog_names.index(term)
    beta, se = hc3.params[index], hc3.bse[index]
    # The historical R script uses the literal 1.96 and a normal tail.
    critical = stats.t.ppf(.975, hc3.df_resid) if use_t else 1.96
    return dict(n=int(hc3.nobs), beta=beta, se=se, ci_low=beta - critical * se,
                ci_high=beta + critical * se, p=hc3.pvalues[index])


def residual(y, design):
    a = np.asarray(design)
    return np.asarray(y) - a @ np.linalg.lstsq(a, y, rcond=None)[0]


def bootstrap_betas(d, xs, ys, indices):
    """Standardized OLS beta for each resampled complete participant set."""
    nuisance = covariate_frame(d).to_numpy()[indices]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = z(d[list(xs)].to_numpy(float)[indices], axis=1)
        y = z(d[list(ys)].to_numpy(float)[indices], axis=1)
    design = np.concatenate((np.repeat(nuisance[:, None], len(xs), axis=1),
                             np.swapaxes(x, 1, 2)[..., None]), axis=-1)
    transpose = np.swapaxes(design, -1, -2)
    rhs = transpose @ y[:, None]
    xtx = transpose @ design
    valid = np.isfinite(rhs).all(axis=(-1, -2)) & np.isfinite(xtx).all(axis=(-1, -2))
    valid[valid] = np.linalg.matrix_rank(xtx[valid], hermitian=True) == design.shape[-1]
    beta = np.full((len(indices), len(xs), len(ys)), np.nan)
    beta[valid] = np.linalg.solve(xtx[valid], rhs[valid])[:, -1, :]
    return beta


def percentile(x, probabilities=(.025, .975)):
    values = np.asarray(x)
    values = values[np.isfinite(values)]
    return np.quantile(values, probabilities) if len(values) else np.full(len(probabilities), np.nan)


def distribution(x):
    x = pd.Series(x).dropna().astype(float)
    q = percentile(x, (.025, .1, .25, .5, .75, .9, .975))
    return dict(n=len(x), mean=x.mean(), sd=x.std(), min=x.min(), p025=q[0], p10=q[1],
                p25=q[2], median=q[3], p75=q[4], p90=q[5], p975=q[6], max=x.max())


class Inputs:
    def __init__(self, analysis_root, project_root):
        self.root = Path(analysis_root).resolve(strict=True)
        self.project = Path(project_root).resolve(strict=True)
        self.lock = self.root / "00_FINAL64_DATABASE_LOCK"
        self.frozen = self.lock / "frozen_data"
        self.hist = self.root / "03_FINAL64_FULL_SCALE_REANALYSIS"
        self.tables = self.hist / "tables"
        self.review = self.root.parent / "02" / "subject_level_data_export"
        self.canonical = self.project / "CURRENT64_CANONICAL_RERUN"
        self.can_frozen = self.canonical / "01_FROZEN_INPUTS"
        self.paths = {
            "manifest": self.lock / "manifests/FROZEN_DATA_MANIFEST.tsv",
            "sample": self.lock / "analysis_preparation/ANALYSIS_SAMPLE_MATRIX.tsv",
            "master": self.frozen / "subject_level_brain_behaviour_master_final64.tsv",
            "crosswalk": self.frozen / "SUBJECT_CROSSWALK.tsv",
            "clinical_wide": self.frozen / "subject_level_clinical_behaviour_wide.tsv",
            "items": self.frozen / "subject_level_questionnaire_items_long.tsv",
            "sim": self.frozen / "subject_level_similarity_long.tsv",
            "fc": self.frozen / "subject_level_fc_long.tsv",
            "roi": self.frozen / "subject_level_fc_target_roi_long.tsv",
            "derived": self.frozen / "subject_level_derived_imaging_metrics.tsv",
            "clinical_dict": self.frozen / "CLINICAL_BEHAVIOUR_DATA_DICTIONARY.tsv",
            "export_qc": self.frozen / "DATA_EXPORT_QC_ISSUES.tsv",
            "review_xlsx": self.review / "SUBJECT_LEVEL_BRAIN_BEHAVIOUR_DATABASE.xlsx",
            "clinical_map": self.root / "02_FINAL64_BRAIN_BEHAVIOUR_MAP/IMAGING_CLINICAL_ASSOCIATION_MAP.tsv",
            "task_map": self.root / "02_FINAL64_BRAIN_BEHAVIOUR_MAP/IMAGING_TASK_BEHAVIOUR_ASSOCIATION_MAP.tsv",
            "fixed_G1_audit": self.lock / "metric_audit/FIXED_G1_REFERENCE_AUDIT.md",
            "can_manifest": self.canonical / "00_README/INPUT_MANIFEST.tsv",
            "canonical_covariates": self.can_frozen / "covariates/current64_covariates.tsv",
            "g1": self.can_frozen / "canonical_G1/375_reference_G1_manuscript_2608.npy",
            "bins": self.can_frozen / "canonical_bins/375_reference_bins_manuscript_2608.npy",
            "current_bins_2847": self.project.parent.parent / "gradient_p0_inventory_20261008_173949/new/current_bins/g1_voxel_table_with_bins_CURRENT.tsv",
            "current_bins_2608_alias": self.project / "FIGURES_V4_REFERENCE_LOCKED/source_data/Figure1_3_G1_spatial_2608.tsv",
            "voxel_mapping": self.can_frozen / "voxel_order/row_mapping_2847_to_2608.npy",
            "historical_fingerprint_hashes": self.project / "SERVER_RECOVERED_CODE_REVIEW/FINAL_SCIENTIFIC_VALIDATION/RUN_ARTIFACTS/current_from_arrays/fc_input_hashes.tsv",
            "map": self.can_frozen / "reward_stage_mapping/reward_stage_mapping_65.tsv",
            "subfamily_map": self.can_frozen / "roi_subfamily_mapping/roi_subfamily_mapping_65.tsv",
            "registry": self.tables / "SCALE_ANALYSIS_REGISTRY.tsv",
            "eligible": self.tables / "ELIGIBLE_SCALE_IMAGING_ASSOCIATIONS.tsv",
            "within": self.tables / "ALL_SCALE_IMAGING_WITHIN_GROUP_AUDIT.tsv",
            "interaction": self.tables / "ALL_SCALE_IMAGING_GROUP_INTERACTION_AUDIT.tsv",
            "historical_groups": self.tables / "HC_SZ_IMAGING_GROUP_EFFECTS.tsv",
            "reliability": self.tables / "SCALE_INTERNAL_CONSISTENCY.tsv",
            "historical_scale_info": self.tables / "SCALE_INFORMATION_CONTENT.tsv",
            "ppt_points": self.tables / "MANUSCRIPT_FIGURE2_PARTICIPANT_DATA.tsv",
            "ppt_bands": self.tables / "MANUSCRIPT_FIGURE2_HC3_BANDS.tsv",
            "ppt_stats": self.tables / "MANUSCRIPT_FIGURE2_STATISTICS.tsv",
            "historical_provenance": self.tables / "RUN_INPUT_PROVENANCE.tsv",
            "historical_scoring_script": self.hist / "scripts/run_full_reanalysis.R",
            "historical_audit_script": self.hist / "scripts/audit_all_scale_imaging_groups.R",
            "historical_ppt_script": self.hist / "scripts/make_requested_manuscript_figures.R",
            "similarity_script": self.project / "SERVER_RECOVERED_CODE_REVIEW/FINAL_SCIENTIFIC_VALIDATION/run_stage3_validation.py",
            "prior_all_bin_similarity": self.project / "SERVER_RECOVERED_CODE_REVIEW/FINAL_SCIENTIFIC_VALIDATION/RUN_ARTIFACTS/current_from_arrays/stage_specific_subject_bin_metrics.tsv",
            "historical_contribution": self.project / "FINAL_64SUBJECT_RESULTS_PACKAGE/tables/25_03_Bin10_subfamily_drivers_FINAL.tsv",
            "historical_raw_fc_roi": self.project / "THREE_REGION_RAW_FC_VALIDATION_20261028/tables/three_region_rawFC_validation.tsv",
            "roi_map_3": self.project / "THREE_REGION_RAW_FC_VALIDATION_20261028/tables/three_region_ROI_mapping.tsv",
            "historical_individual_contribution": self.project.parent.parent / "SMA_IFG_FORENSIC_BUNDLE_20260915_142030/SMA_IFG_FORENSIC_BUNDLE_20260915_142030/target_dir/24_normative_G1_FC_deviation_FIXED/Bin10_driver_decomposition_FINAL/04_Bin10_subject_ROI_contributions.tsv",
            "scoring_provenance": self.root.parent / "final64_gradient_clinical_stats/01_source_audit/scale_item_provenance.tsv",
            "prior_precision_audit": self.project / "clinical_extension_analysis/measurement_precision_audit/MEASUREMENT_PRECISION_AUDIT_REPORT.md",
            "medication_audit": self.project / "clinical_extension_analysis/medication_audit.csv",
            "clinical_audit": self.project / "clinical_extension_analysis/CLINICAL_EXTENSION_ANALYSIS_REPORT.md",
        }

    def read(self, role, **kwargs):
        return read_tsv(self.paths[role], **kwargs)

    @cached_property
    def registry(self):
        return self.read("registry").query("imaging_eligible").set_index("variable", drop=False)

    @cached_property
    def data(self):
        sample = self.read("sample").set_index(ID, verify_integrity=True).sort_index()
        master = self.read("master").set_index(ID, verify_integrity=True).reindex(sample.index)
        assert master.Group.equals(sample.Group)
        for variable, row in self.registry.iterrows():
            source = pd.to_numeric(master[row.source_col], errors="coerce")
            if variable in sample:
                assert close(sample[variable], source), variable
            sample[variable] = source
        for variable in CLINICAL_HISTORY:
            sample[variable] = master[f"final64_v3__{variable}"]
        return sample

    @cached_property
    def sz(self):
        return self.data[self.data.Group.eq("SZ")].copy()

    @cached_property
    def similarity_long(self):
        return self.read("sim")

    @cached_property
    def target_fc(self):
        return self.read("roi", usecols=[ID, "Group", "Bin", "Target_ROI", "Subfamily",
                                         "RewardStage", "FC", "Reference_G1_Version"])

    @cached_property
    def mapping(self):
        return self.read("map").sort_values("roi_index")

    @cached_property
    def profiles(self):
        fc = self.target_fc.pivot(index=[ID, "Bin"], columns="Target_ROI", values="FC")
        fc = fc.reindex(index=pd.MultiIndex.from_product([self.data.index, range(1, 11)]),
                        columns=self.mapping.roi_name)
        assert fc.notna().all().all()
        return fc.to_numpy().reshape(64, 10, 65)

    @cached_property
    def current_bins(self):
        table = self.read("current_bins_2847")
        row_map = np.load(self.paths["voxel_mapping"]).ravel()
        alias = self.read("current_bins_2608_alias")
        assert len(table) == 2847 and len(row_map) == len(alias) == 2608
        assert np.array_equal(alias.voxel_row_0based, np.arange(2608))
        assert np.array_equal(table[["i", "j", "k"]].to_numpy()[row_map], alias[["i", "j", "k"]])
        assert close(table.G1_current.to_numpy()[row_map], np.load(self.paths["g1"]))
        bins = table.G1_bin_current.to_numpy()[row_map]
        assert np.array_equal(bins, alias.current_bin)
        assert np.array_equal(np.load(self.paths["bins"]), alias.stage_bin)
        return bins

    def fingerprint_path(self, subject):
        return self.can_frozen / f"fc_fingerprints_2608x65/{subject}_fingerprint_z_2608x65.npy"

    @cached_property
    def current_profiles(self):
        arrays = [np.load(self.fingerprint_path(subject)).astype(float) for subject in self.data.index]
        assert all(a.shape == (2608, 65) and np.isfinite(a).all() for a in arrays)
        stack = np.stack(arrays)
        return np.stack([stack[:, self.current_bins == b].mean(axis=1) for b in range(1, 11)], axis=1)

    @cached_property
    def templates(self):
        hc = self.data.Group.eq("HC").to_numpy()
        template_sum = self.current_profiles[hc].sum(axis=0)
        templates = np.broadcast_to(template_sum / hc.sum(), self.current_profiles.shape).copy()
        templates[hc] = (template_sum - self.current_profiles[hc]) / (hc.sum() - 1)
        return templates

    def stage_mask(self, stage):
        return self.mapping.reward_stage.eq(stage).to_numpy()


def cases(data, score, outcome, terms=COVARIATES):
    return data.dropna(subset=list(dict.fromkeys([score, outcome, *terms])))
