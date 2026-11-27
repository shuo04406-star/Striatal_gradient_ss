from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels


OUT_DIR = Path(os.environ["STRIATAL_OUTPUT_ROOT"])
SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from _stage_coupling_common import (  # noqa: E402
    BAYES_RESULTS,
    BIN10_STAGE,
    BOOT_RESULTS,
    BRAIN_BEHAVIOR,
    CLINICAL_IMAGING,
    FROZEN_BEHAVIOR,
    INDIVIDUAL_SLOPES,
    JOINT_RESULTS,
    LOO_RESULTS,
    RAW_ITEM_TABLE,
    RESULTS,
    ROI_BEHAVIOR,
    SUBJECT_DATA,
    fmt,
    format_p,
    sha256,
)


PIPELINE = [
    "01_audit_stage_coupling_data.py",
    "02_build_stage_contrasts.py",
    "03_run_stage_coupling_correlations.py",
    "04_run_joint_stage_models.py",
    "05_run_bayesian_analysis.py",
    "06_run_stability_analysis.py",
    "07_plot_stage_coupling.py",
]


def run_pipeline() -> None:
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    for script_name in PIPELINE:
        script = SCRIPT_DIR / script_name
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=OUT_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        log = (
            f"COMMAND: {sys.executable} {script}\n"
            f"RETURN_CODE: {completed.returncode}\n\n"
            f"STDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}\n"
        )
        (OUT_DIR / "logs" / f"{script.stem}.log").write_text(log, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"Pipeline failed in {script_name}; see logs/{script.stem}.log")
        print(completed.stdout.strip())


def build_report() -> None:
    subject = pd.read_csv(SUBJECT_DATA)
    eligible = subject.loc[subject["eligible_stage_coupling"].astype(bool)].copy()
    results = pd.read_csv(RESULTS)
    standardized = results.loc[results["contrast_scale"].eq("stagewise_z_difference")].set_index("endpoint")
    raw = results.loc[results["contrast_scale"].eq("raw_stage_difference")].set_index("endpoint")
    joint = pd.read_csv(JOINT_RESULTS).set_index("endpoint")
    bayes = pd.read_csv(BAYES_RESULTS)
    bayes_primary = bayes.loc[bayes["prior_role"].eq("primary_prior")].set_index("endpoint")
    bayes_sensitivity = bayes.loc[bayes["prior_role"].eq("prior_sensitivity")].set_index("endpoint")
    stability = pd.read_csv(OUT_DIR / "stability_summary.csv").set_index("endpoint")
    boot = pd.read_csv(BOOT_RESULTS).set_index("endpoint")

    slope = standardized.loc["overall_fc_slope"]
    similarity = standardized.loc["bin10_similarity"]
    slope_joint = joint.loc["overall_fc_slope"]
    sim_joint = joint.loc["bin10_similarity"]
    slope_b = bayes_primary.loc["overall_fc_slope"]
    sim_b = bayes_primary.loc["bin10_similarity"]
    slope_s = stability.loc["overall_fc_slope"]
    sim_s = stability.loc["bin10_similarity"]

    missing_ids = ", ".join(subject.loc[~subject["eligible_stage_coupling"].astype(bool), "subject"])
    ant = eligible["TEPS_anticipatory"]
    con = eligible["TEPS_consummatory"]
    ant_ceiling = int(eligible["TEPS_Ant_theoretical_ceiling"].sum())
    con_ceiling = int(eligible["TEPS_Cons_theoretical_ceiling"].sum())

    def corr_line(endpoint: str) -> str:
        r = standardized.loc[endpoint]
        return (
            f"N={int(r['n'])}, Spearman ρ={r['rho']:.3f}, "
            f"95% bootstrap CI [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}], "
            f"raw p={format_p(float(r['p_raw']))}"
        )

    def joint_line(endpoint: str) -> str:
        r = joint.loc[endpoint]
        return (
            f"β={r['interaction_beta']:.3f}, SE={r['interaction_se']:.3f}, "
            f"95% CI [{r['interaction_ci95_low']:.3f}, {r['interaction_ci95_high']:.3f}], "
            f"p={format_p(float(r['interaction_p']))}"
        )

    def bayes_line(endpoint: str) -> str:
        r = bayes_primary.loc[endpoint]
        return (
            f"posterior median r={r['posterior_median_r']:.3f}, "
            f"95% CrI [{r['credible_interval_95_low']:.3f}, {r['credible_interval_95_high']:.3f}], "
            f"P(r>0)={r['p_r_gt_zero']:.3f}, P(r<0)={r['p_r_lt_zero']:.3f}, "
            f"P(|r|>0.10)={r['p_abs_r_gt_0_10']:.3f}, "
            f"P(|r|>0.20)={r['p_abs_r_gt_0_20']:.3f}"
        )

    overall_conclusion = (
        "Reward-stage neural differentiation was not reliably associated with the corresponding "
        "anticipatory-versus-consummatory pleasure profile in the current SZ sample."
    )
    report = f"""# Reward Stage Coupling Analysis

## 执行摘要

{overall_conclusion}

在冻结的 32 名 SZ 中，28 名同时具有 TEPS Anticipatory、TEPS Consummatory、整体两阶段 FC slope 和 Bin 10 两阶段 HC-reference similarity。主要整体 slope stage-coupling 结果为：{corr_line('overall_fc_slope')}。预先定义的次要 Bin 10 similarity 结果为：{corr_line('bin10_similarity')}。两项点估计均为正，但不确定区间跨 0，不能视为可靠的脑—行为关联。

联合 Stage × Pleasure 模型与差值分析方向一致，但交互项均未达到统计学支持：整体 slope {joint_line('overall_fc_slope')}；Bin 10 similarity {joint_line('bin10_similarity')}。Bayesian posterior 对正向关联有一定倾向，但仍同时容纳接近零与有意义的中等效应；当前更符合“信息量不足”，而不是强零效应证据。

## 1. 数据与既有结果审计

### 1.1 行为数据

- 实际变量名：`TEPS_anticipatory` 与 `TEPS_consummatory`；与既有 ROI-behavior extension 的个体值和 ID 逐例一致。
- 分数方向：较高保留分数表示较高的相应愉悦认可。
- TEPS Anticipatory 为 9 个 1–6 分项目之和，理论范围 9–54；有效 SZ 中均值 {ant.mean():.2f}、SD {ant.std(ddof=1):.2f}、观察范围 {ant.min():.0f}–{ant.max():.0f}。
- TEPS Consummatory 为 10 个 1–6 分项目之和，理论范围 10–60；有效 SZ 中均值 {con.mean():.2f}、SD {con.std(ddof=1):.2f}、观察范围 {con.min():.0f}–{con.max():.0f}。
- 有效 N=28/32；缺失两个分量表的 ID：{missing_ids}。未插补，也未从汇总统计反推个体值。
- 理论下限命中为 0；Anticipatory 理论上限命中 {ant_ceiling} 人，Consummatory 理论上限命中 {con_ceiling} 人。stage contrast 本身没有被截断或重编码。

### 1.2 神经指标 A：整体 stage-specific FC slope

- `Anticipation_slope` 与 `Outcome_slope` 对 64 名冻结参与者均存在，并与 canonical individual-slope 输出逐值一致。
- 两者定义完全相同：每名参与者、每个阶段在 10 个固定 G1 bins 上拟合 `FC ~ (Bin - 5.5)` 的 OLS 斜率，只改变 stage。
- G1 方向固定为低值 VST/caudate-rich、高值 putamen-rich。正 slope 表示 raw FC 随 G1 bin 增加，负 slope 表示下降；它不是任务激活强度。
- 该指标来自固定 375 人参考 G1 坐标和个体级 raw-FC stage profile；本次复用既有结果，没有重算或改动主分析。

### 1.3 神经指标 B：Bin 10 similarity

- 两阶段均以同一算法计算：参与者 Bin 10 的 stage-specific FC fingerprint 与 HC reference fingerprint 做 Pearson 相关，再 Fisher-z。
- Anticipation 使用 43 个 stage ROIs，Outcome 使用 19 个 stage ROIs；差别来自预先固定的 stage mapping，而非结果选择。
- SZ 两阶段均使用全部 32 HC 的均值 reference（`HC32`）。个体值与既有临床扩展数据逐值一致。

完整审计见 `stage_coupling_data_audit.csv` 与 `stage_coupling_audit_metadata.json`。

## 2. Contrast 定义

所有标准化均只在本次 28 名有效 SZ 内进行，使用样本 SD（ddof=1）：

- `TEPS_stage_contrast = TEPS_Ant_z - TEPS_Cons_z`
- `NeuralSlope_stage_contrast = Ant_slope_z - Out_slope_z`
- `Bin10Similarity_stage_contrast = Bin10_Ant_similarity_z - Bin10_Out_similarity_z`

正的 TEPS contrast 表示该患者相对其 Consummatory pleasure 而言，Anticipatory pleasure 在本样本中更高；这不是疾病严重程度。原始 slope difference 与原始 Fisher-z similarity difference 均保留为敏感性比较，没有替代主定义。

## 3. 主要与次要 stage-coupling 相关

| 层级 | 分析 | N | Spearman ρ | 95% bootstrap CI | raw p | bootstrap sign consistency |
|---|---|---:|---:|---:|---:|---:|
| 主要 | TEPS contrast × overall FC slope contrast | {int(slope['n'])} | {slope['rho']:.3f} | [{slope['ci95_low']:.3f}, {slope['ci95_high']:.3f}] | {slope['p_raw']:.3f} | {boot.loc['overall_fc_slope','sign_consistency']:.3f} |
| 次要 | TEPS contrast × Bin 10 similarity contrast | {int(similarity['n'])} | {similarity['rho']:.3f} | [{similarity['ci95_low']:.3f}, {similarity['ci95_high']:.3f}] | {similarity['p_raw']:.3f} | {boot.loc['bin10_similarity','sign_consistency']:.3f} |

由于 TEPS contrast 存在 ties，报告 SciPy 的双侧渐近 Spearman p；没有把不适用的 exact p 强行解释为精确检验。两项分析按主要与预先定义的次要终点分开解释，没有事后改变 contrast，也没有扩展到 ROI 或其他 bins。

原始差敏感性结果：overall slope raw difference ρ={raw.loc['overall_fc_slope','rho']:.3f}, p={raw.loc['overall_fc_slope','p_raw']:.3f}；Bin 10 raw difference ρ={raw.loc['bin10_similarity','rho']:.3f}, p={raw.loc['bin10_similarity','p_raw']:.3f}。这些仅用于判断标准化差值的量纲处理是否改变方向，不作为新增独立发现。

## 4. 联合 Stage × Pleasure 模型

每名患者两行，Outcome 为 reference，神经值与 TEPS 均按 stage 在同一 28 人样本内标准化。由于每人仅两次观测，采用预先允许的 repeated-measures 等价方案：`Neural_z ~ Stage * TEPS_z` 的 OLS，并以 Subject cluster 的 CR1 协方差和 cluster df 校正推断；不强行估计边界不稳定的随机效应方差。

| 神经指标 | N subjects / observations | Stage × TEPS β | SE | 95% CI | p | 与差值 Spearman 方向一致 |
|---|---:|---:|---:|---:|---:|---|
| Overall FC slope | {int(slope_joint['n_subjects'])} / {int(slope_joint['n_observations'])} | {slope_joint['interaction_beta']:.3f} | {slope_joint['interaction_se']:.3f} | [{slope_joint['interaction_ci95_low']:.3f}, {slope_joint['interaction_ci95_high']:.3f}] | {slope_joint['interaction_p']:.3f} | {bool(slope_joint['direction_consistent_with_difference_analysis'])} |
| Bin 10 similarity | {int(sim_joint['n_subjects'])} / {int(sim_joint['n_observations'])} | {sim_joint['interaction_beta']:.3f} | {sim_joint['interaction_se']:.3f} | [{sim_joint['interaction_ci95_low']:.3f}, {sim_joint['interaction_ci95_high']:.3f}] | {sim_joint['interaction_p']:.3f} | {bool(sim_joint['direction_consistent_with_difference_analysis'])} |

模型计算为闭式 OLS，故 convergence 不适用；design rank 完整，condition numbers 分别为 {slope_joint['condition_number']:.2f} 与 {sim_joint['condition_number']:.2f}。

## 5. Bayesian correlation

Bayesian 分析使用与 Spearman 目标一致的 Gaussian-copula rank-likelihood approximation：将 average ranks 映射为正态分数，在 latent copula correlation 上放置 LKJ(η=1) prior，再转换为 population Spearman-type correlation。使用 {int(slope_b['grid_points']):,} 点 deterministic grid quadrature；posterior 权重归一化为 1，边界质量均可忽略，因此没有 MCMC convergence 问题。

- Overall slope：{bayes_line('overall_fc_slope')}。
- Bin 10 similarity：{bayes_line('bin10_similarity')}。

Prior sensitivity 使用更向零收缩的 LKJ(η=2)：overall posterior median 从 {slope_b['posterior_median_r']:.3f} 变为 {bayes_sensitivity.loc['overall_fc_slope','posterior_median_r']:.3f}；Bin 10 从 {sim_b['posterior_median_r']:.3f} 变为 {bayes_sensitivity.loc['bin10_similarity','posterior_median_r']:.3f}。结论没有因 prior sensitivity 改变。

BF10/BF01 未报告，因为本实现没有另拟合 point-null model；不以不可靠的 Savage–Dickey 或事后 prior 替代。Posterior 结果主要说明：overall slope 为“{slope_b['precision_assessment']}”；Bin 10 为“{sim_b['precision_assessment']}”。因此目前不能把未显著结果写成“没有关系”。

## 6. 稳定性、影响点与分布检查

| 分析 | LOO rho min–max | median rho | 方向稳定 | 最大变化的删除对象 | 最大 |Δrho| |
|---|---:|---:|---|---|---:|
| Overall FC slope | {slope_s['loo_min_rho']:.3f} to {slope_s['loo_max_rho']:.3f} | {slope_s['loo_median_rho']:.3f} | {bool(slope_s['loo_direction_stable'])} | {slope_s['max_change_deleted_subject']} | {slope_s['max_absolute_change']:.3f} |
| Bin 10 similarity | {sim_s['loo_min_rho']:.3f} to {sim_s['loo_max_rho']:.3f} | {sim_s['loo_median_rho']:.3f} | {bool(sim_s['loo_direction_stable'])} | {sim_s['max_change_deleted_subject']} | {sim_s['max_absolute_change']:.3f} |

没有因影响大小删除任何患者。Robust-z 检查在 behavior contrast 中标记 {int(slope_s['n_robust_outliers_behavior'])} 个 >3.5 的观测；overall neural contrast 标记 {int(slope_s['n_robust_outliers_neural'])} 个，Bin 10 neural contrast 标记 {int(sim_s['n_robust_outliers_neural'])} 个。TEPS contrast 的 tied values 涉及 {int(slope_s['ties_behavior_n_values'])} 个观测；Spearman average ranks 已保留这些值。按任务边界没有搜索 nonlinear models；散点图用于记录而非据图删点。

## 7. Figure 1

![Figure 1](figures/Figure1_stage_coupling.png)

Panel a 为主要 overall slope coupling；Panel b 为预先定义的次要 Bin 10 coupling；Panel c 显示逐一删除患者后的 ρ；Panel d 显示 Bayesian posterior median 与 95% CrI。图中不使用显著性星号。SVG/PDF/TIFF/PNG 及 source data 位于 `figures/`。

## 8. 对七个研究问题的直接回答

1. **行为 contrast 是否与整体 neural slope contrast 相关？** 点估计为正（ρ={slope['rho']:.3f}），但 bootstrap CI 跨 0、raw p={slope['p_raw']:.3f}，当前没有可靠证据。
2. **是否与 Bin 10 similarity contrast 相关？** 点估计较小且为正（ρ={similarity['rho']:.3f}），CI 更宽并跨 0、raw p={similarity['p_raw']:.3f}，未获得可靠支持。
3. **两个 stage 的 brain–behavior association 是否显著不同？** 两个 Stage × TEPS interaction 均未获得统计支持；见第 4 节。
4. **Bayesian posterior 支持零效应还是信息不足？** 主要是信息不足：posterior 对正向关系有不同程度倾向，但 95% CrI 仍包括接近零，并允许至少一个中等方向；没有强零效应证据。
5. **是否由个别患者强烈驱动？** LOO 方向{'' if bool(slope_s['loo_direction_stable']) and bool(sim_s['loo_direction_stable']) else '并非完全'}稳定，最大 |Δρ| 为 {max(slope_s['max_absolute_change'], sim_s['max_absolute_change']):.3f}；个别患者会改变效应大小，但没有单一删除使结果变为可靠发现，也未据此删人。
6. **差值分析与 joint model 是否方向一致？** 两个指标均一致，但 joint CI 均跨 0，因此一致性只限于点估计方向。
7. **是否增加临床解释？** 仍没有可靠的 brain–behavior association。新增价值在于以同构 stage contrast 正面检验了 stage concordance，并量化了不确定性与稳定性；它不改变既有主分析结论。

## 9. 解释边界

本分析是理论驱动的探索性扩展，不是原始预注册确认性终点。没有遍历 ROI、bins、PANSS/ACIPS items、亚组或 nonlinear models。所有 raw p、bootstrap、LOO 与 Bayesian 结果共同用于解释，不以单一 p 值下结论。

推荐结论：

> Reward-stage neural differentiation was not reliably associated with the corresponding anticipatory-versus-consummatory pleasure profile in the current SZ sample. The current data were insufficiently precise to distinguish near-null from potentially meaningful positive associations.
"""
    (OUT_DIR / "STAGE_COUPLING_ANALYSIS_REPORT.md").write_text(report, encoding="utf-8")


def build_todo() -> None:
    text = """# TODO\n\n## Analysis status\n\n- [x] Audit individual-level TEPS, overall slopes, and Bin 10 stage similarity.\n- [x] Build fixed z-difference stage contrasts without redefining endpoints after inspection.\n- [x] Run primary and prespecified secondary Spearman analyses.\n- [x] Run clustered repeated-measures Stage x TEPS sensitivity models.\n- [x] Run Bayesian rank-correlation analysis with prior sensitivity.\n- [x] Run 10,000-resample bootstrap and leave-one-out stability analyses.\n- [x] Export the four-panel publication figure and source data.\n- [x] Stop without expanding to ROI/bin/item/subgroup/nonlinear searches.\n\n## Deferred to future independent work (not part of this task)\n\n- [ ] Replicate the two fixed contrasts in a larger independent SZ sample with complete TEPS data.\n- [ ] Before journal submission, check the final target journal's current figure specifications.\n"""
    (OUT_DIR / "TODO.md").write_text(text, encoding="utf-8")


def build_provenance() -> None:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    input_paths = [
        BRAIN_BEHAVIOR,
        INDIVIDUAL_SLOPES,
        FROZEN_BEHAVIOR,
        BIN10_STAGE,
        CLINICAL_IMAGING,
        ROI_BEHAVIOR,
        RAW_ITEM_TABLE,
    ]
    output_paths = [
        OUT_DIR / "STAGE_COUPLING_ANALYSIS_REPORT.md",
        SUBJECT_DATA,
        RESULTS,
        JOINT_RESULTS,
        BAYES_RESULTS,
        LOO_RESULTS,
        BOOT_RESULTS,
        OUT_DIR / "stage_coupling_data_audit.csv",
        OUT_DIR / "stage_coupling_distribution_audit.csv",
        OUT_DIR / "stability_summary.csv",
        OUT_DIR / "scatter_inspection.csv",
        OUT_DIR / "figures" / "Figure1_stage_coupling.svg",
        OUT_DIR / "figures" / "Figure1_stage_coupling.pdf",
        OUT_DIR / "figures" / "Figure1_stage_coupling.tiff",
        OUT_DIR / "figures" / "Figure1_stage_coupling.png",
        OUT_DIR / "figures" / "Figure1_stage_coupling_source_data.csv",
        OUT_DIR / "TODO.md",
    ]
    script_paths = [Path(__file__).resolve()] + [SCRIPT_DIR / x for x in PIPELINE] + [SCRIPT_DIR / "_stage_coupling_common.py"]
    rows: list[dict[str, object]] = []
    for category, paths in [("input", input_paths), ("script", script_paths), ("output", output_paths)]:
        for path in paths:
            if not path.is_file():
                continue
            rows.append(
                {
                    "category": category,
                    "artifact": str(path.relative_to(OUT_DIR)) if path.is_relative_to(OUT_DIR) else str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "generated_at": now,
                    "pipeline_version": "stage_coupling_v1_20261029",
                }
            )
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
    }
    for name, version in versions.items():
        rows.append(
            {
                "category": "software",
                "artifact": name,
                "sha256": "",
                "bytes": "",
                "generated_at": now,
                "pipeline_version": version,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "analysis_provenance.csv", index=False)


def main() -> None:
    run_pipeline()
    build_report()
    build_todo()
    build_provenance()
    print("Stage-coupling analysis complete.")


if __name__ == "__main__":
    main()
