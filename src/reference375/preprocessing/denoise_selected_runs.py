#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
静息态去噪脚本

Input and output locations are supplied with command-line arguments. The
example config in this repository documents the expected private inputs.

去噪策略：
1. 删除 non-steady-state volumes；
2. 24P motion regressors；
3. aCompCor 前5个成分；
4. FD > 0.5 mm spike regressors；
5. 线性去趋势；
6. 0.01–0.08 Hz 带通；
7. 不平滑；
8. 不做 GSR；
9. DOF ratio >= 0.50。

注意：
- spike volumes 不直接删除，而是作为单独 spike regressors 回归；
- non-steady-state volumes 直接从 BOLD 和 confounds 中删除；
- 每名被试只处理 final_rest_runs_for_denoise.tsv 中选中的一个 run。
"""

import os
import re
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import index_img
from nilearn.maskers import NiftiMasker


def get_subject_number(sub):
    m = re.search(r"sub-(\d+)", str(sub))
    return int(m.group(1)) if m else 10**9


def normalize_run_label(run_value):
    run_value = str(run_value)
    if run_value in ["nan", "NA", "run-NA", "None", ""]:
        return "run-NA"
    return run_value


def read_tr_from_bold(bold_file):
    img = nib.load(bold_file)
    zooms = img.header.get_zooms()
    if len(zooms) < 4:
        raise ValueError(f"Cannot read TR from NIfTI header: {bold_file}")
    return float(zooms[3])


def find_mask_file(bold_file, subject, run):
    """
    优先使用同一run的 MNI152NLin2009cAsym res-2 brain mask。
    """
    func_dir = os.path.dirname(bold_file)
    files = os.listdir(func_dir)

    candidates = []

    for fname in files:
        if not fname.endswith(".nii.gz"):
            continue
        if subject not in fname:
            continue
        if "task-rest" not in fname:
            continue
        if "desc-brain_mask" not in fname:
            continue
        if "space-MNI152NLin2009cAsym" not in fname:
            continue
        if "res-2" not in fname:
            continue
        if run != "run-NA" and run not in fname:
            continue

        candidates.append(os.path.join(func_dir, fname))

    if len(candidates) == 0:
        # 修改提示：
        # 如果没有res-2 MNI mask，退一步找任意 MNI task-rest mask。
        for fname in files:
            if not fname.endswith(".nii.gz"):
                continue
            if subject not in fname:
                continue
            if "task-rest" not in fname:
                continue
            if "desc-brain_mask" not in fname:
                continue
            if "space-MNI152NLin2009cAsym" not in fname:
                continue
            if run != "run-NA" and run not in fname:
                continue

            candidates.append(os.path.join(func_dir, fname))

    if len(candidates) == 0:
        # 修改提示：
        # 如果仍没有，最后退一步找任意 task-rest brain mask。
        for fname in files:
            if not fname.endswith(".nii.gz"):
                continue
            if subject not in fname:
                continue
            if "task-rest" not in fname:
                continue
            if "desc-brain_mask" not in fname:
                continue
            if run != "run-NA" and run not in fname:
                continue

            candidates.append(os.path.join(func_dir, fname))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Cannot find brain mask for {subject} {run}, bold={bold_file}"
        )

    return sorted(candidates)[0]


def make_nonsteady_mask(confounds):
    """
    fMRIPrep 的 non-steady-state columns 通常是：
    non_steady_state_outlier00, non_steady_state_outlier01, ...
    """
    cols = [c for c in confounds.columns if c.startswith("non_steady_state_outlier")]

    if len(cols) == 0:
        return np.zeros(len(confounds), dtype=bool), []

    tmp = confounds[cols].fillna(0)
    mask = tmp.sum(axis=1).values > 0

    return mask, cols


def get_motion_24p(confounds):
    """
    Friston-24:
    6 motion + 6 derivative + 6 squared + 6 derivative squared
    fMRIPrep列名通常为：
    trans_x, trans_x_derivative1, trans_x_power2, trans_x_derivative1_power2
    rot_x, ...
    """
    base_cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

    missing = [c for c in base_cols if c not in confounds.columns]
    if missing:
        raise ValueError(f"Missing base motion columns: {missing}")

    cols = []

    for base in base_cols:
        for suffix in ["", "_derivative1", "_power2", "_derivative1_power2"]:
            col = base + suffix
            if col in confounds.columns:
                cols.append(col)

    if len(cols) < 24:
        print(f"[WARNING] Motion 24P columns found: {len(cols)} < 24. Use available columns.")

    motion = confounds[cols].copy()
    return motion, cols


def get_acompcor_first5(confounds):
    """
    选取 fMRIPrep confounds 中前5个 aCompCor 成分。
    """
    cols = [c for c in confounds.columns if re.match(r"^a_comp_cor_\d+", c)]

    def compcor_key(x):
        m = re.search(r"(\d+)$", x)
        return int(m.group(1)) if m else 10**9

    cols = sorted(cols, key=compcor_key)
    selected = cols[:5]

    if len(selected) < 5:
        print(f"[WARNING] aCompCor columns found: {len(selected)} < 5. Use available columns.")

    if len(selected) == 0:
        return pd.DataFrame(index=confounds.index), []

    acompcor = confounds[selected].copy()
    return acompcor, selected


def make_fd_spikes(confounds, fd_cutoff=0.5):
    """
    FD > 0.5 mm 的每个时间点生成一个单独 spike regressor。
    """
    if "framewise_displacement" not in confounds.columns:
        raise ValueError("Missing framewise_displacement column.")

    fd = pd.to_numeric(confounds["framewise_displacement"], errors="coerce").fillna(0.0)
    spike_idx = np.where(fd.values > fd_cutoff)[0]

    spike_df = pd.DataFrame(index=confounds.index)

    for i, idx in enumerate(spike_idx):
        col = f"fd_spike_{i:03d}"
        spike_df[col] = 0.0
        spike_df.iloc[idx, spike_df.columns.get_loc(col)] = 1.0

    return spike_df, list(spike_df.columns), int(len(spike_idx))


def clean_confounds_matrix(confounds):
    """
    数值化混杂矩阵：
    1. 转为数值；
    2. NaN填0；
    3. 删除全零列；
    4. 删除零方差列。
    """
    if confounds.shape[1] == 0:
        return confounds.copy()

    x = confounds.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    keep_cols = []

    for col in x.columns:
        v = x[col].values

        if np.allclose(v, 0):
            continue

        if np.nanstd(v) == 0:
            continue

        keep_cols.append(col)

    return x[keep_cols].copy()


def compute_dof_ratio(n_timepoints, nuisance):
    """
    DOF ratio = (n_timepoints - rank(confounds)) / n_timepoints
    """
    if nuisance.shape[1] == 0:
        rank = 0
    else:
        rank = int(np.linalg.matrix_rank(nuisance.values))

    dof = int(n_timepoints - rank)
    dof_ratio = float(dof / n_timepoints) if n_timepoints > 0 else 0.0

    return dof, rank, dof_ratio


def build_output_name(subject, run):
    if run == "run-NA":
        return f"{subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-denoised_bold.nii.gz"

    return f"{subject}_task-rest_{run}_space-MNI152NLin2009cAsym_res-2_desc-denoised_bold.nii.gz"


def process_one(row, out_root, fd_cutoff, low_pass, high_pass, min_dof_ratio):
    subject = str(row["subject"])
    run = normalize_run_label(row["run"])
    bold_file = str(row["bold_file"])
    confounds_file = str(row["confounds_file"])

    out_dir = os.path.join(out_root, subject)
    os.makedirs(out_dir, exist_ok=True)

    output_file = os.path.join(out_dir, build_output_name(subject, run))
    used_confounds_file = output_file.replace(
        "_desc-denoised_bold.nii.gz",
        "_desc-denoiseConfounds.tsv"
    )

    result = {
        "subject": subject,
        "run": run,
        "bold_file": bold_file,
        "confounds_file": confounds_file,
        "mask_file": "",
        "output_file": output_file,
        "used_confounds_file": used_confounds_file,
        "status": "FAIL",
        "reason": "",
        "n_volumes_original": np.nan,
        "n_nonsteady_removed": np.nan,
        "n_volumes_after_nonsteady": np.nan,
        "tr": np.nan,
        "n_motion_regressors": np.nan,
        "motion_regressors": "",
        "n_acompcor_regressors": np.nan,
        "acompcor_regressors": "",
        "n_fd_spike_regressors": np.nan,
        "n_total_regressors_after_cleaning": np.nan,
        "confound_rank": np.nan,
        "dof": np.nan,
        "dof_ratio": np.nan,
        "high_pass": high_pass,
        "low_pass": low_pass,
        "detrend": "YES",
        "smoothing": "NO",
        "gsr": "NO",
    }

    try:
        if not os.path.exists(bold_file):
            raise FileNotFoundError(f"bold_file does not exist: {bold_file}")

        if not os.path.exists(confounds_file):
            raise FileNotFoundError(f"confounds_file does not exist: {confounds_file}")

        if os.path.getsize(bold_file) == 0:
            raise ValueError(f"bold_file is zero size: {bold_file}")

        if os.path.getsize(confounds_file) == 0:
            raise ValueError(f"confounds_file is zero size: {confounds_file}")

        confounds = pd.read_csv(confounds_file, sep="\t")
        n_original = len(confounds)
        result["n_volumes_original"] = int(n_original)

        tr = read_tr_from_bold(bold_file)
        result["tr"] = float(tr)

        nonsteady_mask, nonsteady_cols = make_nonsteady_mask(confounds)
        keep_idx = np.where(~nonsteady_mask)[0]

        n_nonsteady = int(nonsteady_mask.sum())
        n_after_nonsteady = int(len(keep_idx))

        result["n_nonsteady_removed"] = n_nonsteady
        result["n_volumes_after_nonsteady"] = n_after_nonsteady

        if n_after_nonsteady < 10:
            raise ValueError("Too few volumes after removing non-steady-state volumes.")

        # 修改提示：
        # non-steady-state volumes 直接删除，confounds同步删除对应行。
        conf_valid = confounds.iloc[keep_idx].reset_index(drop=True)

        motion_df, motion_cols = get_motion_24p(conf_valid)
        acompcor_df, acompcor_cols = get_acompcor_first5(conf_valid)
        spike_df, spike_cols, n_spikes = make_fd_spikes(conf_valid, fd_cutoff=fd_cutoff)

        nuisance = pd.concat(
            [motion_df, acompcor_df, spike_df],
            axis=1
        )

        nuisance = clean_confounds_matrix(nuisance)

        result["n_motion_regressors"] = int(len(motion_cols))
        result["motion_regressors"] = ";".join(motion_cols)
        result["n_acompcor_regressors"] = int(len(acompcor_cols))
        result["acompcor_regressors"] = ";".join(acompcor_cols)
        result["n_fd_spike_regressors"] = int(n_spikes)
        result["n_total_regressors_after_cleaning"] = int(nuisance.shape[1])

        dof, rank, dof_ratio = compute_dof_ratio(n_after_nonsteady, nuisance)

        result["confound_rank"] = int(rank)
        result["dof"] = int(dof)
        result["dof_ratio"] = float(dof_ratio)

        if dof_ratio < min_dof_ratio:
            result["status"] = "FAIL"
            result["reason"] = f"dof_ratio_lt_{min_dof_ratio}"
            return result

        mask_file = find_mask_file(bold_file, subject, run)
        result["mask_file"] = mask_file

        # 修改提示：
        # BOLD也同步删除 non-steady-state volumes。
        bold_valid = index_img(bold_file, keep_idx)

        masker = NiftiMasker(
            mask_img=mask_file,
            standardize=False,
            detrend=True,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=tr,
            smoothing_fwhm=None,
            memory=None,
            verbose=0
        )

        cleaned_ts = masker.fit_transform(
            bold_valid,
            confounds=nuisance.values if nuisance.shape[1] > 0 else None
        )

        cleaned_img = masker.inverse_transform(cleaned_ts)
        cleaned_img.to_filename(output_file)

        nuisance.to_csv(used_confounds_file, sep="\t", index=False)

        result["status"] = "OK"
        result["reason"] = "ok"

        return result

    except Exception as e:
        result["status"] = "FAIL"
        result["reason"] = f"{type(e).__name__}: {str(e)}"
        result["traceback"] = traceback.format_exc()
        return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-tsv",
        required=True,
        help="Private run manifest (not distributed).",
    )

    parser.add_argument(
        "--out-root",
        required=True,
        help="Output directory for denoised images.",
    )

    parser.add_argument(
        "--qc-out",
        required=True,
        help="Output TSV for denoising QC.",
    )

    parser.add_argument("--n-proc", type=int, default=1)
    parser.add_argument("--fd-cutoff", type=float, default=0.5)
    parser.add_argument("--high-pass", type=float, default=0.01)
    parser.add_argument("--low-pass", type=float, default=0.08)
    parser.add_argument("--min-dof-ratio", type=float, default=0.50)

    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.qc_out), exist_ok=True)

    df = pd.read_csv(args.input_tsv, sep="\t")

    required_cols = ["subject", "run", "bold_file", "confounds_file"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Input TSV missing required columns: {missing}")

    df = df.sort_values(
        "subject",
        key=lambda s: s.map(get_subject_number)
    ).reset_index(drop=True)

    print("============================================================")
    print("Rest denoising started")
    print("Input TSV:", args.input_tsv)
    print("Runs:", len(df))
    print("Output root:", args.out_root)
    print("QC output:", args.qc_out)
    print("n_proc:", args.n_proc)
    print("Confounds: 24P + first 5 aCompCor + FD spikes > 0.5")
    print("Remove non-steady-state volumes: YES")
    print("Detrend: YES")
    print("Bandpass: 0.01–0.08 Hz")
    print("Smoothing: NO")
    print("GSR: NO")
    print("Min DOF ratio:", args.min_dof_ratio)
    print("============================================================")

    results = []

    if args.n_proc <= 1:
        for i, row in df.iterrows():
            print(f"[{i + 1}/{len(df)}] {row['subject']} {row['run']}")
            res = process_one(
                row=row,
                out_root=args.out_root,
                fd_cutoff=args.fd_cutoff,
                low_pass=args.low_pass,
                high_pass=args.high_pass,
                min_dof_ratio=args.min_dof_ratio
            )
            results.append(res)
            print(res["status"], res["reason"])
    else:
        with ProcessPoolExecutor(max_workers=args.n_proc) as executor:
            future_map = {}

            for idx, row in df.iterrows():
                fut = executor.submit(
                    process_one,
                    row,
                    args.out_root,
                    args.fd_cutoff,
                    args.low_pass,
                    args.high_pass,
                    args.min_dof_ratio
                )
                future_map[fut] = idx

            for fut in as_completed(future_map):
                idx = future_map[fut]
                res = fut.result()
                results.append(res)
                print(
                    f"[{idx + 1}/{len(df)}] "
                    f"{res['subject']} {res['run']} "
                    f"{res['status']} {res['reason']}"
                )

    qc = pd.DataFrame(results)

    if not qc.empty:
        qc["subject_num"] = qc["subject"].apply(get_subject_number)
        qc = qc.sort_values("subject_num").drop(columns=["subject_num"])

    qc.to_csv(args.qc_out, sep="\t", index=False)

    n_ok = int((qc["status"] == "OK").sum())
    n_fail = int((qc["status"] != "OK").sum())

    print("============================================================")
    print("Rest denoising finished")
    print("OK:", n_ok)
    print("FAIL:", n_fail)
    print("QC:", args.qc_out)
    print("============================================================")

    if n_fail > 0:
        print("Failed runs:")
        print(
            qc.loc[
                qc["status"] != "OK",
                ["subject", "run", "reason"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
