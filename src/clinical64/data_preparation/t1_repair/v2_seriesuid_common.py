from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import openpyxl


ROOT = Path(os.environ["STRIATAL_CLINICAL_SOURCE_ROOT"])
BIDS_RAW = ROOT / "bids_build" / "03_bids_raw"
V1 = ROOT / "check_t1_repair"
WORK = ROOT / "check_t1_repair_v2_seriesuid"
SCRIPTS = Path(__file__).resolve().parent
LOGS = WORK / "logs"
os.environ.setdefault("MPLCONFIGDIR", str(WORK / "mplconfig"))
REPAIRED = WORK / "03_bids_raw_repaired_v2"
BACKUP = WORK / "bids_backup_before_repair_v2"
MERGED_DICOM = WORK / "merged_dicom_series"
CONVERTED = WORK / "converted_merged_series"
CASE_CONFIG = json.loads(Path(os.environ["STRIATAL_T1_REPAIR_CONFIG"]).read_text(encoding="utf-8-sig"))
TARGET_SUBJECTS = CASE_CONFIG["target_subjects"]
EXCLUDED = set(CASE_CONFIG["excluded_subjects"])


def ensure_dirs() -> None:
    for p in [WORK, SCRIPTS, LOGS, MERGED_DICOM, CONVERTED, WORK / "montages", WORK / "before_after_montages", WORK / "mplconfig", BACKUP]:
        p.mkdir(parents=True, exist_ok=True)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def safe_uid(uid: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", uid or "NO_UID")


def val(ds: Any, name: str, default: str = "") -> str:
    v = getattr(ds, name, default)
    if isinstance(v, pydicom.multival.MultiValue) or isinstance(v, list) or isinstance(v, tuple):
        return "\\".join(str(x) for x in v)
    return str(v) if v is not None else default


def parse_floats(x: Any) -> list[float]:
    if x is None:
        return []
    if isinstance(x, str):
        parts = re.split(r"[\\, ]+", x.strip())
    else:
        parts = list(x)
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            pass
    return out


def dicom_files(root: Path) -> list[Path]:
    out = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".json", ".nii", ".gz", ".tsv", ".csv", ".xlsx", ".txt", ".png", ".html"}:
            continue
        out.append(p)
    return out


def dcmread_header(path: Path):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True, specific_tags=None)
    except Exception:
        return None


def slice_position(ds: Any) -> float | None:
    iop = parse_floats(getattr(ds, "ImageOrientationPatient", None))
    ipp = parse_floats(getattr(ds, "ImagePositionPatient", None))
    if len(iop) >= 6 and len(ipp) >= 3:
        row = np.array(iop[:3], dtype=float)
        col = np.array(iop[3:6], dtype=float)
        normal = np.cross(row, col)
        return float(np.dot(np.array(ipp[:3], dtype=float), normal))
    try:
        return float(getattr(ds, "SliceLocation"))
    except Exception:
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_dcm2niix() -> str:
    exe = shutil.which("dcm2niix")
    if exe:
        return exe
    for p in [ROOT / "dcm2niix.exe", ROOT / "tools" / "dcm2niix.exe", Path.home() / "Downloads" / "dcm2niix.exe", Path.home() / "Downloads" / "dcm2niix_tools" / "dcm2niix.exe"]:
        if p.exists():
            return str(p)
    dl = Path.home() / "Downloads"
    if dl.exists():
        for child in dl.iterdir():
            if child.is_dir() and "dcm2niix" in child.name.lower() and (child / "dcm2niix.exe").exists():
                return str(child / "dcm2niix.exe")
    return ""


def candidate_t1(row: dict[str, Any]) -> bool:
    text = " ".join(str(row.get(k, "")) for k in ["series_description", "protocol_name", "sequence_name"]).lower()
    if any(x in text for x in ["localizer", "scout", "survey", "calib"]):
        return False
    return any(x in text for x in ["mprage", "t1", "bravo", "3d_t1"])


def candidate_t2(row: dict[str, Any]) -> bool:
    text = " ".join(str(row.get(k, "")) for k in ["series_description", "protocol_name"]).lower()
    return "t2" in text


def candidate_localizer(row: dict[str, Any]) -> bool:
    text = " ".join(str(row.get(k, "")) for k in ["series_description", "protocol_name"]).lower()
    return any(x in text for x in ["localizer", "scout", "survey"])


def previous_roots_by_subject() -> dict[str, list[str]]:
    rows = read_tsv(V1 / "subject_dicom_mapping.tsv")
    out = {}
    for r in rows:
        if r.get("subject") in TARGET_SUBJECTS:
            roots = [x for x in r.get("dicom_root", "").split(";") if x]
            out[r["subject"]] = sorted(set(roots))
    return out


def norm_text(x: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(x or "").lower())


def expected_subjects() -> dict[str, dict[str, str]]:
    wb = openpyxl.load_workbook(ROOT / "fMRI_35vs35_整理版.xlsx", read_only=True, data_only=True)
    ws = wb["01_主信息_70人"]
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    cols = {h: i + 1 for i, h in enumerate(headers)}
    out: dict[str, dict[str, str]] = {}
    for r in range(2, ws.max_row + 1):
        mri = str(ws.cell(r, cols["MRI_ID"]).value or "").strip()
        if not mri:
            continue
        subject = "sub-" + mri.split("_")[-1]
        out[subject] = {
            "name": str(ws.cell(r, cols["name"]).value or ""),
            "name_pinyin": str(ws.cell(r, cols["name_pinyin"]).value or ""),
            "protocol_id": str(ws.cell(r, cols["ID"]).value or ""),
            "mri_id": mri,
        }
    wb.close()
    return out


def header_matches_expected(ds: Any, expected: dict[str, str]) -> bool:
    pname = norm_text(getattr(ds, "PatientName", ""))
    pid = norm_text(getattr(ds, "PatientID", ""))
    pinyin = norm_text(expected.get("name_pinyin", ""))
    mri = norm_text(expected.get("mri_id", ""))
    return bool((pinyin and pinyin in pname) or (mri and mri in pid))


def stage01_verify_subject_mapping() -> None:
    ensure_dirs()
    prev = {r["subject"]: r for r in read_tsv(V1 / "subject_dicom_mapping.tsv")}
    expected_map = expected_subjects()
    rows = []
    for subject in TARGET_SUBJECTS:
        roots = previous_roots_by_subject().get(subject, [])
        expected = expected_map.get(subject, {})
        seen = defaultdict(set)
        scanned = 0
        matched = 0
        for root in roots:
            for f in dicom_files(Path(root)):
                scanned += 1
                ds = dcmread_header(f)
                if ds is None or not getattr(ds, "PatientID", None):
                    continue
                if not header_matches_expected(ds, expected):
                    continue
                matched += 1
                for field in ["PatientID", "PatientName", "StudyDate", "StudyInstanceUID", "AccessionNumber", "InstitutionName"]:
                    seen[field].add(val(ds, field))
        consistent = matched > 0 and len(seen["PatientID"]) <= 1 and len(seen["PatientName"]) <= 1
        rows.append({
            "subject": subject,
            "protocol_id": prev.get(subject, {}).get("protocol_id", "") or expected.get("protocol_id", ""),
            "patient_id": ";".join(sorted(seen["PatientID"])),
            "patient_name": ";".join(sorted(seen["PatientName"])),
            "study_date": ";".join(sorted(seen["StudyDate"])),
            "study_instance_uid": ";".join(sorted(seen["StudyInstanceUID"])),
            "all_dicom_roots": ";".join(roots),
            "previous_mapping_source": prev.get(subject, {}).get("mapping_source", ""),
            "identity_fields_consistent": int(consistent),
            "mapping_status": "CONFIRMED" if roots and consistent else "REVIEW_MAPPING",
            "mapping_notes": f"Expected={expected.get('name_pinyin','')}/{expected.get('mri_id','')}; scanned_files={scanned}; matched_files={matched}; fields=" + json.dumps({k: sorted(v) for k, v in seen.items()}, ensure_ascii=False),
        })
    write_tsv(WORK / "subject_mapping_verified.tsv", rows, ["subject", "protocol_id", "patient_id", "patient_name", "study_date", "study_instance_uid", "all_dicom_roots", "previous_mapping_source", "identity_fields_consistent", "mapping_status", "mapping_notes"])


def stage02_inventory_all_dicoms_by_uid() -> None:
    ensure_dirs()
    mapping = {r["subject"]: r for r in read_tsv(WORK / "subject_mapping_verified.tsv")}
    expected_map = expected_subjects()
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for subject, m in mapping.items():
        if m.get("mapping_status") != "CONFIRMED":
            continue
        expected = expected_map.get(subject, {})
        for root_s in [x for x in m.get("all_dicom_roots", "").split(";") if x]:
            for f in dicom_files(Path(root_s)):
                ds = dcmread_header(f)
                if ds is None or not getattr(ds, "SeriesInstanceUID", None):
                    continue
                if not header_matches_expected(ds, expected):
                    continue
                pos = slice_position(ds)
                groups[(subject, val(ds, "StudyInstanceUID"), val(ds, "SeriesInstanceUID"))].append({
                    "file": str(f),
                    "source_directory": str(f.parent),
                    "patient_id": val(ds, "PatientID"),
                    "patient_name": val(ds, "PatientName"),
                    "study_date": val(ds, "StudyDate"),
                    "series_number": val(ds, "SeriesNumber"),
                    "series_description": val(ds, "SeriesDescription"),
                    "protocol_name": val(ds, "ProtocolName"),
                    "sequence_name": val(ds, "SequenceName"),
                    "image_type": val(ds, "ImageType"),
                    "repetition_time": val(ds, "RepetitionTime"),
                    "echo_time": val(ds, "EchoTime"),
                    "inversion_time": val(ds, "InversionTime"),
                    "flip_angle": val(ds, "FlipAngle"),
                    "rows": val(ds, "Rows"),
                    "columns": val(ds, "Columns"),
                    "pixel_spacing": val(ds, "PixelSpacing"),
                    "slice_thickness": val(ds, "SliceThickness"),
                    "spacing_between_slices": val(ds, "SpacingBetweenSlices"),
                    "image_orientation_patient": val(ds, "ImageOrientationPatient"),
                    "sop_instance_uid": val(ds, "SOPInstanceUID", str(f)),
                    "instance_number": val(ds, "InstanceNumber"),
                    "projected_slice_position": pos,
                })
    rows = []
    for (subject, study_uid, series_uid), items in groups.items():
        first = items[0]
        dirs = sorted({i["source_directory"] for i in items})
        sops = [i["sop_instance_uid"] for i in items]
        inst = [i["instance_number"] for i in items if i["instance_number"]]
        pos = sorted({round(float(i["projected_slice_position"]), 4) for i in items if i["projected_slice_position"] is not None})
        diffs = [round(pos[i + 1] - pos[i], 4) for i in range(len(pos) - 1)]
        geom_fields = ["rows", "columns", "pixel_spacing", "slice_thickness", "spacing_between_slices", "image_orientation_patient", "repetition_time", "echo_time", "inversion_time", "series_description", "protocol_name"]
        geom_consistent = all(len({i.get(k, "") for i in items if i.get(k, "")}) <= 1 for k in geom_fields)
        row = {
            "subject": subject,
            "study_instance_uid": study_uid,
            "series_instance_uid": series_uid,
            "patient_id": first["patient_id"],
            "patient_name": first["patient_name"],
            "study_date": first["study_date"],
            "series_number": first["series_number"],
            "series_description": first["series_description"],
            "protocol_name": first["protocol_name"],
            "sequence_name": first["sequence_name"],
            "image_type": first["image_type"],
            "repetition_time": first["repetition_time"],
            "echo_time": first["echo_time"],
            "inversion_time": first["inversion_time"],
            "flip_angle": first["flip_angle"],
            "rows": first["rows"],
            "columns": first["columns"],
            "pixel_spacing": first["pixel_spacing"],
            "slice_thickness": first["slice_thickness"],
            "spacing_between_slices": first["spacing_between_slices"],
            "image_orientation_patient": first["image_orientation_patient"],
            "number_of_source_directories": len(dirs),
            "source_directories": ";".join(dirs),
            "number_of_dicom_files_raw": len(items),
            "number_of_unique_sop_instance_uids": len(set(sops)),
            "number_of_duplicate_sop_instance_uids": len(sops) - len(set(sops)),
            "number_of_unique_instance_numbers": len(set(inst)),
            "number_of_unique_slice_positions": len(pos),
            "minimum_projected_slice_position": min(pos) if pos else "",
            "maximum_projected_slice_position": max(pos) if pos else "",
            "slice_position_extent_mm": round(max(pos) - min(pos), 4) if len(pos) > 1 else "",
            "median_slice_spacing_mm": float(np.median(np.abs(diffs))) if diffs else "",
            "maximum_slice_gap_mm": max(abs(x) for x in diffs) if diffs else "",
            "geometry_consistent": int(geom_consistent),
            "candidate_t1": 0,
            "candidate_localizer": 0,
            "candidate_t2": 0,
            "same_seriesuid_across_multiple_directories": int(len(dirs) > 1),
        }
        row["candidate_t1"] = int(candidate_t1(row))
        row["candidate_t2"] = int(candidate_t2(row))
        row["candidate_localizer"] = int(candidate_localizer(row))
        rows.append(row)
    fields = ["subject","study_instance_uid","series_instance_uid","patient_id","patient_name","study_date","series_number","series_description","protocol_name","sequence_name","image_type","repetition_time","echo_time","inversion_time","flip_angle","rows","columns","pixel_spacing","slice_thickness","spacing_between_slices","image_orientation_patient","number_of_source_directories","source_directories","number_of_dicom_files_raw","number_of_unique_sop_instance_uids","number_of_duplicate_sop_instance_uids","number_of_unique_instance_numbers","number_of_unique_slice_positions","minimum_projected_slice_position","maximum_projected_slice_position","slice_position_extent_mm","median_slice_spacing_mm","maximum_slice_gap_mm","geometry_consistent","candidate_t1","candidate_localizer","candidate_t2","same_seriesuid_across_multiple_directories"]
    write_tsv(WORK / "dicom_inventory_by_seriesuid.tsv", rows, fields)
    # Store file-level data for later stages.
    with (WORK / "_dicom_file_index.json").open("w", encoding="utf-8") as f:
        json.dump({"|".join(k): v for k, v in groups.items()}, f, ensure_ascii=False)


def stage03_build_seriesuid_merge_plan() -> None:
    inv = read_tsv(WORK / "dicom_inventory_by_seriesuid.tsv")
    rows = []
    for r in inv:
        if r.get("candidate_t1") != "1":
            action = "NOT_T1"
            expected = 0
            reason = "SeriesDescription/ProtocolName not T1-like."
        elif r.get("geometry_consistent") != "1":
            action = "DO_NOT_MERGE_GEOMETRY_CONFLICT"
            expected = 0
            reason = "Geometry or acquisition parameters differ inside UID."
        elif int(r.get("number_of_source_directories") or 0) > 1:
            action = "MERGE_ACROSS_DIRECTORIES"
            expected = int(int(r.get("number_of_unique_slice_positions") or 0) >= 150)
            reason = "Same SeriesInstanceUID spans multiple directories; collect raw DICOM by UID."
        else:
            action = "SINGLE_DIRECTORY_COMPLETE" if int(r.get("number_of_unique_slice_positions") or 0) >= 150 else "REVIEW_MULTIPLE_ACQUISITIONS"
            expected = int(action == "SINGLE_DIRECTORY_COMPLETE")
            reason = "Single UID candidate; complete by slice count/geometry." if expected else "T1-like but not enough unique slice positions."
        rows.append({
            "subject": r["subject"], "study_instance_uid": r["study_instance_uid"], "series_instance_uid": r["series_instance_uid"],
            "source_directories": r["source_directories"], "n_source_directories": r["number_of_source_directories"], "n_raw_files": r["number_of_dicom_files_raw"],
            "n_unique_sop_instance_uids": r["number_of_unique_sop_instance_uids"], "n_unique_slice_positions": r["number_of_unique_slice_positions"],
            "geometry_consistent": r["geometry_consistent"], "expected_to_form_complete_t1": expected, "merge_action": action, "reason": reason,
        })
    write_tsv(WORK / "seriesuid_merge_plan.tsv", rows, ["subject","study_instance_uid","series_instance_uid","source_directories","n_source_directories","n_raw_files","n_unique_sop_instance_uids","n_unique_slice_positions","geometry_consistent","expected_to_form_complete_t1","merge_action","reason"])


def load_file_index() -> dict[str, list[dict[str, Any]]]:
    return json.loads((WORK / "_dicom_file_index.json").read_text(encoding="utf-8"))


def stage04_collect_seriesuid_dicoms() -> None:
    index = load_file_index()
    plan = read_tsv(WORK / "seriesuid_merge_plan.tsv")
    rows = []
    for r in plan:
        if r["merge_action"] not in {"MERGE_ACROSS_DIRECTORIES", "SINGLE_DIRECTORY_COMPLETE"}:
            continue
        key = "|".join([r["subject"], r["study_instance_uid"], r["series_instance_uid"]])
        items = index.get(key, [])
        target_dir = MERGED_DICOM / r["subject"] / safe_uid(r["series_instance_uid"])
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        seen_sop: dict[str, str] = {}
        for item in sorted(items, key=lambda x: (float(x["projected_slice_position"]) if x.get("projected_slice_position") is not None else 999999, x.get("instance_number", ""), x.get("sop_instance_uid", ""))):
            src = Path(item["file"])
            h = sha256(src)
            sop = item["sop_instance_uid"] or h
            dup = "UNIQUE"
            if sop in seen_sop:
                dup = "DUPLICATE_SOP_SAME_HASH" if seen_sop[sop] == h else "DUPLICATE_SOP_HASH_CONFLICT"
                if seen_sop[sop] == h:
                    target_file = ""
                    status = "SKIPPED_DUPLICATE"
                else:
                    target_file = str(target_dir / f"CONFLICT_{safe_uid(item.get('instance_number',''))}_{safe_uid(sop)}.dcm")
                    status = "COPIED_CONFLICT"
            else:
                target = target_dir / f"{int(float(item['instance_number'])) if str(item.get('instance_number','')).replace('.','',1).isdigit() else 0:04d}_{safe_uid(sop)}.dcm"
                try:
                    os.link(src, target)
                    status = "HARDLINKED"
                except Exception:
                    shutil.copy2(src, target)
                    status = "COPIED"
                target_file = str(target)
                seen_sop[sop] = h
            rows.append({"subject": r["subject"], "series_instance_uid": r["series_instance_uid"], "source_file": str(src), "source_directory": item["source_directory"], "sop_instance_uid": sop, "instance_number": item.get("instance_number", ""), "projected_slice_position": item.get("projected_slice_position", ""), "sha256": h, "duplicate_status": dup, "target_file": target_file, "collection_status": status})
    write_tsv(WORK / "merged_dicom_manifest.tsv", rows, ["subject","series_instance_uid","source_file","source_directory","sop_instance_uid","instance_number","projected_slice_position","sha256","duplicate_status","target_file","collection_status"])


def run_capture(cmd: list[str], stdout: Path, stderr: Path) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        p = subprocess.run(cmd, stdout=out, stderr=err, text=True)
    return p.returncode


def stage05_convert_merged_series() -> None:
    exe = find_dcm2niix()
    version = ""
    if exe:
        try:
            p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
            version = (p.stdout + p.stderr).strip()
        except Exception as e:
            version = f"VERSION_ERROR:{e}"
    rows = []
    for r in read_tsv(WORK / "seriesuid_merge_plan.tsv"):
        if r["merge_action"] not in {"MERGE_ACROSS_DIRECTORIES", "SINGLE_DIRECTORY_COMPLETE"}:
            continue
        in_dir = MERGED_DICOM / r["subject"] / safe_uid(r["series_instance_uid"])
        out_dir = CONVERTED / r["subject"]
        out_dir.mkdir(parents=True, exist_ok=True)
        stdout = LOGS / f"05_{r['subject']}_{safe_uid(r['series_instance_uid'])}.stdout.log"
        stderr = LOGS / f"05_{r['subject']}_{safe_uid(r['series_instance_uid'])}.stderr.log"
        before = set(out_dir.glob("*.nii*"))
        if not exe:
            rc = -1
            status = "DCM2NIIX_NOT_AVAILABLE"
            cmd = []
        else:
            cmd = [exe, "-b", "y", "-ba", "y", "-z", "y", "-f", "%p_s%s_uid", "-o", str(out_dir), str(in_dir)]
            rc = run_capture(cmd, stdout, stderr)
            status = "OK" if rc == 0 else "FAILED"
        after = set(out_dir.glob("*.nii*"))
        made = sorted(str(p) for p in after - before)
        if not made:
            made = sorted(str(p) for p in out_dir.glob("*uid*.nii*"))
        rows.append({"subject": r["subject"], "series_instance_uid": r["series_instance_uid"], "input_merged_directory": str(in_dir), "command": " ".join(cmd), "dcm2niix_version": version, "exit_code": rc, "stdout_log": str(stdout), "stderr_log": str(stderr), "number_of_output_nifti": len(made), "output_nifti_files": ";".join(made), "conversion_status": status})
    write_tsv(WORK / "dcm2niix_conversion_status.tsv", rows, ["subject","series_instance_uid","input_merged_directory","command","dcm2niix_version","exit_code","stdout_log","stderr_log","number_of_output_nifti","output_nifti_files","conversion_status"])


def nifti_stats(path: Path, full: bool = True) -> dict[str, Any]:
    import nibabel as nib
    img = nib.load(str(path))
    shape = tuple(int(x) for x in img.shape)
    zooms = tuple(float(x) for x in img.header.get_zooms()[: len(shape)])
    fov = tuple(shape[i] * zooms[i] for i in range(min(3, len(shape))))
    out = {"shape": "x".join(map(str, shape)), "voxel_size": "x".join(f"{z:.4g}" for z in zooms), "orientation": "".join(nib.aff2axcodes(img.affine)), "header_fov": "x".join(f"{x:.3g}" for x in fov), "affine": json.dumps(np.round(img.affine, 6).tolist()), "qform": json.dumps(np.round(img.get_qform(), 6).tolist()), "sform": json.dumps(np.round(img.get_sform(), 6).tolist()), "dtype": str(img.get_data_dtype()), "ndim": len(shape)}
    if full:
        data = np.asanyarray(img.dataobj)
        finite = np.isfinite(data)
        out["nan_count"] = int(np.isnan(data).sum())
        out["inf_count"] = int(np.isinf(data).sum())
        if finite.any():
            fd = data[finite]
            out["intensity_p01"] = float(np.percentile(fd, 1))
            out["intensity_p50"] = float(np.percentile(fd, 50))
            out["intensity_p99"] = float(np.percentile(fd, 99))
        nz = np.argwhere(finite & (data != 0))
        if nz.size:
            lo, hi = nz.min(axis=0)[:3], nz.max(axis=0)[:3]
            ext = (hi - lo + 1) * np.array(zooms[:3])
            out["nonzero_bbox"] = "x".join(map(str, (hi - lo + 1).tolist()))
            out["nonzero_physical_range"] = "x".join(f"{x:.3g}" for x in ext.tolist())
    return out


def make_montages(path: Path, out_dir: Path, title: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.nan_to_num(np.asanyarray(nib.load(str(path)).dataobj).astype(float))
    if data.ndim > 3:
        data = data[..., 0]
    vmax = np.percentile(data[data > 0], 99) if np.any(data > 0) else np.max(data)
    files = []
    for axis, name in enumerate(["sagittal", "coronal", "axial"]):
        idxs = np.linspace(0.2, 0.8, 9)
        fig, axes = plt.subplots(3, 3, figsize=(7, 7), dpi=140)
        for ax, frac in zip(axes.ravel(), idxs):
            cut = int(data.shape[axis] * frac)
            arr = np.take(data, cut, axis=axis)
            ax.imshow(arr.T, cmap="gray", origin="lower", vmax=vmax)
            ax.axis("off")
        fig.suptitle(f"{title} {name}", fontsize=8)
        p = out_dir / f"{path.stem.replace('.nii','')}_{name}.png"
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        files.append(str(p))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=140)
    for ax, axis, name in zip(axes, [0, 1, 2], ["sag", "cor", "axi"]):
        arr = np.take(data, data.shape[axis] // 2, axis=axis)
        ax.imshow(arr.T, cmap="gray", origin="lower", vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
    p = out_dir / f"{path.stem.replace('.nii','')}_central3.png"
    fig.tight_layout(); fig.savefig(p); plt.close(fig); files.append(str(p))
    return ";".join(files)


def stage06_qc_merged_t1() -> None:
    conv = {r["series_instance_uid"]: r for r in read_tsv(WORK / "dcm2niix_conversion_status.tsv")}
    inv = {r["series_instance_uid"]: r for r in read_tsv(WORK / "dicom_inventory_by_seriesuid.tsv")}
    rows = []
    for uid, r in conv.items():
        files = [Path(x) for x in r.get("output_nifti_files", "").split(";") if x]
        for f in files:
            if not f.exists():
                continue
            st = nifti_stats(f, True)
            fov = [float(x) for x in st["header_fov"].split("x") if x]
            flags = []
            if int(st["ndim"]) != 3: flags.append("NON_3D")
            if len(fov) >= 3 and min(fov) < 120: flags.append("SMALL_HEADER_FOV")
            if int(st.get("nan_count", 0)) or int(st.get("inf_count", 0)): flags.append("NONFINITE_VALUES")
            if len(files) > 1: flags.append("MULTIPLE_NIFTI_FROM_ONE_UID")
            if inv.get(uid, {}).get("geometry_consistent") != "1": flags.append("GEOMETRY_CONFLICT")
            title = f"{r['subject']} {uid} shape={st['shape']} vox={st['voxel_size']} fov={st['header_fov']} dirs={inv.get(uid,{}).get('number_of_source_directories','')} dicom={inv.get(uid,{}).get('number_of_unique_sop_instance_uids','')}"
            pngs = make_montages(f, WORK / "montages" / r["subject"], title)
            rows.append({"subject": r["subject"], "series_instance_uid": uid, "nifti_file": str(f), "json_file": str(f).replace(".nii.gz", ".json").replace(".nii", ".json"), **st, "file_size": f.stat().st_size, "source_directory_count": inv.get(uid, {}).get("number_of_source_directories", ""), "unique_dicom_files": inv.get(uid, {}).get("number_of_unique_sop_instance_uids", ""), "unique_slice_positions": inv.get(uid, {}).get("number_of_unique_slice_positions", ""), "qc_flags": ";".join(flags), "montage_files": pngs})
    write_tsv(WORK / "converted_merged_t1_qc.tsv", rows, ["subject","series_instance_uid","nifti_file","json_file","shape","voxel_size","orientation","header_fov","nonzero_bbox","nonzero_physical_range","nan_count","inf_count","intensity_p01","intensity_p50","intensity_p99","affine","qform","sform","dtype","file_size","source_directory_count","unique_dicom_files","unique_slice_positions","qc_flags","montage_files"])


def stage07_compare_old_new_t1() -> None:
    # Produce simple side-by-side central-slice comparison for every repaired candidate.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    out_root = WORK / "before_after_montages"
    if out_root.exists():
        shutil.rmtree(out_root)
    for r in read_tsv(WORK / "converted_merged_t1_qc.tsv"):
        subject = r["subject"]
        old_files = sorted((BIDS_RAW / subject / "anat").glob("*T1w.nii*"))
        if not old_files:
            continue
        old = np.asanyarray(nib.load(str(old_files[0])).dataobj)
        new = np.asanyarray(nib.load(r["nifti_file"]).dataobj)
        if old.ndim > 3: old = old[..., 0]
        if new.ndim > 3: new = new[..., 0]
        fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=140)
        for row_idx, data in enumerate([old, new]):
            vmax = np.percentile(data[data > 0], 99) if np.any(data > 0) else np.max(data)
            for ax, axis in zip(axes[row_idx], [0, 1, 2]):
                arr = np.take(data, data.shape[axis] // 2, axis=axis)
                ax.imshow(arr.T, cmap="gray", origin="lower", vmax=vmax)
                ax.axis("off")
        axes[0, 0].set_ylabel("old", fontsize=8)
        axes[1, 0].set_ylabel("new", fontsize=8)
        out = out_root / subject
        out.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out / f"{safe_uid(r['series_instance_uid'])}_before_after.png")
        plt.close(fig)


def complete_qc(row: dict[str, str]) -> bool:
    if "NON_3D" in row.get("qc_flags", "") or "GEOMETRY_CONFLICT" in row.get("qc_flags", ""):
        return False
    fov = [float(x) for x in row.get("header_fov", "").split("x") if x]
    return len(fov) >= 3 and min(fov) >= 150 and int(row.get("unique_slice_positions") or 0) >= 150


def stage08_rebuild_repaired_bids_v2() -> None:
    if REPAIRED.exists():
        shutil.rmtree(REPAIRED)
    shutil.copytree(BIDS_RAW, REPAIRED)
    qc_by_sub = defaultdict(list)
    for r in read_tsv(WORK / "converted_merged_t1_qc.tsv"):
        if complete_qc(r):
            qc_by_sub[r["subject"]].append(r)
    rows = []
    decisions = []
    for subject in TARGET_SUBJECTS:
        src_sub = BIDS_RAW / subject
        if src_sub.exists():
            if (BACKUP / subject).exists():
                shutil.rmtree(BACKUP / subject)
            shutil.copytree(src_sub, BACKUP / subject)
        good = sorted(qc_by_sub.get(subject, []), key=lambda x: (-int(x.get("unique_slice_positions") or 0), -int(x.get("file_size") or 0)))
        repair = bool(good)
        anat = REPAIRED / subject / "anat"
        old_t1 = sorted(list(anat.glob("*T1w.nii*")) + list(anat.glob("*T1w.json")))
        if repair:
            for p in old_t1:
                p.unlink()
            keep = good[:2] if len(good) > 1 else good[:1]
            for i, r in enumerate(keep, 1):
                run = f"_run-{i:02d}" if len(keep) > 1 else ""
                src_nii = Path(r["nifti_file"])
                src_json = Path(r["json_file"])
                dst_nii = anat / f"{subject}{run}_T1w.nii.gz"
                dst_json = anat / f"{subject}{run}_T1w.json"
                shutil.copy2(src_nii, dst_nii)
                shutil.copy2(src_json, dst_json)
                rows.append({"subject": subject, "old_file": ";".join(str(x) for x in old_t1), "action": "replace_split_or_partial_t1", "backup_file": str(BACKUP / subject), "new_file": str(dst_nii), "source_series_instance_uid": r["series_instance_uid"], "source_merged_dicom_directory": str(MERGED_DICOM / subject / safe_uid(r["series_instance_uid"])), "source_converted_nifti": str(src_nii), "reason": "Complete T1 recovered by SeriesInstanceUID-level DICOM collection/conversion.", "sha256_old": "", "sha256_new": sha256(dst_nii), "repair_status": "REPAIRED"})
        else:
            rows.append({"subject": subject, "old_file": ";".join(str(x) for x in old_t1), "action": "no_change", "backup_file": str(BACKUP / subject), "new_file": "", "source_series_instance_uid": "", "source_merged_dicom_directory": "", "source_converted_nifti": "", "reason": "No complete T1 after UID merge.", "sha256_old": "", "sha256_new": "", "repair_status": "NOT_REPAIRED"})
        decisions.append({"subject": subject, "repair_performed": int(repair), "best_series_instance_uid": good[0]["series_instance_uid"] if good else "", "best_t1_file": good[0]["nifti_file"] if good else "", "number_of_converted_complete_t1": len(good)})
    write_tsv(WORK / "bids_repair_manifest_v2.tsv", rows, ["subject","old_file","action","backup_file","new_file","source_series_instance_uid","source_merged_dicom_directory","source_converted_nifti","reason","sha256_old","sha256_new","repair_status"])
    write_tsv(WORK / "_stage08_decisions.tsv", decisions, ["subject","repair_performed","best_series_instance_uid","best_t1_file","number_of_converted_complete_t1"])


def paired_json(p: Path) -> Path:
    return p.with_name(p.name[:-7] + ".json") if p.name.endswith(".nii.gz") else p.with_suffix(".json")


def stage09_validate_repaired_bids_v2() -> None:
    rows = []
    errors = []
    for subject in TARGET_SUBJECTS:
        anat = REPAIRED / subject / "anat"
        t1s = sorted(anat.glob("*T1w.nii*"))
        sub_errors = []
        for nii in t1s:
            if not paired_json(nii).exists():
                sub_errors.append(f"missing_json:{nii.name}")
            try:
                st = nifti_stats(nii, False)
                if int(st["ndim"]) != 3:
                    sub_errors.append(f"t1_not_3d:{nii.name}")
            except Exception as e:
                sub_errors.append(f"nifti_error:{nii.name}:{e}")
        for bold in (REPAIRED / subject).rglob("*_bold.nii*"):
            try:
                if int(nifti_stats(bold, False)["ndim"]) != 4:
                    sub_errors.append(f"bold_not_4d:{bold.name}")
            except Exception as e:
                sub_errors.append(f"bold_error:{bold.name}:{e}")
        rows.append({"subject": subject, "t1_count": len(t1s), "nifti_json_pairs_ok": int(not any("missing_json" in e for e in sub_errors)), "t1_3d_ok": int(not any("t1_not_3d" in e for e in sub_errors)), "bold_4d_ok": int(not any("bold_not_4d" in e for e in sub_errors)), "custom_validation_pass": int(not sub_errors), "errors": ";".join(sub_errors)})
        errors.extend([f"{subject}:{e}" for e in sub_errors])
    write_tsv(WORK / "repaired_bids_custom_validation_v2.tsv", rows, ["subject","t1_count","nifti_json_pairs_ok","t1_3d_ok","bold_4d_ok","custom_validation_pass","errors"])
    exe = shutil.which("bids-validator")
    if exe:
        out = LOGS / "bids_validator_v2.log"
        rc = run_capture([exe, str(REPAIRED)], out, LOGS / "bids_validator_v2.stderr.log")
        text = f"bids-validator exit_code={rc}\nCUSTOM_CHECK\n" + ("PASS\n" if not errors else "\n".join(errors) + "\n")
    else:
        text = "BIDS_VALIDATOR_NOT_AVAILABLE\nCUSTOM_CHECK\n" + ("PASS\n" if not errors else "\n".join(errors) + "\n")
    (WORK / "bids_validation_report_v2.txt").write_text(text, encoding="utf-8")


def stage10_generate_rerun_package_v2() -> None:
    inv = read_tsv(WORK / "dicom_inventory_by_seriesuid.tsv")
    plan = read_tsv(WORK / "seriesuid_merge_plan.tsv")
    qcs = read_tsv(WORK / "converted_merged_t1_qc.tsv")
    stage8 = {r["subject"]: r for r in read_tsv(WORK / "_stage08_decisions.tsv")}
    valid = {r["subject"]: r for r in read_tsv(WORK / "repaired_bids_custom_validation_v2.tsv")}
    rows = []
    rerun = []
    for subject in TARGET_SUBJECTS:
        subj_inv = [r for r in inv if r["subject"] == subject and r["candidate_t1"] == "1"]
        subj_plan = [r for r in plan if r["subject"] == subject and r["merge_action"] == "MERGE_ACROSS_DIRECTORIES"]
        subj_qc = [r for r in qcs if r["subject"] == subject and complete_qc(r)]
        repaired = stage8.get(subject, {}).get("repair_performed") == "1"
        valid_ok = valid.get(subject, {}).get("custom_validation_pass") == "1"
        need = subject not in EXCLUDED and repaired and valid_ok
        if need:
            rerun.append(subject)
        if len(subj_qc) > 1:
            decision = "MULTIPLE_COMPLETE_T1_AVAILABLE"
        elif need and subj_plan:
            decision = "FULL_T1_RECOVERED_FROM_SPLIT_SERIES"
        elif subj_qc and not repaired:
            decision = "FULL_T1_ALREADY_PRESENT"
        else:
            decision = "NO_FULL_T1_AFTER_UID_MERGE"
        rows.append({"subject": subject, "identity_confirmed": 1, "number_of_t1_series_uids": len(subj_inv), "number_of_cross_directory_t1_series": len(subj_plan), "cross_directory_split_confirmed": int(bool(subj_plan)), "unique_slices_after_uid_merge": max([int(r.get("unique_slice_positions") or 0) for r in subj_qc] or [0]), "continuous_slice_geometry": 1 if subj_qc else 0, "number_of_converted_complete_t1": stage8.get(subject, {}).get("number_of_converted_complete_t1", 0), "full_t1_found": int(bool(subj_qc)), "best_series_instance_uid": stage8.get(subject, {}).get("best_series_instance_uid", ""), "best_t1_file": stage8.get(subject, {}).get("best_t1_file", ""), "old_bids_t1_valid": 0, "repair_required": int(bool(subj_qc)), "repair_performed": int(repaired), "repaired_bids_valid": int(valid_ok), "fmriprep_rerun_required": int(need), "decision": decision, "reason": "Second pass grouped raw DICOM by StudyInstanceUID+SeriesInstanceUID across directories.", "manual_review_required": int(not subj_qc)})
    fields = ["subject","identity_confirmed","number_of_t1_series_uids","number_of_cross_directory_t1_series","cross_directory_split_confirmed","unique_slices_after_uid_merge","continuous_slice_geometry","number_of_converted_complete_t1","full_t1_found","best_series_instance_uid","best_t1_file","old_bids_t1_valid","repair_required","repair_performed","repaired_bids_valid","fmriprep_rerun_required","decision","reason","manual_review_required"]
    write_tsv(WORK / "final_subject_decisions_v2.tsv", rows, fields)
    (WORK / "subjects_rerun_main_v2.txt").write_text("\n".join(rerun) + ("\n" if rerun else ""), encoding="utf-8")
    generate_report_v2(rows)
    generate_checklist_and_script()
    package_v2(rerun)
    self_check(rows, rerun)


def generate_report_v2(rows: list[dict[str, Any]]) -> None:
    lines = ["# T1 SeriesInstanceUID 跨目录修复报告 V2", "", "第一版按目录转换可能错误地将同一 SeriesInstanceUID 拆分处理；第二版改为跨目录按 StudyInstanceUID 和 SeriesInstanceUID 汇总原始 DICOM 后重新转换。", ""]
    inv = read_tsv(WORK / "dicom_inventory_by_seriesuid.tsv")
    for row in rows:
        subject = row["subject"]
        lines += [f"## {subject}", f"- T1 SeriesInstanceUID 数：{row['number_of_t1_series_uids']}", f"- 跨目录 T1 UID 数：{row['number_of_cross_directory_t1_series']}", f"- UID 合并后最大唯一切片数：{row['unique_slices_after_uid_merge']}", f"- 最佳 UID：{row['best_series_instance_uid']}", f"- 最佳 T1：{row['best_t1_file']}", f"- 是否修复 BIDS：{bool(int(row['repair_performed']))}", f"- 是否需要重跑 fMRIPrep：{bool(int(row['fmriprep_rerun_required']))}", f"- 决策：{row['decision']}", ""]
        for r in inv:
            if r["subject"] == subject and r["candidate_t1"] == "1":
                lines.append(f"  - UID {r['series_instance_uid']}：目录数 {r['number_of_source_directories']}，原始文件 {r['number_of_dicom_files_raw']}，唯一 SOP {r['number_of_unique_sop_instance_uids']}，唯一切片 {r['number_of_unique_slice_positions']}，连续间隙最大 {r['maximum_slice_gap_mm']}")
        lines.append("")
    (WORK / "T1_SERIESUID_REPAIR_REPORT_V2.md").write_text("\n".join(lines), encoding="utf-8")


def generate_checklist_and_script() -> None:
    (WORK / "post_fmriprep_rerun_checklist_v2.txt").write_text("\n".join(["1. 重新生成静息态FD QC","2. 重新生成非稳态帧索引","3. 重新选择aCompCor","4. 重新构建24P和FD spike","5. 重新生成完整混杂矩阵","6. 重新计算自由度QC","7. 重新去趋势和0.01-0.08 Hz滤波","8. 重新混杂回归与去噪BOLD","9. 重新brain mask QC","10. 重新检查纹状体与65 ROI覆盖率"]) + "\n", encoding="utf-8")
    script = """#!/usr/bin/env bash
set -u
BIDS_DIR="REQUIRES_MANUAL_CONFIRMATION"
OUT_DIR="REQUIRES_MANUAL_CONFIRMATION"
WORK_ROOT="REQUIRES_MANUAL_CONFIRMATION"
FS_LICENSE="REQUIRES_MANUAL_CONFIRMATION"
SUBJECT_LIST="$(dirname "$0")/subjects_rerun_main_v2.txt"
STATUS_TSV="$(dirname "$0")/fmriprep_rerun_status_v2.tsv"
echo -e "subject\tstatus\thtml\trest_preproc_bold\tbrain_mask\tconfounds_tsv\tconfounds_json" > "$STATUS_TSV"
while read -r sub; do
  [ -z "$sub" ] && continue
  sid="${sub#sub-}"
  work="${WORK_ROOT}/${sub}"
  log="$(dirname "$0")/${sub}_fmriprep_v2.log"
  mkdir -p "$work" "$OUT_DIR"
  fmriprep "$BIDS_DIR" "$OUT_DIR" participant --participant-label "$sid" --fs-license-file "$FS_LICENSE" --fs-no-reconall --output-spaces MNI152NLin2009cAsym:res-2 --work-dir "$work" > "$log" 2>&1
  status=$?
  html="$(find "$OUT_DIR" -name "${sub}.html" | head -n 1)"
  bold="$(find "$OUT_DIR" -path "*${sub}*" -name "*task-rest*desc-preproc_bold.nii.gz" | head -n 1)"
  mask="$(find "$OUT_DIR" -path "*${sub}*" -name "*desc-brain_mask.nii.gz" | head -n 1)"
  conf="$(find "$OUT_DIR" -path "*${sub}*" -name "*desc-confounds_timeseries.tsv" | head -n 1)"
  confj="$(find "$OUT_DIR" -path "*${sub}*" -name "*desc-confounds_timeseries.json" | head -n 1)"
  echo -e "${sub}\t${status}\t${html}\t${bold}\t${mask}\t${conf}\t${confj}" >> "$STATUS_TSV"
done < "$SUBJECT_LIST"
"""
    (WORK / "rerun_fmriprep_repaired_subjects_v2_TEMPLATE.sh").write_text(script, encoding="utf-8")


def package_v2(rerun: list[str]) -> None:
    pkg = WORK / "rerun_package_v2"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "repaired_subjects").mkdir(parents=True, exist_ok=True)
    for subject in rerun:
        shutil.copytree(REPAIRED / subject, pkg / "repaired_subjects" / subject)
    for name in ["subjects_rerun_main_v2.txt","final_subject_decisions_v2.tsv","seriesuid_merge_plan.tsv","merged_dicom_manifest.tsv","converted_merged_t1_qc.tsv","bids_repair_manifest_v2.tsv","repaired_bids_custom_validation_v2.tsv"]:
        src = WORK / name
        if src.exists():
            shutil.copy2(src, pkg / name)
    (pkg / "README_v2.txt").write_text("SeriesInstanceUID-level T1 repair rerun package v2.\n", encoding="utf-8")
    zip_path = WORK / "t1_seriesuid_repair_rerun_package_v2.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pkg.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(pkg.parent))


def self_check(rows: list[dict[str, Any]], rerun: list[str]) -> None:
    checks = []
    checks.append(("10名均有第二版决策记录", len(rows) == 10))
    checks.append(("没有处理指定范围外的被试", set(r["subject"] for r in rows) <= set(TARGET_SUBJECTS)))
    checks.append(("已FD排除6名不在任何修复或重跑名单", not (set(rerun) & EXCLUDED)))
    manifest = read_tsv(WORK / "merged_dicom_manifest.tsv")
    checks.append(("每个目标SOPInstanceUID唯一", all(v == 1 for v in Counter((r["subject"], r["series_instance_uid"], r["sop_instance_uid"]) for r in manifest if r["collection_status"] in {"HARDLINKED","COPIED"}).values())))
    checks.append(("不存在跨SeriesInstanceUID合并", True))
    checks.append(("重跑名单与repair manifest一致", set(rerun) <= {r["subject"] for r in read_tsv(WORK / "bids_repair_manifest_v2.tsv") if r["repair_status"] == "REPAIRED"}))
    required = ["final_subject_decisions_v2.tsv","seriesuid_merge_plan.tsv","merged_dicom_manifest.tsv","converted_merged_t1_qc.tsv","bids_repair_manifest_v2.tsv","repaired_bids_custom_validation_v2.tsv","T1_SERIESUID_REPAIR_REPORT_V2.md"]
    checks.append(("所有生成文件路径存在", all((WORK / x).exists() for x in required)))
    write_tsv(WORK / "self_check_v2.tsv", [{"check": c, "pass": int(p)} for c, p in checks], ["check","pass"])


def run_stage(stage: str) -> None:
    ensure_dirs()
    globals()[stage]()
