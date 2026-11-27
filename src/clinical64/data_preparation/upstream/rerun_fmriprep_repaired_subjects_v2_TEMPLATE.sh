#!/usr/bin/env bash
set -u
BIDS_DIR="${STRIATAL_BIDS_DIR:?Configure the private fMRIPrep path}"
OUT_DIR="${STRIATAL_FMRIPREP_OUTPUT:?Configure the private fMRIPrep path}"
WORK_ROOT="${STRIATAL_WORK_ROOT:?Configure the private fMRIPrep path}"
FS_LICENSE="${STRIATAL_FS_LICENSE:?Configure the private fMRIPrep path}"
SUBJECT_LIST="${STRIATAL_T1_RERUN_LIST:?Set the private rerun list}"
STATUS_TSV="$OUT_DIR/fmriprep_rerun_status_v2.tsv"
mkdir -p "$OUT_DIR"
echo -e "subject	status	html	rest_preproc_bold	brain_mask	confounds_tsv	confounds_json" > "$STATUS_TSV"
while read -r sub; do
  [ -z "$sub" ] && continue
  sid="${sub#sub-}"
  work="${WORK_ROOT}/${sub}"
  log="$OUT_DIR/${sub}_fmriprep_v2.log"
  mkdir -p "$work" "$OUT_DIR"
  fmriprep "$BIDS_DIR" "$OUT_DIR" participant --participant-label "$sid" --fs-license-file "$FS_LICENSE" --fs-no-reconall --output-spaces MNI152NLin2009cAsym:res-2 --work-dir "$work" > "$log" 2>&1
  status=$?
  html="$(find "$OUT_DIR" -name "${sub}.html" | head -n 1)"
  bold="$(find "$OUT_DIR" -path "*${sub}*" -name "*task-rest*desc-preproc_bold.nii.gz" | head -n 1)"
  mask="$(find "$OUT_DIR" -path "*${sub}*" -name "*desc-brain_mask.nii.gz" | head -n 1)"
  conf="$(find "$OUT_DIR" -path "*${sub}*" -name "*desc-confounds_timeseries.tsv" | head -n 1)"
  confj="$(find "$OUT_DIR" -path "*${sub}*" -name "*desc-confounds_timeseries.json" | head -n 1)"
  echo -e "${sub}	${status}	${html}	${bold}	${mask}	${conf}	${confj}" >> "$STATUS_TSV"
done < "$SUBJECT_LIST"
