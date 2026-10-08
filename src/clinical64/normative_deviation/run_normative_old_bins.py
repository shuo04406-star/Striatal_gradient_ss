#!/usr/bin/env python3
from __future__ import annotations

import os, re, sys, json, math, hashlib, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore', category=RuntimeWarning)

BASE_VALUE = os.environ.get('STRIATAL_ANALYSIS_ROOT')
if not BASE_VALUE:
    raise RuntimeError('Set STRIATAL_ANALYSIS_ROOT to the private analysis root')
BASE = Path(BASE_VALUE).resolve()
OUT = Path(os.environ.get('OUT', str(BASE / '24_normative_G1_FC_deviation'))).resolve()
FC_ROOT_OVERRIDE = os.environ.get('FC_ROOT', '').strip()
COV_FILE_OVERRIDE = os.environ.get('COV_FILE', '').strip()
G1_FILE_OVERRIDE = os.environ.get('G1_FILE', '').strip()
MAP_FILE_OVERRIDE = os.environ.get('MAP_FILE', '').strip()
COMMON_IJK_OVERRIDE = os.environ.get('COMMON_IJK_FILE', '').strip()
ROI_LABEL_OVERRIDE = os.environ.get('ROI_LABEL_FILE', '').strip()
FC_SCALE = os.environ.get('FC_SCALE', 'auto').strip().lower()  # auto | r | z

N_COMMON = int(os.environ.get('N_COMMON', '2608'))
N_ROI = int(os.environ.get('N_ROI', '65'))
N_BINS = int(os.environ.get('N_BINS', '10'))
N_BOOT = int(os.environ.get('N_BOOT', '1000'))
SEED = int(os.environ.get('SEED', '20260911'))
ALPHA = float(os.environ.get('ALPHA', '0.05'))
ALLOW_2608_G1 = os.environ.get('ALLOW_2608_G1', '0').strip() == '1'

TABLE_DIR = OUT / 'tables'
FIG_DIR = OUT / 'figures'
LOG_DIR = OUT / 'logs'
for d in (OUT, TABLE_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def log(msg=''):
    print(msg, flush=True)


def sha256_file(path: Path, block=1024*1024):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_under(p: Path, root: Path):
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def norm_col(x):
    return re.sub(r'[^a-z0-9]+', '', str(x).lower())


def subject_from_text(x):
    if pd.notna(x):
        try:
            fx=float(x)
            if np.isfinite(fx) and abs(fx-round(fx))<1e-9:
                return int(round(fx))
        except Exception:
            pass
    s = str(x)
    m = re.search(r'(?i)sub[-_ ]*0*(\d+)', s)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r'\s*0*(\d+)\s*', s)
    if m:
        return int(m.group(1))
    return None


def fc_file_subject(path: Path):
    return subject_from_text(path.as_posix())


def load_numeric_array(path: Path):
    suf = ''.join(path.suffixes).lower()
    if suf.endswith('.npy'):
        return np.load(path, mmap_mode='r')
    if suf.endswith('.npz'):
        z = np.load(path, allow_pickle=False)
        candidates = []
        for k in z.files:
            a = z[k]
            if np.issubdtype(a.dtype, np.number):
                candidates.append((k, a))
        if not candidates:
            raise ValueError(f'No numeric array in {path}')
        candidates.sort(key=lambda kv: kv[1].size, reverse=True)
        return candidates[0][1]
    raise ValueError(f'Unsupported array file: {path}')


def probe_fc(path: Path):
    try:
        a = load_numeric_array(path)
        return tuple(a.shape)
    except Exception:
        return None


def keyword_score(text, pos=(), neg=()):
    t = text.lower()
    s = 0
    for kw, val in pos:
        if kw in t:
            s += val
    for kw, val in neg:
        if kw in t:
            s -= val
    return s


def discover_fc_files():
    log('='*110)
    log('STEP 1. DISCOVER 64 SUBJECT × 2608 × 65 FC MATRICES')
    log('='*110)
    root = Path(FC_ROOT_OVERRIDE).resolve() if FC_ROOT_OVERRIDE else BASE
    if not root.exists():
        raise FileNotFoundError(root)

    candidates = []
    for ext in ('*.npy', '*.npz'):
        for p in root.rglob(ext):
            if is_under(p, OUT):
                continue
            sid = fc_file_subject(p)
            if sid is None:
                continue
            shp = probe_fc(p)
            if shp in ((N_COMMON, N_ROI), (N_ROI, N_COMMON)):
                candidates.append((p.resolve(), sid, shp))

    if not candidates:
        raise RuntimeError(f'No per-subject FC arrays with shape {N_COMMON}×{N_ROI} found under {root}')

    # Count unique subjects for every ancestor. Deep coherent roots are preferred.
    anc = defaultdict(lambda: {'subs': set(), 'files': []})
    for p, sid, shp in candidates:
        cur = p.parent
        while True:
            if not is_under(cur, root):
                break
            anc[cur]['subs'].add(sid)
            anc[cur]['files'].append((p, sid, shp))
            if cur == root:
                break
            cur = cur.parent

    ranked = []
    for a, obj in anc.items():
        n = len(obj['subs'])
        if n < 60:
            continue
        rel_depth = len(a.relative_to(root).parts) if a != root else 0
        txt = a.as_posix().lower()
        score = n * 3 + rel_depth * 4
        if n == 64:
            score += 200
        score += keyword_score(txt, [
            ('current64', 80), ('375matched', 70), ('375_matched', 70),
            ('2608', 50), ('65roi', 30), ('fc', 25), ('connect', 15),
            ('final', 10), ('reanalysis', 10),
        ], [('old', 25), ('backup', 20), ('tmp', 15), ('test', 15)])
        ranked.append((score, n, rel_depth, a, obj['files']))

    if not ranked:
        raise RuntimeError('Found FC-shaped files, but no coherent subtree contains at least 60 unique subjects.')

    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    log('Top FC roots:')
    for r in ranked[:8]:
        log(f'  score={r[0]:4d} | Nsub={r[1]:2d} | {r[3]}')

    if FC_ROOT_OVERRIDE:
        chosen_root = root
        pool = [(p,s,sh) for p,s,sh in candidates if is_under(p, chosen_root)]
    else:
        chosen_root = ranked[0][3]
        pool = ranked[0][4]

    by_sub = defaultdict(list)
    for p, sid, shp in pool:
        txt = p.as_posix().lower()
        score = keyword_score(txt, [
            ('fc', 20), ('2608', 20), ('65', 5), ('fisher', 5), ('final', 5)
        ], [('gradient', 20), ('tmp', 10), ('test', 10), ('backup', 10)])
        try:
            mt = p.stat().st_mtime
        except Exception:
            mt = 0
        by_sub[sid].append((score, mt, p, shp))

    selected = {}
    for sid, arr in by_sub.items():
        arr.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected[sid] = arr[0][2]

    if len(selected) != 64:
        log(f'WARNING: selected root has {len(selected)} unique subjects, not 64.')
        if len(selected) < 60 or len(selected) > 70:
            raise RuntimeError(f'FC subject count unsafe: {len(selected)}. Set FC_ROOT explicitly.')

    log(f'Chosen FC root: {chosen_root}')
    log(f'Unique FC subjects: {len(selected)}')
    return chosen_root, dict(sorted(selected.items()))


def read_table(path: Path, nrows=None):
    # Fast deterministic separator choice first; fallback to sniffing.
    sep = '\t' if path.suffix.lower() in ('.tsv', '.txt') else ','
    try:
        return pd.read_csv(path, sep=sep, nrows=nrows)
    except Exception:
        return pd.read_csv(path, sep=None, engine='python', nrows=nrows)


def choose_col(columns, aliases):
    cmap = {norm_col(c): c for c in columns}
    for a in aliases:
        if norm_col(a) in cmap:
            return cmap[norm_col(a)]
    for c in columns:
        nc = norm_col(c)
        if any(norm_col(a) in nc for a in aliases):
            return c
    return None


def discover_covariates(fc_sids):
    log('\n' + '='*110)
    log('STEP 2. DISCOVER GROUP / AGE / SEX / MEAN FD TABLE')
    log('='*110)

    aliases = {
        'subject': ['subject', 'subjectid', 'participantid', 'participant', 'sub', 'id'],
        'group': ['group', 'diagnosis', 'dx', 'diagnosticgroup'],
        'age': ['age'],
        'sex': ['sex', 'gender'],
        'fd': ['meanfd', 'mean_fd', 'fdmean', 'meanframewisedisplacement'],
    }

    paths = [Path(COV_FILE_OVERRIDE).resolve()] if COV_FILE_OVERRIDE else []
    if not paths:
        for ext in ('*.tsv', '*.csv', '*.txt'):
            for p in BASE.rglob(ext):
                if is_under(p, OUT):
                    continue
                try:
                    if p.stat().st_size > 80 * 1024 * 1024:
                        continue
                except Exception:
                    continue
                paths.append(p.resolve())

    ranked = []
    for p in paths:
        try:
            h = read_table(p, nrows=8)
        except Exception:
            continue
        cols = {}
        for k,v in aliases.items():
            cols[k] = choose_col(h.columns, v)
        if not all(cols.values()):
            continue
        try:
            df = read_table(p)
        except Exception:
            continue
        subj_int = df[cols['subject']].map(subject_from_text)
        overlap = len(set(subj_int.dropna().astype(int)) & set(fc_sids))
        if overlap < min(50, len(fc_sids)):
            continue
        score = overlap * 10
        score += keyword_score(p.as_posix(), [
            ('covariate', 60), ('demographic', 40), ('subject', 20), ('stage', 10), ('final', 10), ('64', 5)
        ], [('backup', 10), ('old', 10)])
        ranked.append((score, overlap, p, cols, df))

    if not ranked:
        raise RuntimeError('Could not find a table containing subject, group, age, sex, and meanFD for the FC subjects. Set COV_FILE.')

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    log('Top covariate files:')
    for r in ranked[:8]:
        log(f'  score={r[0]:4d} | overlap={r[1]:2d} | {r[2]}')

    score, overlap, path, cols, df = ranked[0]
    out = pd.DataFrame({
        'subject_int': df[cols['subject']].map(subject_from_text),
        'group_raw': df[cols['group']],
        'age': pd.to_numeric(df[cols['age']], errors='coerce'),
        'sex_raw': df[cols['sex']],
        'meanFD': pd.to_numeric(df[cols['fd']], errors='coerce'),
    })
    out = out[out['subject_int'].isin(fc_sids)].copy()

    def norm_group(x):
        try:
            fx=float(x)
            if np.isfinite(fx) and abs(fx-round(fx))<1e-9:
                ix=int(round(fx))
                if ix==0: return 'HC'
                if ix==1: return 'SZ'
        except Exception:
            pass
        s = norm_col(x)
        if s in ('hc','healthycontrol','healthycontrols','control','controls','healthy','0'):
            return 'HC'
        if s in ('sz','scz','schizophrenia','patient','patients','1'):
            return 'SZ'
        if 'schizo' in s or s == 'scz':
            return 'SZ'
        if 'control' in s or 'healthy' in s:
            return 'HC'
        return np.nan

    out['group'] = out['group_raw'].map(norm_group)

    # Sex: only a nuisance covariate. Encode any stable binary coding as 0/1.
    sr = out['sex_raw']
    sn = sr.map(lambda x: norm_col(x) if pd.notna(x) else '')
    sex = pd.Series(np.nan, index=out.index, dtype=float)
    male_tokens = {'m','male','man','1male'}
    female_tokens = {'f','female','woman','0female'}
    sex[sn.isin(male_tokens)] = 1.0
    sex[sn.isin(female_tokens)] = 0.0
    unresolved = sex.isna() & sr.notna()
    if unresolved.any():
        vals = pd.to_numeric(sr[unresolved], errors='coerce')
        uniq = sorted(pd.Series(vals.dropna().unique()).tolist())
        if len(uniq) == 2:
            mapper = {uniq[0]: 0.0, uniq[1]: 1.0}
            sex.loc[unresolved] = vals.map(mapper)
    out['sex'] = sex

    # Collapse repeated stage rows; verify covariates are effectively invariant per subject.
    rows = []
    for sid, g in out.groupby('subject_int'):
        rec = {'subject_int': int(sid)}
        for c in ('group','age','sex','meanFD'):
            vv = g[c].dropna().unique()
            if len(vv) == 0:
                rec[c] = np.nan
            elif len(vv) == 1:
                rec[c] = vv[0]
            else:
                if c in ('age','meanFD') and np.nanmax(vv.astype(float)) - np.nanmin(vv.astype(float)) < 1e-8:
                    rec[c] = float(vv[0])
                else:
                    raise RuntimeError(f'Covariate {c} varies within subject {sid}: {vv[:10]}')
        rows.append(rec)
    out = pd.DataFrame(rows)
    out['subject'] = out['subject_int'].map(lambda x: f'sub-{int(x):04d}')
    out = out[['subject_int','subject','group','age','sex','meanFD']]

    missing = out[['group','age','sex','meanFD']].isna().any(axis=1)
    if missing.any():
        bad = out.loc[missing, ['subject','group','age','sex','meanFD']]
        raise RuntimeError('Missing covariates after parsing:\n' + bad.to_string(index=False))

    counts = out['group'].value_counts().to_dict()
    log(f'Chosen covariate file: {path}')
    log(f'Group counts: {counts}')
    if not ('HC' in counts and 'SZ' in counts):
        raise RuntimeError('Both HC and SZ must be present.')
    return path, out


def try_load_vector(path: Path):
    suf = ''.join(path.suffixes).lower()
    try:
        if suf.endswith(('.npy','.npz')):
            a = np.asarray(load_numeric_array(path))
            if a.ndim == 1:
                return a.astype(float).ravel()
            if a.ndim == 2:
                if 1 in a.shape:
                    return a.astype(float).ravel()
                # Choose a likely G1 column only if it is a narrow table.
                if a.shape[1] <= 10 and a.shape[0] >= 100:
                    return a[:,0].astype(float)
            return None
        if suf.endswith(('.tsv','.csv','.txt')):
            df = read_table(path)
            if len(df) < 100:
                return None
            c = choose_col(df.columns, ['g1','gradient1','gradient_1','firstgradient'])
            if c is not None:
                return pd.to_numeric(df[c], errors='coerce').to_numpy(float)
            nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(nums) == 1:
                return pd.to_numeric(df[nums[0]], errors='coerce').to_numpy(float)
    except Exception:
        return None
    return None


def g1_score(path: Path, n):
    txt = path.as_posix().lower()
    score = 0
    score += keyword_score(txt, [
        ('canonical', 90), ('375', 90), ('g1', 70), ('gradient1', 60), ('gradient_1', 60),
        ('group', 25), ('reference', 25), ('template', 20), ('aligned', 10), ('final', 10)
    ], [('current64', 80), ('scz', 80), ('sz', 50), ('patient', 50), ('subject', 30), ('individual', 30), ('backup', 10)])
    if n > N_COMMON:
        score += 35
    if 2700 <= n <= 3200:
        score += 20
    if n == N_COMMON:
        score += 5
    return score


def discover_g1():
    log('\n' + '='*110)
    log('STEP 3. DISCOVER FROZEN 375-SAMPLE CANONICAL G1')
    log('='*110)

    paths = [Path(G1_FILE_OVERRIDE).resolve()] if G1_FILE_OVERRIDE else []
    if not paths:
        for ext in ('*.npy','*.npz','*.tsv','*.csv','*.txt'):
            for p in BASE.rglob(ext):
                if is_under(p, OUT):
                    continue
                txt = p.as_posix().lower()
                if not any(k in txt for k in ('g1','gradient1','gradient_1','gradient')):
                    continue
                paths.append(p.resolve())

    ranked = []
    for p in paths:
        v = try_load_vector(p)
        if v is None:
            continue
        n = len(v)
        if not (N_COMMON <= n <= 5000):
            continue
        finite = np.isfinite(v)
        if finite.mean() < 0.95 or np.nanstd(v) <= 0:
            continue
        score = g1_score(p, n)
        ranked.append((score, n, p, np.asarray(v, float)))

    if not ranked:
        raise RuntimeError('No plausible canonical G1 vector found. Set G1_FILE to the frozen 375-sample G1 vector/table.')

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    log('Top G1 candidates:')
    for r in ranked[:10]:
        log(f'  score={r[0]:4d} | N={r[1]:4d} | {r[2]}')

    score, n, path, g1 = ranked[0]
    if score < 70 and not G1_FILE_OVERRIDE:
        raise RuntimeError('Top G1 candidate is not clearly a frozen 375/canonical G1. Set G1_FILE explicitly; refusing to guess.')

    log(f'Chosen canonical G1: {path}')
    log(f'Canonical G1 length: {len(g1)}')
    return path, g1


def probe_index_vector(path: Path):
    try:
        a = np.asarray(load_numeric_array(path)) if ''.join(path.suffixes).lower().endswith(('.npy','.npz')) else None
        if a is None:
            df = read_table(path)
            nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(nums) != 1:
                return None
            a = df[nums[0]].to_numpy()
        if a.ndim != 1:
            a = a.ravel()
        if len(a) != N_COMMON:
            return None
        if not np.all(np.isfinite(a)):
            return None
        ai = np.rint(a).astype(int)
        if np.max(np.abs(a - ai)) > 1e-6:
            return None
        return ai
    except Exception:
        return None


def discover_mapping(nfull):
    log('\n' + '='*110)
    log('STEP 4. MAP CANONICAL G1 TO 2608 COMMON VOXELS')
    log('='*110)

    if nfull == N_COMMON:
        log('Canonical G1 already has exactly 2608 entries. No row mapping required.')
        return None, np.arange(N_COMMON, dtype=int), 'direct_2608'

    paths = [Path(MAP_FILE_OVERRIDE).resolve()] if MAP_FILE_OVERRIDE else []
    if not paths:
        for ext in ('*.npy','*.npz','*.tsv','*.csv','*.txt'):
            for p in BASE.rglob(ext):
                if is_under(p, OUT):
                    continue
                txt = p.as_posix().lower()
                if not any(k in txt for k in ('map','mapping','row','index','idx','2608')):
                    continue
                paths.append(p.resolve())

    ranked = []
    for p in paths:
        v = probe_index_vector(p)
        if v is None:
            continue
        # Accept zero-based or one-based.
        base = None
        vv = None
        if v.min() >= 0 and v.max() < nfull and len(np.unique(v)) == N_COMMON:
            base = 'zero_based'
            vv = v
        elif v.min() >= 1 and v.max() <= nfull and len(np.unique(v)) == N_COMMON:
            base = 'one_based'
            vv = v - 1
        if vv is None:
            continue
        txt = p.as_posix().lower()
        score = keyword_score(txt, [
            ('mapping', 70), ('row', 40), ('2608', 40), (str(nfull), 35), ('current', 15), ('old', 10), ('exact', 20), ('final', 10)
        ], [('flat', 30), ('backup', 10), ('test', 10)])
        ranked.append((score, p, vv, base))

    if ranked:
        ranked.sort(key=lambda x: x[0], reverse=True)
        log('Top mapping candidates:')
        for r in ranked[:8]:
            log(f'  score={r[0]:4d} | {r[3]} | min={r[2].min()} max={r[2].max()} | {r[1]}')
        score, p, v, base = ranked[0]
        log(f'Chosen row mapping: {p}')
        return p, v.astype(int), base

    # Fallback: exact IJK matching if both canonical and common IJK arrays exist.
    old = discover_ijk(nfull, prefer=('old','canonical','375'))
    cur = discover_ijk(N_COMMON, prefer=('current','common','2608'))
    if old is not None and cur is not None:
        old_path, old_ijk = old
        cur_path, cur_ijk = cur
        lut = {tuple(map(int, xyz)): i for i, xyz in enumerate(old_ijk)}
        try:
            mapping = np.array([lut[tuple(map(int, xyz))] for xyz in cur_ijk], dtype=int)
        except KeyError as e:
            raise RuntimeError(f'IJK mapping failed; common coordinate absent from canonical IJK: {e}')
        if len(np.unique(mapping)) != N_COMMON:
            raise RuntimeError('IJK mapping is not one-to-one.')
        log(f'Built exact IJK mapping using:\n  canonical IJK: {old_path}\n  common IJK:    {cur_path}')
        return cur_path, mapping, f'ijk_exact_from:{old_path}'

    raise RuntimeError(f'Canonical G1 has N={nfull}, but no valid {N_COMMON}-row mapping was found. Set MAP_FILE.')


def discover_ijk(nrows, prefer=()):
    override = Path(COMMON_IJK_OVERRIDE).resolve() if (COMMON_IJK_OVERRIDE and nrows == N_COMMON) else None
    paths = [override] if override else []
    if not paths:
        for ext in ('*.npy','*.npz'):
            for p in BASE.rglob(ext):
                if is_under(p, OUT):
                    continue
                if 'ijk' not in p.as_posix().lower():
                    continue
                paths.append(p.resolve())
    ranked=[]
    for p in paths:
        try:
            a = np.asarray(load_numeric_array(p))
        except Exception:
            continue
        if a.shape != (nrows,3):
            continue
        if not np.all(np.isfinite(a)):
            continue
        score = 50 if 'ijk' in p.as_posix().lower() else 0
        for k in prefer:
            if k in p.as_posix().lower():
                score += 20
        ranked.append((score,p,np.asarray(a)))
    if not ranked:
        return None
    ranked.sort(key=lambda x:x[0], reverse=True)
    return ranked[0][1], ranked[0][2]


def equal_count_bins(g1, n_bins):
    x = np.asarray(g1, float)
    if not np.all(np.isfinite(x)):
        raise RuntimeError('Canonical G1 contains non-finite values.')
    order = np.argsort(x, kind='mergesort')
    b = np.empty(len(x), dtype=int)
    # exactly/effectively equal-count ordinal bins based on full canonical axis
    labels = (np.floor(np.arange(len(x)) * n_bins / len(x)).astype(int) + 1)
    labels = np.clip(labels, 1, n_bins)
    b[order] = labels
    return b


def discover_roi_labels():
    if ROI_LABEL_OVERRIDE:
        paths = [Path(ROI_LABEL_OVERRIDE).resolve()]
    else:
        paths=[]
        for ext in ('*.tsv','*.csv','*.txt'):
            for p in BASE.rglob(ext):
                if is_under(p, OUT):
                    continue
                txt=p.as_posix().lower()
                if any(k in txt for k in ('roi','65','atlas','target','label','manifest')):
                    paths.append(p.resolve())
    ranked=[]
    for p in paths:
        try:
            df=read_table(p)
        except Exception:
            continue
        if len(df) != N_ROI:
            continue
        c=choose_col(df.columns,['roi_name','roiname','label','name','roi','region'])
        if c is None:
            nonnum=[c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            c=nonnum[0] if nonnum else None
        if c is None:
            continue
        lab=df[c].astype(str).tolist()
        score=keyword_score(p.as_posix(), [('65',40),('roi',30),('manifest',30),('label',20),('atlas',10),('target',10)], [('backup',10)])
        ranked.append((score,p,lab))
    if ranked:
        ranked.sort(key=lambda x:x[0], reverse=True)
        log(f'ROI labels: {ranked[0][1]}')
        return ranked[0][1], ranked[0][2]
    log('ROI label manifest not found; using ROI_01 ... ROI_65.')
    return None, [f'ROI_{i:02d}' for i in range(1,N_ROI+1)]


def decide_fc_scale(fc_files):
    if FC_SCALE in ('r','z'):
        return FC_SCALE, f'user_override:{FC_SCALE}'
    if FC_SCALE != 'auto':
        raise ValueError('FC_SCALE must be auto, r, or z')
    vals=[]
    names=' '.join(str(p).lower() for p in list(fc_files.values())[:10])
    if any(k in names for k in ('fisher','zfc','fisherz','z_fc','fc_z')):
        return 'z','filename_indicates_fisher_z'
    for p in list(fc_files.values())[:5]:
        a=np.asarray(load_numeric_array(p),float)
        if a.shape==(N_ROI,N_COMMON): a=a.T
        vals.append(np.nanmax(np.abs(a)))
    mx=float(np.nanmax(vals))
    if mx > 1.000001:
        return 'z',f'abs_max={mx:.6f}>1'
    return 'r',f'auto_default_raw_correlation_abs_max={mx:.6f}'


def load_fc_matrix(path: Path, scale):
    a=np.asarray(load_numeric_array(path),dtype=float)
    if a.shape==(N_ROI,N_COMMON):
        a=a.T
    if a.shape!=(N_COMMON,N_ROI):
        raise RuntimeError(f'Unexpected FC shape {a.shape}: {path}')
    if not np.isfinite(a).all():
        nbad=np.sum(~np.isfinite(a))
        raise RuntimeError(f'Non-finite FC values ({nbad}) in {path}')
    if scale=='r':
        if np.nanmax(np.abs(a)) > 1.00001:
            raise RuntimeError(f'FC_SCALE=r but values exceed correlation range in {path}')
        a=np.arctanh(np.clip(a,-0.999999,0.999999))
    return a


def fit_mixed(formula, data, label):
    model = MixedLM.from_formula(formula, groups='subject', data=data)
    last=None
    for method in ('lbfgs','bfgs','cg','powell'):
        try:
            res=model.fit(reml=False, method=method, maxiter=3000, disp=False)
            if np.isfinite(res.llf):
                return res, method
        except Exception as e:
            last=e
    raise RuntimeError(f'MixedLM failed for {label}: {last}')


def lrt(full, reduced):
    stat=max(0.0, 2.0*(full.llf-reduced.llf))
    df=max(1, int(round(len(full.fe_params)-len(reduced.fe_params))))
    p=float(stats.chi2.sf(stat,df))
    return stat,df,p


def mean_ci(x, alpha=.05):
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    n=len(x)
    m=float(np.mean(x))
    if n<2:
        return m,np.nan,np.nan
    se=stats.sem(x)
    q=stats.t.ppf(1-alpha/2,n-1)
    return m,float(m-q*se),float(m+q*se)


def corr_safe(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if np.std(a)<1e-12 or np.std(b)<1e-12:
        return np.nan
    return float(np.corrcoef(a,b)[0,1])


def save_matrix_tsv(mat, row_labels, col_labels, path, row_name='bin'):
    df=pd.DataFrame(mat,index=row_labels,columns=col_labels)
    df.index.name=row_name
    df.to_csv(path,sep='\t')


def main():
    if not BASE.exists():
        raise FileNotFoundError(f'BASE does not exist: {BASE}')

    fc_root, fc_files_int = discover_fc_files()
    cov_path, cov = discover_covariates(set(fc_files_int.keys()))
    g1_path, g1_full = discover_g1()
    map_path, rowmap, map_mode = discover_mapping(len(g1_full))
    roi_path, roi_labels = discover_roi_labels()

    # Frozen bins are defined on the full canonical G1 first, then intersected to common voxels.
    bins_full = equal_count_bins(g1_full, N_BINS)
    if len(g1_full)==N_COMMON:
        if not ALLOW_2608_G1:
            raise RuntimeError('STRICT MODE: the selected canonical G1 has only 2608 entries. Point G1_FILE to the FULL frozen 375-sample canonical G1 and MAP_FILE to the exact full→2608 mapping. Set ALLOW_2608_G1=1 only for a deliberate sensitivity run.')
        g1_common=np.asarray(g1_full,float)
        bins_common=bins_full.copy()
        bin_method='equal_count_bins_on_2608_G1_WARNING_full_canonical_not_available'
        log('\nWARNING: chosen canonical G1 itself has 2608 entries. Bins are therefore defined on these 2608 entries.')
        log('For the strictest frozen-axis analysis, point G1_FILE to the full 375 canonical G1 and MAP_FILE to the exact full→2608 row mapping.')
    else:
        g1_common=np.asarray(g1_full[rowmap],float)
        bins_common=np.asarray(bins_full[rowmap],int)
        bin_method=f'equal_count_bins_on_full_canonical_N{len(g1_full)}_then_map_to_{N_COMMON}'

    # Common IJK is optional; used for Figure A only.
    ijk_info=discover_ijk(N_COMMON, prefer=('current','common','2608'))
    common_ijk_path = ijk_info[0] if ijk_info is not None else None
    common_ijk = ijk_info[1] if ijk_info is not None else None

    # Align FC file labels to covariates using integer IDs.
    cov = cov[cov['subject_int'].isin(fc_files_int.keys())].copy()
    cov = cov.sort_values('subject_int').reset_index(drop=True)
    if len(cov) != len(fc_files_int):
        missing=set(fc_files_int)-set(cov.subject_int)
        raise RuntimeError(f'Covariates missing for FC subjects: {sorted(missing)}')
    fc_files={f'sub-{sid:04d}':p for sid,p in fc_files_int.items()}

    scale, scale_reason = decide_fc_scale(fc_files)
    log('\n'+'='*110)
    log('STEP 5. LOAD FC AND COLLAPSE 2608 VOXELS → 10 FIXED G1 BINS')
    log('='*110)
    log(f'FC scale decision: {scale} ({scale_reason})')
    if scale=='r':
        log('Action: Pearson-r FC → Fisher z via arctanh before voxel averaging.')
    else:
        log('Action: FC treated as already Fisher-z / unbounded z values; no second transform.')

    subjects=cov['subject'].tolist()
    nsub=len(subjects)
    F=np.empty((nsub,N_BINS,N_ROI),dtype=float)
    for i,row in cov.iterrows():
        sid=row['subject']
        p=fc_files[sid]
        a=load_fc_matrix(p,scale)
        for b in range(1,N_BINS+1):
            mask=(bins_common==b)
            if mask.sum()==0:
                raise RuntimeError(f'No common voxels in canonical bin {b}')
            F[i,b-1,:]=a[mask,:].mean(axis=0)
        log(f'  [{i+1:02d}/{nsub}] {sid} | {row.group} | {p.name}')

    hc_idx=np.where(cov.group.to_numpy()=='HC')[0]
    sz_idx=np.where(cov.group.to_numpy()=='SZ')[0]
    if len(hc_idx)<20 or len(sz_idx)<20:
        raise RuntimeError(f'Unexpected group sizes: HC={len(hc_idx)}, SZ={len(sz_idx)}')

    log('\n'+'='*110)
    log('STEP 6. BUILD HC NORMATIVE FC REFERENCE AND SUBJECT-LEVEL DEVIATIONS')
    log('='*110)

    hc_F=F[hc_idx]
    mu_hc=hc_F.mean(axis=0)
    sd_hc=hc_F.std(axis=0,ddof=1)
    tiny=(sd_hc<1e-8)
    if tiny.any():
        log(f'WARNING: {tiny.sum()} HC bin×ROI SD cells <1e-8; clipped to 1e-8 for z-scoring.')
        sd_hc=sd_hc.copy(); sd_hc[tiny]=1e-8

    Z=np.empty_like(F)
    prof_r=np.empty((nsub,N_BINS),float)

    # SZ: all HC reference.
    for i in sz_idx:
        Z[i]=(F[i]-mu_hc)/sd_hc
        for b in range(N_BINS):
            prof_r[i,b]=corr_safe(F[i,b],mu_hc[b])

    # HC: leave-one-out reference.
    for i in hc_idx:
        others=hc_idx[hc_idx!=i]
        mu=F[others].mean(axis=0)
        sd=F[others].std(axis=0,ddof=1)
        sd=np.where(sd<1e-8,1e-8,sd)
        Z[i]=(F[i]-mu)/sd
        for b in range(N_BINS):
            prof_r[i,b]=corr_safe(F[i,b],mu[b])

    rms=np.sqrt(np.mean(Z**2,axis=2))
    burden=np.mean(np.abs(Z)>1.96,axis=2)
    signed_mean=np.mean(Z,axis=2)
    prof_z=np.arctanh(np.clip(prof_r,-0.999999,0.999999))

    # Tables for normative reference.
    bin_names=[f'Bin{i}' for i in range(1,N_BINS+1)]
    save_matrix_tsv(mu_hc,bin_names,roi_labels,TABLE_DIR/'HC_normative_mean_FC_10x65.tsv')
    save_matrix_tsv(sd_hc,bin_names,roi_labels,TABLE_DIR/'HC_normative_SD_FC_10x65.tsv')
    save_matrix_tsv(np.mean(Z[sz_idx],axis=0),bin_names,roi_labels,TABLE_DIR/'SZ_mean_z_deviation_10x65.tsv')
    save_matrix_tsv(np.mean(Z[hc_idx],axis=0),bin_names,roi_labels,TABLE_DIR/'HC_LOO_mean_z_deviation_10x65.tsv')

    # Canonical bin manifest.
    bm=[]
    for b in range(1,N_BINS+1):
        mf=(bins_full==b)
        mc=(bins_common==b)
        bm.append({
            'bin':b,
            'canonical_N':int(mf.sum()),
            'common_N':int(mc.sum()),
            'canonical_G1_min':float(g1_full[mf].min()),
            'canonical_G1_max':float(g1_full[mf].max()),
            'canonical_G1_mean':float(g1_full[mf].mean()),
            'common_G1_mean':float(g1_common[mc].mean()),
            'common_G1_median':float(np.median(g1_common[mc])),
        })
    bin_manifest=pd.DataFrame(bm)
    bin_manifest.to_csv(TABLE_DIR/'canonical_G1_bin_manifest.tsv',sep='\t',index=False)

    # Long metrics table.
    rows=[]
    cov_idx=cov.set_index('subject')
    for i,sid in enumerate(subjects):
        cr=cov_idx.loc[sid]
        for b in range(1,N_BINS+1):
            g1m=float(bin_manifest.loc[bin_manifest.bin==b,'common_G1_mean'].iloc[0])
            rows.append({
                'subject':sid,'group':cr['group'],'bin':b,'g1_mean':g1m,
                'rms_deviation':float(rms[i,b-1]),
                'profile_r':float(prof_r[i,b-1]),
                'profile_z':float(prof_z[i,b-1]),
                'abnormal_burden':float(burden[i,b-1]),
                'signed_mean_z':float(signed_mean[i,b-1]),
                'age':float(cr['age']),'sex':float(cr['sex']),'meanFD':float(cr['meanFD'])
            })
    long=pd.DataFrame(rows)
    long['group_sz']=(long.group=='SZ').astype(int)
    long['age_c']=long.age-long[['subject','age']].drop_duplicates().age.mean()
    long['sex_c']=long.sex-long[['subject','sex']].drop_duplicates().sex.mean()
    long['meanFD_c']=long.meanFD-long[['subject','meanFD']].drop_duplicates().meanFD.mean()
    # G1 standardized over the ten fixed bin coordinates.
    g1_mu=bin_manifest.common_G1_mean.mean()
    g1_sd=bin_manifest.common_G1_mean.std(ddof=0)
    long['g1_z']=(long.g1_mean-g1_mu)/g1_sd
    long['g1_z2']=long.g1_z**2
    long.to_csv(TABLE_DIR/'subject_bin_metrics.tsv',sep='\t',index=False)

    # Group-bin summary.
    summ=[]
    for metric in ('rms_deviation','profile_z','abnormal_burden','signed_mean_z'):
        for grp in ('HC','SZ'):
            for b in range(1,N_BINS+1):
                x=long.loc[(long.group==grp)&(long.bin==b),metric]
                m,lo,hi=mean_ci(x)
                summ.append({'metric':metric,'group':grp,'bin':b,'N':len(x),'mean':m,'ci95_low':lo,'ci95_high':hi,'sd':float(x.std(ddof=1))})
    summary=pd.DataFrame(summ)
    summary.to_csv(TABLE_DIR/'group_bin_summary.tsv',sep='\t',index=False)

    log('\n'+'='*110)
    log('STEP 7. PRIMARY AND SECONDARY MIXED-EFFECTS MODELS')
    log('='*110)
    model_rows=[]
    fitted={}
    for metric,role in [('rms_deviation','PRIMARY'),('profile_z','SECONDARY'),('abnormal_burden','SECONDARY')]:
        f_full=f'{metric} ~ group_sz * C(bin) + age_c + sex_c + meanFD_c'
        f_red =f'{metric} ~ group_sz + C(bin) + age_c + sex_c + meanFD_c'
        full,m1=fit_mixed(f_full,long,f'{metric}_full')
        red,m0=fit_mixed(f_red,long,f'{metric}_reduced')
        stat,df,p=lrt(full,red)
        model_rows.append({'role':role,'metric':metric,'test':'Group×Bin omnibus LRT','chi2':stat,'df':df,'p':p,'full_method':m1,'reduced_method':m0,'full_llf':full.llf,'reduced_llf':red.llf})
        fitted[metric]=(full,red)
        log(f'{role:9s} | {metric:18s} | Group×Bin: chi2({df})={stat:.4f}, p={p:.6g}')
        with open(TABLE_DIR/f'model_{metric}_full_summary.txt','w') as f:
            f.write(full.summary().as_text())

    primary_p=float([r['p'] for r in model_rows if r['metric']=='rms_deviation'][0])

    # Linear and quadratic G1 sensitivity for RMS only.
    sens=[]
    fl='rms_deviation ~ group_sz * g1_z + age_c + sex_c + meanFD_c'
    rl='rms_deviation ~ group_sz + g1_z + age_c + sex_c + meanFD_c'
    full,mf=fit_mixed(fl,long,'rms_linear_full')
    red,mr=fit_mixed(rl,long,'rms_linear_reduced')
    st,df,p=lrt(full,red)
    sens.append({'model':'linear','test':'Group×G1','chi2':st,'df':df,'p':p,'full_method':mf,'reduced_method':mr})
    log(f'SENSITIVITY | linear Group×G1: chi2({df})={st:.4f}, p={p:.6g}')

    fq='rms_deviation ~ group_sz + g1_z + g1_z2 + group_sz:g1_z + group_sz:g1_z2 + age_c + sex_c + meanFD_c'
    rq='rms_deviation ~ group_sz + g1_z + g1_z2 + age_c + sex_c + meanFD_c'
    fullq,mf=fit_mixed(fq,long,'rms_quadratic_full')
    redq,mr=fit_mixed(rq,long,'rms_quadratic_reduced')
    st,df,p=lrt(fullq,redq)
    sens.append({'model':'quadratic','test':'joint Group×G1 + Group×G1²','chi2':st,'df':df,'p':p,'full_method':mf,'reduced_method':mr})
    log(f'SENSITIVITY | quadratic joint interactions: chi2({df})={st:.4f}, p={p:.6g}')

    pd.DataFrame(model_rows).to_csv(TABLE_DIR/'mixed_model_omnibus_tests.tsv',sep='\t',index=False)
    pd.DataFrame(sens).to_csv(TABLE_DIR/'RMS_G1_linear_quadratic_sensitivity.tsv',sep='\t',index=False)

    # Gatekept post-hoc tests for RMS only.
    log('\n'+'='*110)
    log('STEP 8. GATEKEPT BIN-LEVEL POST-HOC TESTS')
    log('='*110)
    post=[]
    if primary_p < ALPHA:
        for b in range(1,N_BINS+1):
            d=long[long.bin==b].copy()
            res=smf.ols('rms_deviation ~ group_sz + age_c + sex_c + meanFD_c',data=d).fit(cov_type='HC3')
            ci=res.conf_int().loc['group_sz'].tolist()
            post.append({'bin':b,'beta_SZ_minus_HC':float(res.params['group_sz']),'HC3_SE':float(res.bse['group_sz']),'t':float(res.tvalues['group_sz']),'p_raw':float(res.pvalues['group_sz']),'ci95_low':float(ci[0]),'ci95_high':float(ci[1])})
        post=pd.DataFrame(post)
        reject,q,_,_=multipletests(post.p_raw.to_numpy(),alpha=ALPHA,method='fdr_bh')
        post['p_FDR_BH']=q
        post['FDR_significant']=reject
        post['gate_status']='OPEN_primary_GroupxBin_p<0.05'
        post.to_csv(TABLE_DIR/'RMS_bin_posthoc_FDR.tsv',sep='\t',index=False)
        log(f'Primary gate OPEN (p={primary_p:.6g}). Bin-level HC3 contrasts computed and BH-FDR corrected.')
    else:
        pd.DataFrame([{'gate_status':f'CLOSED_primary_GroupxBin_p={primary_p:.8g}; no inferential bin-level tests performed'}]).to_csv(TABLE_DIR/'RMS_bin_posthoc_FDR.tsv',sep='\t',index=False)
        log(f'Primary gate CLOSED (p={primary_p:.6g}). No inferential bin-level tests performed.')

    log('\n'+'='*110)
    log('STEP 9. BOOTSTRAP STABILITY OF THE 32-HC NORMATIVE FC PROFILE')
    log('='*110)
    rng=np.random.default_rng(SEED)
    boot_corr=np.empty((N_BOOT,N_BINS),float)
    for k in range(N_BOOT):
        samp=rng.integers(0,len(hc_F),size=len(hc_F))
        m=hc_F[samp].mean(axis=0)
        for b in range(N_BINS):
            boot_corr[k,b]=corr_safe(m[b],mu_hc[b])
    bs=[]
    for b in range(N_BINS):
        x=boot_corr[:,b]
        bs.append({'bin':b+1,'N_boot':N_BOOT,'median_profile_r':float(np.nanmedian(x)),'p2_5':float(np.nanpercentile(x,2.5)),'p97_5':float(np.nanpercentile(x,97.5)),'mean_profile_r':float(np.nanmean(x))})
    bs=pd.DataFrame(bs)
    bs.to_csv(TABLE_DIR/'HC_reference_bootstrap_stability.tsv',sep='\t',index=False)
    log(bs.to_string(index=False))

    log('\n'+'='*110)
    log('STEP 10. FIGURES')
    log('='*110)

    # Figure A: common-voxel canonical-bin map if IJK exists, otherwise occupancy panel.
    if common_ijk is not None:
        fig,axs=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
        views=[(1,2,'Sagittal projection','Y','Z'),(0,2,'Coronal projection','X','Z'),(0,1,'Axial projection','X','Y')]
        for ax,(u,v,title,xl,yl) in zip(axs,views):
            sc=ax.scatter(common_ijk[:,u],common_ijk[:,v],c=bins_common,s=5,cmap='turbo',vmin=1,vmax=N_BINS,rasterized=True)
            ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect('equal',adjustable='box')
        cbar=fig.colorbar(sc,ax=axs,shrink=.82,pad=.02); cbar.set_label('Canonical G1 bin')
        fig.suptitle('Fixed 375-sample canonical G1 bins on 2608 common striatal voxels')
        fig.savefig(FIG_DIR/'FigureA_common_voxel_canonical_G1_bins.png',dpi=300,bbox_inches='tight')
        plt.close(fig)
    else:
        fig,ax=plt.subplots(figsize=(8,5))
        ax.bar(bin_manifest.bin,bin_manifest.common_N)
        ax.set_xlabel('Canonical G1 bin'); ax.set_ylabel('Common voxel count')
        ax.set_title('2608 common voxels retained from fixed canonical G1 bins')
        ax.set_xticks(range(1,N_BINS+1))
        fig.savefig(FIG_DIR/'FigureA_common_voxel_canonical_G1_bins.png',dpi=300,bbox_inches='tight')
        plt.close(fig)

    def heatmap(mat,title,cbar_label,path,symmetric=True):
        fig,ax=plt.subplots(figsize=(18,6))
        if symmetric:
            vmax=float(np.nanpercentile(np.abs(mat),98))
            if vmax<=0: vmax=1.0
            im=ax.imshow(mat,aspect='auto',interpolation='nearest',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        else:
            im=ax.imshow(mat,aspect='auto',interpolation='nearest',cmap='viridis')
        ax.set_yticks(np.arange(N_BINS)); ax.set_yticklabels(bin_names)
        ax.set_xticks(np.arange(N_ROI)); ax.set_xticklabels(roi_labels,rotation=90,fontsize=6)
        ax.set_xlabel('Target ROI'); ax.set_ylabel('Canonical G1 position')
        ax.set_title(title)
        cb=fig.colorbar(im,ax=ax,pad=.01); cb.set_label(cbar_label)
        fig.tight_layout()
        fig.savefig(path,dpi=300,bbox_inches='tight')
        plt.close(fig)

    heatmap(mu_hc,'HC normative striatal FC architecture along canonical G1','Mean Fisher-z FC',FIG_DIR/'FigureB_HC_normative_FC_along_G1.png',True)

    def line_metric(metric,ylabel,title,path):
        fig,ax=plt.subplots(figsize=(8.5,5.5))
        for grp in ('HC','SZ'):
            s=summary[(summary.metric==metric)&(summary.group==grp)].sort_values('bin')
            line,=ax.plot(
                s.bin.to_numpy(dtype=float),
                s['mean'].to_numpy(dtype=float),
                marker='o',
                label=grp
            )
            ax.fill_between(s.bin.to_numpy(float),s.ci95_low.to_numpy(float),s.ci95_high.to_numpy(float),alpha=.18,color=line.get_color())
        ax.set_xticks(range(1,N_BINS+1)); ax.set_xlabel('Fixed canonical G1 bin'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(frameon=False)
        ax.text(.02,.98,f'Group × Bin omnibus p = {primary_p:.4g}' if metric=='rms_deviation' else '',transform=ax.transAxes,va='top',ha='left')
        fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches='tight'); plt.close(fig)

    line_metric('rms_deviation','RMS connectivity deviation (HC SD units)','Overall FC abnormality along the normative striatal G1',FIG_DIR/'FigureC_primary_RMS_deviation_along_G1.png')
    heatmap(np.mean(Z[sz_idx],axis=0),'SZ connectivity deviation along canonical G1','Mean SZ deviation (HC SD)',FIG_DIR/'FigureD_SZ_mean_deviation_heatmap.png',True)
    line_metric('profile_z','Fisher-z profile similarity','Similarity to the normative 65-ROI FC fingerprint',FIG_DIR/'FigureE_profile_similarity_along_G1.png')
    line_metric('abnormal_burden','Proportion |Z| > 1.96','Abnormal-connection burden along canonical G1',FIG_DIR/'FigureF_abnormal_burden_along_G1.png')

    fig,ax=plt.subplots(figsize=(8.5,5.2))
    ax.plot(
        bs.bin.to_numpy(dtype=float),
        bs.median_profile_r.to_numpy(dtype=float),
        marker='o'
    )
    ax.fill_between(bs.bin.to_numpy(float),bs.p2_5.to_numpy(float),bs.p97_5.to_numpy(float),alpha=.2)
    ax.set_xticks(range(1,N_BINS+1)); ax.set_ylim(-1,1.02)
    ax.set_xlabel('Canonical G1 bin'); ax.set_ylabel('Bootstrap profile correlation with full-HC reference')
    ax.set_title('Stability of the 32-HC normative FC reference')
    fig.tight_layout(); fig.savefig(FIG_DIR/'FigureS_HC_reference_bootstrap_stability.png',dpi=300,bbox_inches='tight'); plt.close(fig)

    # Provenance.
    log('\n'+'='*110)
    log('STEP 11. PROVENANCE / FREEZE MANIFEST')
    log('='*110)
    prov=[]
    def addprov(kind,p,extra=''):
        if p is None: return
        p=Path(p)
        rec={'kind':kind,'path':str(p),'sha256':'','extra':extra}
        try: rec['sha256']=sha256_file(p)
        except Exception as e: rec['sha256']=f'ERROR:{e}'
        prov.append(rec)
    addprov('canonical_G1',g1_path,f'N={len(g1_full)}')
    addprov('covariates',cov_path,f'Nsubjects={len(cov)}')
    addprov('row_mapping',map_path,map_mode)
    addprov('common_IJK',common_ijk_path,'figure_only')
    addprov('ROI_labels',roi_path,f'N={N_ROI}')
    for sid,p in sorted(fc_files.items()):
        addprov(f'FC_{sid}',p,f'scale={scale}')
    pd.DataFrame(prov).to_csv(TABLE_DIR/'provenance_selected_inputs.tsv',sep='\t',index=False)

    config={
        'BASE':str(BASE),'OUT':str(OUT),'N_COMMON':N_COMMON,'N_ROI':N_ROI,'N_BINS':N_BINS,
        'N_BOOT':N_BOOT,'SEED':SEED,'ALPHA':ALPHA,'FC_scale_used':scale,'FC_scale_reason':scale_reason,
        'bin_method':bin_method,'canonical_G1_N':len(g1_full),'mapping_mode':map_mode,
        'primary_hypothesis':'RMS deviation: Group × Bin omnibus likelihood-ratio test',
        'primary_p':primary_p,
        'posthoc_gate':'Only if primary Group×Bin p < 0.05; 10 bin-level HC3 OLS contrasts; BH-FDR across 10 bins',
        'HC_reference':'SZ uses all HC; HC uses leave-one-out HC reference',
        'covariates':'age, sex, meanFD centered at clinical-sample subject means',
    }
    with open(OUT/'analysis_config.json','w') as f:
        json.dump(config,f,indent=2,ensure_ascii=False)

    # Concise final summary text.
    with open(OUT/'RESULTS_README.txt','w') as f:
        f.write('NORMATIVE G1-CONDITIONED FC DEVIATION ANALYSIS\n')
        f.write('='*70+'\n')
        f.write(f'HC N = {len(hc_idx)}; SZ N = {len(sz_idx)}\n')
        f.write(f'Canonical G1 N = {len(g1_full)}; common voxels = {N_COMMON}; bins = {N_BINS}\n')
        f.write(f'Primary RMS Group×Bin p = {primary_p:.8g}\n')
        f.write('Primary interpretation rule:\n')
        f.write('  p < .05: overall FC abnormality depends on normative G1 position; inspect gatekept bin contrasts.\n')
        f.write('  p >= .05: no evidence for preferential localization of overall FC abnormality along G1.\n')
        f.write('Do not interpret individual bin p-values if the omnibus gate is closed.\n')

    log('\n'+'='*110)
    log('DONE')
    log('='*110)
    log(f'Output: {OUT}')
    log(f'Primary RMS Group×Bin p = {primary_p:.8g}')
    log(f'Key table:  {TABLE_DIR / "mixed_model_omnibus_tests.tsv"}')
    log(f'Key figure: {FIG_DIR / "FigureC_primary_RMS_deviation_along_G1.png"}')
    log('')
    log('IMPORTANT: inspect provenance_selected_inputs.tsv before using any result in the manuscript.')


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        log('\nFATAL ERROR:')
        log(str(e))
        log('\nIf this is an input-discovery ambiguity, rerun with explicit environment variables, e.g.:')
        log('  FC_ROOT=/path/to/fc COV_FILE=/path/cov.tsv G1_FILE=/path/canonical_G1.npy MAP_FILE=/path/row_mapping.npy bash run_normative_G1_FC_deviation.sh')
        raise
