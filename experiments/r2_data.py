"""Datasets for round 2: raw-count multi-omics data with cell-type labels.

Every loader returns the dict format of scpika.loaders:
    views : list of cells x features matrices (raw counts; CSR for RNA/ATAC, dense for ADT)
    names : e.g. ['RNA', 'ADT'] or ['RNA', 'ATAC'] or ['RNA', 'ATAC', 'ADT']
    y     : int labels (0..K-1)
    mask  : (n, V) bool, True if the cell was measured in that view
    meta  : dict(classes=..., batch=array or None, source=...)

Count datasets (all scvi-tools baselines can run):
    SLN111, SLN206      RNA + ADT   totalVI spleen / lymph node (already in datasets/)
    CellMix             RNA + ATAC  SNARE-seq cell-line mixture  (already in datasets/)
    ASAP_PBMC           RNA/ATAC + ADT, mosaic (already in datasets/)
    PBMC3k_Multiome     RNA + ATAC  10x Multiome PBMC 3k, labels of the DCCA repository
                                    (needs the 10x .h5 file, see download_round2.py)
    BMMC_CITE           RNA + ADT   NeurIPS 2021 BMMC (GSE194122), subset of batches
    BMMC_Multiome       RNA + ATAC  NeurIPS 2021 BMMC (GSE194122), subset of batches
Processed-only datasets (only scPIKA's own encoder and PCA-based methods can run):
    BMNC, SMAGE3K, Ma2020_1, PBMC10k_Multiome_feat
"""
import os, sys, gzip, shutil, glob
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from scpika import loaders as L  # noqa: E402

COUNT_DATASETS = ['SLN111', 'SLN206', 'CellMix', 'ASAP_PBMC', 'PBMC3k_Multiome', 'BMMC_CITE', 'BMMC_Multiome']
PROCESSED_DATASETS = ['BMNC', 'SMAGE3K', 'Ma2020_1', 'PBMC10k_Multiome_feat']
BMMC_BATCHES = ['s1d1', 's1d2', 's1d3']        # site 1, three donors (change with --bmmc-batches)
N_PEAKS = 20000                                 # most frequently open peaks kept for ATAC


def root():
    return L.ROOT


def set_root(path):
    L.ROOT = os.path.abspath(path)


# ----------------------------------------------------------------------------- helpers
def _encode(labels):
    u, y = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return y, list(u)


def top_peaks(X, n_top=N_PEAKS):
    X = sp.csr_matrix(X)
    cnt = np.asarray((X > 0).sum(0)).ravel()
    keep = np.sort(np.argsort(-cnt)[:min(n_top, X.shape[1])])
    return X[:, keep]


def _read_10x_h5(path):
    """10x Genomics feature-barcode matrix (.h5, v3). Returns (cells x features CSR, barcodes, feature types, names)."""
    import h5py
    with h5py.File(path, 'r') as f:
        m = f['matrix']
        shape = tuple(m['shape'][...])                     # (features, barcodes)
        X = sp.csc_matrix((m['data'][...], m['indices'][...], m['indptr'][...]), shape=shape).T.tocsr()
        dec = lambda a: np.array([s.decode() if isinstance(s, bytes) else s for s in a[...]])
        bc = dec(m['barcodes'])
        ft = dec(m['features']['feature_type'])
        nm = dec(m['features']['name'])
    return X.astype(np.float32), bc, ft, nm


def _gunzip_once(path_gz):
    out = path_gz[:-3]
    if not os.path.exists(out):
        print(f'  decompressing {os.path.basename(path_gz)} (once) ...', flush=True)
        with gzip.open(path_gz, 'rb') as fi, open(out + '.tmp', 'wb') as fo:
            shutil.copyfileobj(fi, fo, length=64 << 20)
        os.replace(out + '.tmp', out)
    return out


def _h5ad_column(g, name):
    """Read a column of obs/var from an .h5ad group (handles categorical encodings of anndata >= 0.8 and older)."""
    import h5py
    obj = g[name]
    dec = lambda a: np.array([s.decode() if isinstance(s, bytes) else s for s in a])
    if isinstance(obj, h5py.Group) and 'categories' in obj:            # anndata >= 0.8
        cats = dec(obj['categories'][...]); codes = obj['codes'][...]
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)], '')
    vals = obj[...]
    if '__categories' in g and name in g['__categories']:              # anndata < 0.8
        cats = dec(g['__categories'][name][...])
        return np.where(vals >= 0, cats[np.clip(vals, 0, None)], '')
    return dec(vals) if vals.dtype.kind in 'SO' else vals


def _h5ad_csr_rows(grp, rows, n_cols):
    """Rows of a CSR matrix stored in an .h5ad group, read by contiguous runs (no full load)."""
    indptr = grp['indptr'][...]
    rows = np.sort(np.asarray(rows))
    breaks = np.where(np.diff(rows) != 1)[0] + 1
    parts = []
    for run in np.split(rows, breaks):
        a, b = indptr[run[0]], indptr[run[-1] + 1]
        data = grp['data'][a:b]; idx = grp['indices'][a:b]
        ptr = indptr[run[0]:run[-1] + 2] - a
        parts.append(sp.csr_matrix((data, idx, ptr), shape=(run.size, n_cols)))
    return sp.vstack(parts).tocsr().astype(np.float32)


# ----------------------------------------------------------------------------- new datasets
def load_PBMC3k_Multiome():
    """10x Multiome PBMC 3k with the cell-type labels distributed by DCCA (pbmc_3k_cell_cluster_2874.txt).
    The labels come from a Seurat WNN annotation, not from an experimental ground truth."""
    import pandas as pd
    lab = pd.read_csv(L.P('RNA_ATAC', 'PBMC3k_labels_DCCA', 'pbmc_3k_cell_cluster_2874.txt'), sep='\t')
    want = set(lab['Cell'])
    cands = glob.glob(L.P('RNA_ATAC', 'PBMC3k*_10x', '*filtered_feature_bc_matrix.h5'))
    if not cands:
        raise FileNotFoundError('The .h5 file of 10x PBMC 3k Multiome is missing. Run: python download_round2.py PBMC3K')
    best = None
    for c in cands:
        X, bc, ft, nm = _read_10x_h5(c)
        ov = len(want & set(bc))
        print(f'  {os.path.basename(c)}: {ov}/{len(want)} barcodes with matching labels', flush=True)
        if best is None or ov > best[0]:
            best = (ov, X, bc, ft, c)
    ov, X, bc, ft, path = best
    if ov < 0.9 * len(want):
        raise RuntimeError(f'Only {ov}/{len(want)} barcodes match the DCCA labels: the .h5 file is not the right data set.')
    pos = {b: i for i, b in enumerate(bc)}
    lab = lab[lab['Cell'].isin(pos)]
    rows = np.array([pos[b] for b in lab['Cell']])
    X = X[rows]
    rna = X[:, ft == 'Gene Expression'].tocsr()
    atac = top_peaks(X[:, ft == 'Peaks'])
    y, classes = _encode(lab['Cluster'].values)
    return L._pack([rna, atac], ['RNA', 'ATAC'], y, classes=classes, batch=None,
                   source=f'10x Multiome PBMC 3k ({os.path.basename(path)}), labels DCCA')


def _bmmc(kind, batches=None):
    import h5py
    batches = batches or BMMC_BATCHES
    d = os.environ.get('BMMC_DIR') or L.P('RNA_ATAC_and_RNA_ADT', 'BMMC_NeurIPS2021')   # BMMC_DIR: set by r17_bmmc.py
    pat = os.path.join(d, f'GSE194122_openproblems_neurips2021_{kind}_BMMC_processed.h5ad')
    if not os.path.exists(pat):
        if os.path.exists(pat + '.gz'):
            pat = _gunzip_once(pat + '.gz')
        else:
            raise FileNotFoundError(f'{os.path.basename(pat)}(.gz) is missing. Run: python download_round2.py BMMC')
    with h5py.File(pat, 'r') as f:
        obs, var = f['obs'], f['var']
        bcol = 'batch' if 'batch' in obs else None
        ccol = 'cell_type' if 'cell_type' in obs else None
        if bcol is None or ccol is None:
            raise KeyError(f'Columns batch/cell_type not found in obs. Available columns: {list(obs.keys())}')
        batch = _h5ad_column(obs, bcol); ctype = _h5ad_column(obs, ccol)
        ftype = _h5ad_column(var, 'feature_types')
        rows = np.where(np.isin(batch, batches))[0]
        if rows.size == 0:
            raise ValueError(f'No cells in {batches}. Available batches: {sorted(set(batch))}')
        src = f['layers']['counts'] if 'layers' in f and 'counts' in f['layers'] else f['X']
        X = _h5ad_csr_rows(src, rows, len(ftype))
    rna = X[:, ftype == 'GEX'].tocsr()
    other = 'ADT' if kind == 'cite' else 'ATAC'
    O = X[:, ftype == other]
    O = np.asarray(O.todense(), np.float32) if other == 'ADT' else top_peaks(O)
    y, classes = _encode(ctype[rows])
    return L._pack([rna, O], ['RNA', other], y, classes=classes, batch=np.asarray(batch[rows]).astype(str),
                   source=f'GSE194122 {kind} BMMC, batches {",".join(batches)}')


def load_BMMC_CITE():
    return _bmmc('cite')


def load_BMMC_Multiome():
    return _bmmc('multiome')


NEW = {'PBMC3k_Multiome': load_PBMC3k_Multiome, 'BMMC_CITE': load_BMMC_CITE, 'BMMC_Multiome': load_BMMC_Multiome}


def load(name):
    """Raw-count (or processed) dataset with meta['batch'] filled when available."""
    if name in NEW:
        d = NEW[name]()
    else:
        d = L.load(name)
        if name == 'ASAP_PBMC':
            d['meta']['batch'] = np.asarray(d['meta']['batch']).astype(str)
            d['views'][1] = top_peaks(d['views'][1])
        if name == 'CellMix':
            d['views'][1] = top_peaks(d['views'][1])
    d['meta'].setdefault('batch', None)
    return d


def available(names):
    ok = []
    for n in names:
        try:
            load(n); ok.append(n)
        except Exception as e:
            print(f'[skipped] {n}: {e}')
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Check the round-2 data sets')
    ap.add_argument('--data-root', default=os.environ.get('EVPFC_DATA', os.path.join(HERE, '..', 'datasets')))
    a = ap.parse_args()
    set_root(a.data_root)
    for n in COUNT_DATASETS + PROCESSED_DATASETS:
        try:
            d = load(n)
            sh = [tuple(v.shape) for v in d['views']]
            b = d['meta'].get('batch')
            print(f'{n:22s} OK  views={d["names"]} shapes={sh} K={len(np.unique(d["y"]))} '
                  f'missing={(~d["mask"]).mean():.2f} batches={None if b is None else len(set(b))}')
        except Exception as e:
            print(f'{n:22s} NOT USABLE YET: {e}')
