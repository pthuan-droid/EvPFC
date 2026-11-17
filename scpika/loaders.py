"""Load the benchmark data sets into a common format for multi-view clustering.

Every load_* function returns a dict:
    views : list[np.ndarray | scipy.sparse]  -- each element is the (n_cells x n_features) matrix of one view
    names : list[str]                        -- view names, e.g. ['RNA', 'ADT']
    y     : np.ndarray | None                -- cell labels (integers 0..K-1), None if not available
    mask  : np.ndarray (n_cells x V) bool    -- True if the cell has that view (mosaic or missing-view data)
    meta  : dict

Example:
    from loaders import load, DATASETS
    d = load('BMNC'); X_rna, X_adt = d['views']; y = d['y']

Can: numpy, scipy, h5py, pandas, anndata  (pip install numpy scipy h5py pandas anndata)
"""
import os
import numpy as np
import scipy.sparse as sp

ROOT = os.path.abspath(os.environ.get('EVPFC_DATA', os.path.join(os.path.dirname(__file__), '..', 'datasets')))  # <repo>/datasets
P = lambda *a: os.path.join(ROOT, *a)


def _encode(labels):
    u, y = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return y, list(u)


def _pack(views, names, y=None, mask=None, **meta):
    n = views[0].shape[0]
    if mask is None:
        mask = np.ones((n, len(views)), bool)
    return dict(views=views, names=names, y=y, mask=mask, meta=meta)


def _mat_two_views(path):
    """.mat v7.3 file of scEMC/scUNC: X is a cell array {view1, view2}, each view (features x cells)."""
    import h5py
    f = h5py.File(path, 'r')
    views = [np.array(f[r]).T.astype(np.float32) for r in f['X'][0]]
    y = np.array(f['Y']).ravel()
    y = np.unique(y, return_inverse=True)[1]
    return views, y


# ---------------- RNA + ADT ----------------
def load_BMNC():
    v, y = _mat_two_views(P('RNA_ADT', 'BMNC', 'BMNC.mat'))
    return _pack(v, ['RNA', 'ADT'], y, source='scEMC repo (Stuart et al. 2019 CITE-seq, bone marrow)')


def _totalvi(path, labels=None, drop_doublets=True):
    import anndata as ad
    a = ad.read_h5ad(path)
    rna = sp.csr_matrix(a.X)
    adt = np.asarray(a.obsm['protein_expression'], dtype=np.float32)
    y, classes = None, None
    if labels is not None:
        labels = np.asarray(labels).astype(str)
        keep = np.ones(a.n_obs, bool)
        if drop_doublets:
            keep = ~np.char.find(np.char.lower(labels), 'doublet').__ge__(0)
            keep &= ~np.char.find(np.char.lower(labels), 'low quality').__ge__(0)
        rna, adt, labels = rna[keep], adt[keep], labels[keep]
        y, classes = _encode(labels)
    return _pack([rna, adt], ['RNA', 'ADT'], y, classes=classes, source=os.path.basename(path))


def _sln_labels():
    """annotations_subclustered.csv is ordered as the 16828 SLN111 cells followed by the 15820 SLN206 cells
    (checked: the leiden_subclusters column matches the obs of each file 100%)."""
    import pandas as pd
    lab = pd.read_csv(P('RNA_ADT', 'SLN_annotations', 'annotations_subclustered.csv'), index_col=0)['labels'].values
    return lab[:16828], lab[16828:]


def load_SLN111(drop_doublets=True):
    return _totalvi(P('RNA_ADT', 'SLN111_D1', 'spleen_lymph_111.h5ad'), _sln_labels()[0], drop_doublets)


def load_SLN206(drop_doublets=True):
    return _totalvi(P('RNA_ADT', 'SLN206_D2', 'spleen_lymph_206.h5ad'), _sln_labels()[1], drop_doublets)


def load_PBMC10k_CITE():  # no cell labels
    return _totalvi(P('RNA_ADT', 'PBMC10k_CITE', 'pbmc_10k_protein_v3.h5ad'))


def load_PBMC5k_CITE():   # no cell labels
    return _totalvi(P('RNA_ADT', 'PBMC5k_CITE', 'pbmc_5k_protein_v3.h5ad'))


def load_MALT10k_CITE():  # no cell labels
    return _totalvi(P('RNA_ADT', 'MALT10k_CITE', 'malt_10k_protein_v3.h5ad'))


# ---------------- RNA + ATAC ----------------
def load_SMAGE3K():
    v, y = _mat_two_views(P('RNA_ATAC', 'SMAGE-3K', 'SMAGE-3K.mat'))
    return _pack(v, ['RNA', 'ATAC'], y, source='scUNC repo (10x Multiome PBMC 3k)')


def load_CellMix():
    import anndata as ad
    r = ad.read_h5ad(P('RNA_ATAC', 'CellMix_SNARE_scCLC', 'data.h5ad'))
    a = ad.read_h5ad(P('RNA_ATAC', 'CellMix_SNARE_scCLC', 'ATAC', 'data.h5ad'))
    a = a[r.obs_names]
    y, classes = _encode(r.obs['cell_type'].values)
    return _pack([sp.csr_matrix(r.X), sp.csr_matrix(a.X)], ['RNA', 'ATAC'], y, classes=classes,
                 source='scCLC repo (SNARE-seq cell-line mixture, GSE126074)')


def _scmdcl(name):
    d = P('RNA_ATAC', f'{name}_features_scMDCL')
    v = [np.load(os.path.join(d, 'RNA_fea.npy')), np.load(os.path.join(d, 'ATAC_fea.npy'))]
    y = np.load(os.path.join(d, 'label.npy'))
    y = np.unique(y, return_inverse=True)[1]
    graphs = [sp.load_npz(os.path.join(d, f'{m}_euc_20.npz')) for m in ('RNA', 'ATAC')]
    return _pack(v, ['RNA', 'ATAC'], y, knn_graphs=graphs,
                 note='Reduced (100-d) features of scMDCL, NOT the raw count matrix')


def load_Ma2020_1():
    return _scmdcl('Ma-2020-1')


def load_PBMC10k_Multiome_feat():
    return _scmdcl('PBMC-10k')


# ---------------- Mosaic (views missing by batch) ----------------
def load_ASAP_PBMC():
    """4 batches: c1,c2 = RNA+ADT (CITE-seq); c3,c4 = ATAC+ADT (ASAP-seq).
    Returned views: [RNA (gene), ATAC (region), ADT (protein)] for all cells; a missing view = zero rows and mask=False."""
    import pandas as pd
    d = P('Mosaic', 'ASAP-PBMC_scMoMaT')
    ld = lambda f: sp.load_npz(os.path.join(d, f)).T.tocsr()  # -> cells x features
    G = {1: ld('GxC1.npz'), 2: ld('GxC2.npz')}
    R = {3: ld('RxC3.npz'), 4: ld('RxC4.npz')}
    Pr = {b: ld(f'PxC{b}.npz') for b in (1, 2, 3, 4)}
    n = {b: Pr[b].shape[0] for b in (1, 2, 3, 4)}
    nG, nR = G[1].shape[1], R[3].shape[1]
    rna = sp.vstack([G.get(b, sp.csr_matrix((n[b], nG))) for b in (1, 2, 3, 4)]).tocsr()
    atac = sp.vstack([R.get(b, sp.csr_matrix((n[b], nR))) for b in (1, 2, 3, 4)]).tocsr()
    adt = sp.vstack([Pr[b] for b in (1, 2, 3, 4)]).tocsr()
    mask = np.vstack([np.tile([b in G, b in R, True], (n[b], 1)) for b in (1, 2, 3, 4)])
    meta = pd.concat([pd.read_csv(os.path.join(d, f'meta_c{b}.csv'), index_col=0) for b in (1, 2, 3, 4)])
    y, classes = _encode(meta['coarse_cluster'].values)
    batch = np.concatenate([[b] * n[b] for b in (1, 2, 3, 4)])
    gxr = sp.load_npz(os.path.join(d, 'GxR.npz'))  # ma tran gen x vung (gene activity)
    return _pack([rna, atac, adt], ['RNA', 'ATAC', 'ADT'], y, mask, classes=classes, batch=batch,
                 fine_labels=meta['cluster'].values, gene_region=gxr)


# ---------------- scRNA-seq (for derived multi-view data) ----------------
def load_Quake_Spleen():
    import h5py
    f = h5py.File(P('scRNA_only', 'Quake_10x_Spleen', 'data.h5'), 'r')
    e = f['exprs']
    X = sp.csr_matrix((e['data'][...], e['indices'][...], e['indptr'][...]), shape=tuple(e['shape'][...]))
    lab = [s.decode() if isinstance(s, bytes) else s for s in f['obs']['cell_type1'][...]]
    y, classes = _encode(lab)
    return _pack([X], ['RNA'], y, classes=classes)


DATASETS = {
    'BMNC': load_BMNC, 'SLN111': load_SLN111, 'SLN206': load_SLN206,
    'PBMC10k_CITE': load_PBMC10k_CITE, 'PBMC5k_CITE': load_PBMC5k_CITE, 'MALT10k_CITE': load_MALT10k_CITE,
    'SMAGE3K': load_SMAGE3K, 'CellMix': load_CellMix, 'Ma2020_1': load_Ma2020_1,
    'PBMC10k_Multiome_feat': load_PBMC10k_Multiome_feat, 'ASAP_PBMC': load_ASAP_PBMC,
    'Quake_Spleen': load_Quake_Spleen,
}


def load(name):
    return DATASETS[name]()


def make_incomplete(mask, missing_rate, seed=0, keep_view=0):
    """Create a random missing-view mask (every cell keeps at least one view; view keep_view is always kept).
    Used to simulate experiments with 10-70% missing views."""
    rng = np.random.default_rng(seed)
    m = mask.copy()
    n, V = m.shape
    drop = rng.random((n, V)) < missing_rate
    drop[:, keep_view] = False
    m &= ~drop
    return m


if __name__ == '__main__':
    for k, f in DATASETS.items():
        try:
            d = f()
            shapes = [tuple(v.shape) for v in d['views']]
            K = None if d['y'] is None else len(np.unique(d['y']))
            print(f'{k:24s} views={d["names"]} shapes={shapes} K={K} missing={(~d["mask"]).mean():.2f}')
        except Exception as e:
            print(f'{k:24s} ERROR: {e}')
