"""Turn the raw benchmark datasets into model-ready views.

Each view gets
    x     : encoder input (CSR or dense float32), one row per cell (rows of missing cells are empty)
    target: reconstruction target (counts for NB/ZINB, 0/1 for Bernoulli, standardised values for Gaussian)
    lik   : 'zinb' | 'nb' | 'bernoulli' | 'gaussian'
    sf    : size factor per cell (count likelihoods only)
    q     : quality deficit per cell (z-score of -log library size, clipped to [-3, 3]; 0 for processed data)
    mask  : bool per cell, True if the cell was measured in this view
Preprocessing choices are deliberately standard (library-size normalisation + log1p for RNA,
CLR for ADT, binarisation for ATAC, z-scoring for already processed features).
"""
import numpy as np
import scipy.sparse as sp


def _is_counts(X, n_check=2000):
    x = X[:n_check]
    vals = x.data if sp.issparse(x) else np.asarray(x).ravel()
    vals = vals[vals != 0]
    return vals.size > 0 and vals.min() >= 0 and np.allclose(vals[:100000], np.round(vals[:100000]))


def _hvg(Xc, n_top):
    """Highly variable features by normalised dispersion of log-normalised data (Seurat v1 style)."""
    if Xc.shape[1] <= n_top:
        return np.arange(Xc.shape[1])
    lib = np.asarray(Xc.sum(1)).ravel()
    lib[lib == 0] = 1
    Xn = sp.diags(1e4 / lib) @ Xc
    Xn = Xn.log1p() if sp.issparse(Xn) else np.log1p(Xn)
    mean = np.asarray(Xn.mean(0)).ravel()
    sq = np.asarray(Xn.multiply(Xn).mean(0)).ravel() if sp.issparse(Xn) else (Xn ** 2).mean(0)
    var = sq - mean ** 2
    disp = var / (mean + 1e-8)
    disp[mean <= 0] = -np.inf
    return np.sort(np.argsort(-disp)[:n_top])


def _quality(lib, mask):
    q = np.zeros(lib.shape[0], np.float32)
    l = np.log(lib[mask] + 1.0)
    if l.size > 1 and l.std() > 0:
        q[mask] = np.clip(-(l - l.mean()) / l.std(), -3, 3)
    return q


def build_view(X, name, mask, n_hvg=2000, n_peaks=10000):
    X = sp.csr_matrix(X) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    counts = _is_counts(X[mask] if mask.sum() < n else X)
    v = dict(name=name, mask=mask.copy())
    if not counts:  # already processed features (log-normalised or reduced dimensions)
        Xd = X.toarray() if sp.issparse(X) else X
        Xd = Xd.astype(np.float32)
        mu, sd = Xd[mask].mean(0), Xd[mask].std(0) + 1e-6
        Z = np.clip((Xd - mu) / sd, -10, 10).astype(np.float32)
        Z[~mask] = 0
        v.update(x=Z, target=Z, lik='gaussian', sf=np.ones(n, np.float32), q=np.zeros(n, np.float32))
        return v
    Xc = sp.csr_matrix(X, dtype=np.float32)
    if name.upper() == 'ATAC':
        Xb = Xc.copy(); Xb.data[:] = 1.0
        freq = np.asarray(Xb[mask].sum(0)).ravel()
        keep = np.sort(np.argsort(-freq)[:n_peaks])
        Xb = Xb[:, keep]
        lib = np.asarray(Xb.sum(1)).ravel()
        v.update(x=Xb, target=Xb, lik='bernoulli', sf=np.ones(n, np.float32), q=_quality(lib, mask))
        return v
    if name.upper() == 'ADT':
        Xa = Xc.toarray()
        lib = Xa.sum(1)
        L = np.log1p(Xa)
        clr = (L - L.mean(1, keepdims=True)).astype(np.float32)
        clr[~mask] = 0
        sf = np.ones(n, np.float32)
        sf[mask] = lib[mask] / np.median(lib[mask][lib[mask] > 0])
        sf[sf <= 0] = 1
        v.update(x=clr, target=Xa.astype(np.float32), lik='nb', sf=sf, q=_quality(lib, mask))
        return v
    # RNA counts
    keep = _hvg(Xc[mask], n_hvg)
    Xh = Xc[:, keep]
    lib_all = np.asarray(Xc.sum(1)).ravel()
    lib = lib_all.copy(); lib[lib == 0] = 1
    Xn = (sp.diags(1e4 / lib) @ Xh).log1p().astype(np.float32)
    sf = np.ones(n, np.float32)
    sf[mask] = lib[mask] / np.median(lib[mask])
    v.update(x=Xn.tocsr(), target=Xh.tocsr(), lik='zinb', sf=sf, q=_quality(lib_all, mask))
    return v


def build(dataset, n_hvg=2000, n_peaks=10000, force_gaussian=False, reduce='counts'):
    """dataset: dict from loaders.load(...). Returns (views, y, meta)."""
    views = []
    for j, (X, nm) in enumerate(zip(dataset['views'], dataset['names'])):
        v = build_view(X, nm, dataset['mask'][:, j].astype(bool), n_hvg, n_peaks)
        if force_gaussian and v['lik'] != 'gaussian':   # ablation: drop the count likelihood
            X_in = v['x'].toarray() if hasattr(v['x'], 'toarray') else np.asarray(v['x'])
            m = v['mask']; mu, sd = X_in[m].mean(0), X_in[m].std(0) + 1e-6
            Z = np.clip((X_in - mu) / sd, -10, 10).astype(np.float32); Z[~m] = 0
            v.update(x=Z, target=Z, lik='gaussian')
        if reduce == 'counts':          # default pipeline: embed raw count views, keep processed features
            if v['lik'] != 'gaussian':
                v = reduce_view(v, 50)
        elif reduce:
            v = reduce_view(v, int(reduce))
        views.append(v)
    return views, dataset['y'], dataset.get('meta', {})


def rows(M, idx):
    """Dense float32 rows of a CSR/dense matrix."""
    r = M[idx]
    return (r.toarray() if sp.issparse(r) else r).astype(np.float32)


def reduce_view(v, n_comp=50, seed=0):
    """Replace a view by its standard low-dimensional embedding (PCA of the normalised input;
    TF-IDF + LSI with the first component dropped for accessibility). Gaussian likelihood afterwards."""
    from sklearn.decomposition import PCA, TruncatedSVD
    m = v['mask']; X = v['x']
    Xm = X[m]
    k = int(min(n_comp, Xm.shape[1] - 1, Xm.shape[0] - 1))
    if v['lik'] == 'bernoulli':
        Xm = sp.csr_matrix(Xm)
        tf = sp.diags(1.0 / np.maximum(np.asarray(Xm.sum(1)).ravel(), 1)) @ Xm
        idf = np.log1p(Xm.shape[0] / np.maximum(np.asarray(Xm.sum(0)).ravel(), 1))
        E = TruncatedSVD(k + 1, random_state=seed).fit_transform((tf @ sp.diags(idf)).log1p())[:, 1:]
    elif sp.issparse(Xm):
        E = TruncatedSVD(k, random_state=seed).fit_transform(Xm)
    else:
        E = PCA(k, random_state=seed).fit_transform(np.asarray(Xm, np.float32))
    E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    Z = np.zeros((X.shape[0], E.shape[1]), np.float32); Z[m] = np.clip(E, -10, 10)
    w = dict(v); w.update(x=Z, target=Z, lik='gaussian', sf=np.ones(X.shape[0], np.float32))
    return w
