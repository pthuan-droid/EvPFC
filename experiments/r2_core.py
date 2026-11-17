"""Shared machinery of round 2: protocol, encoders (latent spaces), heads, label audit, CSV writer."""
import os, sys, json, time, csv, copy
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
from scpika import data as D, metrics as M, baselines as B  # noqa: E402
from scpika.core import SCPIKA, Config  # noqa: E402
import r2_data  # noqa: E402

CACHE = os.path.join(HERE, 'cache')
RESULTS = os.path.join(HERE, 'results')
OPTS = dict(max_epochs=None, threads=4, verbose=False, cfg_extra={})   # set by r2_run.py (cfg_extra: smoke tests)

# v1 configuration of the paper = Config() defaults; open-world mode = V2a of the improvement study
CFG_CLOSED = {}
CFG_OPEN = dict(know_open=True, decision='fill', target='fill', free_init_theta=True)


# ============================================================================ protocol
def make_setting(y, seed, frac, noise=0.0, hide=0, miss=0.0):
    """Same random stream as run_experiments.run_job: rng = RandomState(1000 + seed); draws in the order
    missing views -> hidden types -> labelled cells -> label noise."""
    rng = np.random.RandomState(1000 + seed)
    n = len(y)
    drop = (rng.rand(n) < miss) if miss else np.zeros(n, bool)
    valid = y >= 0
    classes = np.unique(y[valid])
    yy = y.copy()
    hidden = []
    if hide:
        sizes = np.array([np.sum(y == c) for c in classes])
        cand = classes[(sizes >= 100) & (sizes < sizes.max())]
        hidden = list(rng.choice(cand, hide, replace=False))
        known = [c for c in classes if c not in hidden]
        remap = {c: i for i, c in enumerate(list(known) + hidden)}
        yy = np.array([remap.get(c, -2) for c in y])
    K = len(classes)
    T = K - len(hidden)
    lab = valid & (rng.rand(n) < frac) & (yy < T)
    yk = np.full(n, -1)
    yk[lab] = yy[lab]
    noisy = np.zeros(n, bool)
    if noise > 0:
        noisy = lab & (rng.rand(n) < noise)
        for i in np.where(noisy)[0]:
            yk[i] = rng.choice([c for c in range(T) if c != yy[i]])
    return dict(y=yy, yk=yk, lab=lab, ev=valid & ~lab, valid=valid, T=T, K=K, drop=drop,
                noisy=noisy, hidden=[int(h) for h in hidden])


def apply_missing(d, drop, view_name):
    """Copy of dataset d in which the cells in `drop` lost view `view_name` (rows zeroed, mask False)."""
    if not drop.any():
        return d
    d = dict(d); d['views'] = list(d['views']); d['mask'] = d['mask'].copy()
    j = d['names'].index(view_name)
    d['mask'][drop, j] = False
    keep = (~drop).astype(np.float32)
    X = d['views'][j]
    d['views'][j] = (sp.diags(keep) @ X).tocsr() if sp.issparse(X) else X * keep[:, None]
    return d


def miss_view(d):
    return 'ATAC' if 'ATAC' in d['names'] else d['names'][-1]


def subsample(d, idx):
    d = dict(d)
    d['views'] = [v[idx] for v in d['views']]
    d['mask'] = d['mask'][idx]; d['y'] = d['y'][idx]
    d['meta'] = dict(d['meta'])
    if d['meta'].get('batch') is not None:
        d['meta']['batch'] = np.asarray(d['meta']['batch'])[idx]
    return d


# ============================================================================ encoders
def encoders_for(d, processed=False):
    if processed:
        return ['pca']
    names = d['names']
    has_rna_all = 'RNA' in names and d['mask'][:, names.index('RNA')].all()
    enc = ['pca']
    if has_rna_all:
        enc.append('scvi')
    if 'ADT' in names and 'ATAC' not in names:
        enc.append('totalvi')
    if 'ATAC' in names:
        enc.append('multivi')
    return enc


def _hvg_counts(X, n_top=2000):
    X = sp.csr_matrix(X)
    return X[:, D._hvg(X, n_top)].astype(np.float32)


def _dense(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, np.float32)


def _scvi_settings(seed):
    import scvi
    scvi.settings.seed = seed
    scvi.settings.dl_num_workers = 0
    torch.set_num_threads(OPTS['threads'])
    return dict(early_stopping=True, accelerator='auto', enable_progress_bar=bool(OPTS['verbose']),
                max_epochs=OPTS['max_epochs'])


def _rna_adata(d):
    import anndata as ad
    ri = d['names'].index('RNA')
    a = ad.AnnData(_hvg_counts(d['views'][ri]))
    a.obs_names = [f'c{i}' for i in range(a.n_obs)]
    b = d['meta'].get('batch')
    if b is not None:
        a.obs['batch'] = np.asarray(b).astype(str)
    return a


def _train_scvi(d, seed, path_model):
    import scvi
    kw = _scvi_settings(seed)
    a = _rna_adata(d)
    a.obs['label'] = 'Unknown'
    scvi.model.SCVI.setup_anndata(a, batch_key='batch' if 'batch' in a.obs else None, labels_key='label')
    m = scvi.model.SCVI(a)
    m.train(**kw)
    m.save(path_model, overwrite=True, save_anndata=False)
    return m.get_latent_representation()


def _train_totalvi(d, seed):
    import scvi
    kw = _scvi_settings(seed)
    a = _rna_adata(d)
    a.obsm['protein'] = _dense(d['views'][d['names'].index('ADT')])
    scvi.model.TOTALVI.setup_anndata(a, protein_expression_obsm_key='protein',
                                     batch_key='batch' if 'batch' in a.obs else None)
    m = scvi.model.TOTALVI(a)
    m.train(**kw)
    return m.get_latent_representation()


def _train_multivi(d, seed):
    import scvi, anndata as ad, mudata as md
    kw = _scvi_settings(seed)
    n, names, mask = d['y'].shape[0], d['names'], d['mask']
    ri, ai = names.index('RNA'), names.index('ATAC')
    Xr = sp.diags(mask[:, ri].astype(np.float32)) @ _hvg_counts(d['views'][ri])
    Xa = sp.diags(mask[:, ai].astype(np.float32)) @ sp.csr_matrix(d['views'][ai]).astype(np.float32)
    mods = {'rna': ad.AnnData(sp.csr_matrix(Xr)), 'atac': ad.AnnData(sp.csr_matrix(Xa))}
    mods['rna'].var_names = [f'g{i}' for i in range(Xr.shape[1])]
    mods['atac'].var_names = [f'p{i}' for i in range(Xa.shape[1])]
    cfg = dict(rna_layer='rna', atac_layer='atac')
    if 'ADT' in names:
        P = _dense(d['views'][names.index('ADT')])
        mods['protein'] = ad.AnnData(P)
        mods['protein'].var_names = [f'a{i}' for i in range(P.shape[1])]
        cfg['protein_layer'] = 'protein'
    for k in mods:
        mods[k].obs_names = [f'c{i}' for i in range(n)]
    mdata = md.MuData(mods)
    b = d['meta'].get('batch')
    if b is not None:
        mdata.obs['batch'] = np.asarray(b).astype(str)
    scvi.model.MULTIVI.setup_mudata(mdata, batch_key='batch' if b is not None else None, modalities=cfg)
    m = scvi.model.MULTIVI(mdata, n_genes=Xr.shape[1], n_regions=Xa.shape[1])
    m.train(**kw)
    return m.get_latent_representation()


def pca_embedding(d, seed):
    views, _, _ = D.build(d, reduce='counts')
    return B.joint_embedding(views, seed=seed).astype(np.float32)


def ae_embedding(d, seed):
    """Unsupervised latent of scPIKA's own view autoencoders: exactly the pre-training stage of SCPIKA.fit
    (reconstruction loss only, no label is used). Views: PCA / LSI of counts or the processed features.
    Returns the concatenation of the per-view latents (zeros where a view is missing)."""
    from scpika.model import ViewAE
    cfg = Config(seed=seed, threads=OPTS['threads'], **OPTS.get('cfg_extra', {}))
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(OPTS['threads'])
    views, _, _ = D.build(d, reduce='counts')
    m = SCPIKA(cfg)
    m.views = [dict(v) for v in views]
    m.V, m.n = len(views), views[0]['mask'].shape[0]
    m.aes = [ViewAE(v['x'].shape[1], v['lik'], cfg.d, cfg.hidden) for v in m.views]
    opt = torch.optim.Adam([p for ae in m.aes for p in ae.parameters()], lr=cfg.lr)
    for ep in range(cfg.pre_epochs):
        for idx in m._batches():
            loss = 0.0
            for ae, v in zip(m.aes, m.views):
                b = idx[v['mask'][idx]]
                if b.size < 2:
                    continue
                x, t = m._batch_tensors(v, b)
                loss = loss + ae.recon_loss(ae.encode(x), t, None)
            opt.zero_grad(); loss.backward(); opt.step()
    Z = m._encode_all()
    return torch.cat(Z, 1).numpy().astype(np.float32)


def latent(ds, d, enc, seed, miss=0.0, tag=''):
    """Unsupervised latent space of dataset d (cached). Labels are never used here, so one latent serves
    every label fraction / noise / hidden-type setting of the same (dataset, encoder, seed, missing rate)."""
    os.makedirs(os.path.join(CACHE, 'latents'), exist_ok=True)
    key = f'{ds}__{enc}__s{seed}__m{miss}{tag}'
    f = os.path.join(CACHE, 'latents', key + '.npy')
    fj = f[:-4] + '.json'
    if os.path.exists(f):
        return np.load(f), json.load(open(fj))['time_s']
    t0 = time.time()
    if enc == 'pca':
        Z = pca_embedding(d, seed)
    elif enc == 'ae':
        Z = ae_embedding(d, seed)
    elif enc == 'scvi':
        Z = _train_scvi(d, seed, os.path.join(CACHE, 'models', key))
    elif enc == 'totalvi':
        Z = _train_totalvi(d, seed)
    elif enc == 'multivi':
        Z = _train_multivi(d, seed)
    else:
        raise ValueError(enc)
    t = time.time() - t0
    np.save(f, np.asarray(Z, np.float32))
    json.dump(dict(time_s=t, n=int(Z.shape[0]), dim=int(Z.shape[1])), open(fj, 'w'))
    return np.asarray(Z, np.float32), t


SCANVI_FALLBACK = []


def scanvi(ds, d, seed, yk, miss=0.0):
    """scANVI (RNA) initialised from the cached scVI model of the same (dataset, seed)."""
    import scvi
    kw = _scvi_settings(seed)
    key = f'{ds}__scvi__s{seed}__m{miss}'
    path = os.path.join(CACHE, 'models', key)
    if not os.path.exists(path):
        latent(ds, d, 'scvi', seed, miss)
    kw2 = dict(kw); kw2['max_epochs'] = OPTS['max_epochs'] and min(OPTS['max_epochs'], 20)

    def fit(fallback=False):
        a = _rna_adata(d)
        a.obs['label'] = 'Unknown'
        scvi.model.SCVI.setup_anndata(a, batch_key='batch' if 'batch' in a.obs else None, labels_key='label')
        m = scvi.model.SCVI.load(path, adata=a)
        a.obs['label'] = np.where(yk >= 0, yk.astype(str), 'Unknown')
        s = scvi.model.SCANVI.from_scvi_model(m, unlabeled_category='Unknown', labels_key='label', adata=a)
        if fallback:
            s.train(**dict(kw2, early_stopping=False, train_size=1.0))
        else:
            s.train(**kw2)
        return s
    try:
        s = fit()
    except ValueError:
        # very few labelled cells: no labelled cell falls in scANVI's validation split, so early stopping on
        # the validation loss is impossible; train on all cells without early stopping instead
        s = fit(True)
        SCANVI_FALLBACK.append(True)
    P = s.predict(soft=True)
    T = int(yk.max()) + 1
    proba = np.zeros((len(yk), T))
    for c in P.columns:
        if str(c) != 'Unknown':
            proba[:, int(c)] = P[c].values
    return proba.argmax(1), proba


# ============================================================================ heads
def _full_proba(dist, classes, T):
    P = np.zeros((dist.shape[0], T))
    P[:, np.asarray(classes, int)] = dist
    return P


def head_ls(Z, yk, T):
    from sklearn.semi_supervised import LabelSpreading
    ls = LabelSpreading(kernel='knn', n_neighbors=15, alpha=0.2, max_iter=100).fit(Z, yk)
    P = _full_proba(ls.label_distributions_, ls.classes_[ls.classes_ >= 0], T)
    return P.argmax(1), P


def head_knn(Z, yk, T, ks=(1, 3, 5, 10, 15, 30)):
    """k-NN classifier (distance-weighted); k chosen by 5-fold cross-validated accuracy on the labelled cells."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import KFold
    lab = np.where(yk >= 0)[0]
    best = (-1, 15)
    for k in ks:
        acc = []
        for tr, te in KFold(5, shuffle=True, random_state=0).split(lab):
            if tr.size < k:
                continue
            c = KNeighborsClassifier(n_neighbors=k, weights='distance').fit(Z[lab[tr]], yk[lab[tr]])
            acc.append(np.mean(c.predict(Z[lab[te]]) == yk[lab[te]]))
        if acc and np.mean(acc) > best[0]:
            best = (np.mean(acc), k)
    k = min(best[1], lab.size)
    c = KNeighborsClassifier(n_neighbors=k, weights='distance').fit(Z[lab], yk[lab])
    P = _full_proba(c.predict_proba(Z), c.classes_, T)
    return P.argmax(1), P


def _eknn_model(Z, yk, T, seed, **cfg):
    m = SCPIKA(Config(seed=seed, **cfg))
    n = Z.shape[0]
    m.views = [dict(mask=np.ones(n, bool))]
    m.n, m.K, m.V, m.y_known = n, T, 1, yk
    return m


def head_eknn(Z, yk, T, seed, **cfg):
    """Evidential k-NN knowledge view of scPIKA used as a classifier. Returns pred, BetP, m(Theta),
    and the leave-one-out BetP of every labelled cell (for the label audit)."""
    lab = np.where(yk >= 0)[0]
    m = _eknn_model(Z, yk, T, seed, **cfg)
    S, Th, bl = m._eknn_view([torch.from_numpy(np.ascontiguousarray(Z, np.float32))], [0], lab,
                             torch.from_numpy(yk[lab]), torch.full((lab.size,), float(m.cfg.eps)))
    P = (S + Th[:, None] / T).numpy()
    return P.argmax(1), P, Th.numpy(), bl.numpy()


def head_scpika_latent(Z, yk, T, K, seed, mode='closed', **extra):
    """Full scPIKA with the latent space as its single input view."""
    cfg = dict(CFG_OPEN if mode == 'open' else CFG_CLOSED)
    cfg.update(extra)
    v = D.build_view(np.asarray(Z, np.float32), 'latent', np.ones(Z.shape[0], bool))
    cfg.update(OPTS['cfg_extra'])
    m = SCPIKA(Config(seed=seed, threads=OPTS['threads'], **cfg)).fit([v], yk, T, K)
    return m.predict(), m.uncertainty()['decision_proba'], m.uncertainty()


def head_scpika_own(d, yk, T, K, seed, **cfg):
    """scPIKA with its own encoder on PCA / LSI of the counts (the configuration of the paper)."""
    views, _, _ = D.build(d, reduce='counts')
    cfg = dict(cfg); cfg.update(OPTS['cfg_extra'])
    m = SCPIKA(Config(seed=seed, threads=OPTS['threads'], **cfg)).fit(views, yk, T, K)
    return m.predict(), m.uncertainty()['decision_proba'], m.uncertainty(), m


# ============================================================================ label audit
def audit_scores(Z, yk, T, seed, k=10):
    """Scores of 'this label is wrong' for every labelled cell, all computed in an unsupervised latent space:
    eknn_loo     : 1 - leave-one-out BetP(own label) / max BetP of the evidential k-NN (proposed)
    knn_disagree : share of the k nearest labelled cells with a different label (classic heuristic)
    cl_selfconf  : 1 - out-of-fold probability of the given label (self-confidence of confident learning,
                   Northcutt et al., 2021), 5-fold k-NN classifier."""
    from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
    from sklearn.model_selection import KFold
    lab = np.where(yk >= 0)[0]
    _, _, _, bl = head_eknn(Z, yk, T, seed)
    own = bl[np.arange(lab.size), yk[lab]]
    eknn_loo = 1.0 - own / np.maximum(bl.max(1), 1e-8)
    nb = NearestNeighbors(n_neighbors=min(k + 1, lab.size)).fit(Z[lab]).kneighbors(Z[lab], return_distance=False)
    yl = yk[lab]
    knn_dis = np.mean(yl[nb[:, 1:]] != yl[:, None], 1)
    oof = np.zeros(lab.size)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(lab):
        c = KNeighborsClassifier(n_neighbors=min(k, tr.size), weights='distance').fit(Z[lab[tr]], yl[tr])
        P = _full_proba(c.predict_proba(Z[lab[te]]), c.classes_, T)
        oof[te] = P[np.arange(te.size), yl[te]]
    return dict(eknn_loo=eknn_loo, knn_disagree=knn_dis, cl_selfconf=1.0 - oof)


# ============================================================================ evaluation + CSV
COLS = ['exp', 'ds', 'seed', 'frac', 'noise', 'hide', 'miss', 'encoder', 'head', 'param',
        'ARI', 'NMI', 'ACC', 'F1', 'ECE', 'AURC', 'acc50', 'n_cells', 'K', 'T', 'n_labelled',
        'time_encoder_s', 'time_head_s', 'extra']
KEY = COLS[:10]


def scores(st, pred, proba=None, mask=None):
    ev = st['ev'] if mask is None else (st['ev'] & mask)
    y = st['y']
    out = M.clustering_scores(y[ev], pred[ev]) if ev.sum() > 1 else dict(ARI=np.nan, NMI=np.nan, ACC=np.nan, F1=np.nan)
    out.update(ECE=np.nan, AURC=np.nan, acc50=np.nan)
    if proba is not None and st['T'] == st['K'] and proba.shape[1] == st['K'] and ev.sum() > 1:
        s = M.selective_scores(y[ev], proba[ev])
        out.update(ECE=s['ECE'], AURC=s['AURC'], acc50=s['acc@50'])
    return out


class Writer:
    def __init__(self, name):
        os.makedirs(RESULTS, exist_ok=True)
        self.path = os.path.join(RESULTS, name)
        self.keys = set()
        if os.path.exists(self.path):
            with open(self.path, newline='', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    self.keys.add(tuple(str(r[k]) for k in KEY))

    @staticmethod
    def key(row):
        return tuple(str(row.get(k, '')) for k in KEY)

    def done(self, **kw):
        return self.key(kw) in self.keys

    def write(self, row):
        row = dict(row)
        extra = {k: v for k, v in row.items() if k not in COLS}
        for k in extra:
            row.pop(k)
        if extra:
            row['extra'] = json.dumps(extra, default=float)
        new = not os.path.exists(self.path)
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, '') for k in COLS})
        self.keys.add(self.key(row))
