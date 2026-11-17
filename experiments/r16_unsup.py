"""R16: published unsupervised multi-omics clustering methods under our protocol.

  * scMDC (Lin et al., Nat. Commun. 2022), official code https://github.com/xianglin226/scMDC, run on the data sets
    with raw counts in both modalities (SLN111, SLN206: RNA + ADT; CellMix: RNA + ATAC peaks), with the settings of the
    authors' run script for CITE-seq (BMNC command) and for RNA + ATAC (PBMC10k command).
  * scEMC (Hu et al., Brief. Bioinform. 2024), official code https://github.com/DayuHuu/scEMC, run on the processed
    data sets with its own loader (min-max scaling of each view) and default settings, and on the count data sets after
    log-normalisation and selection of 2,000 variable features per view (as in the BMNC file distributed with scEMC).

Both methods are unsupervised: labels are never passed to them (y=None), K is set to the number of annotated types,
and their stopping rule (fraction of changed assignments < tol) does not use labels. ARI is computed on the cells that
are unlabelled in the 5 % draw of the same seed, as for all other methods, and also on all cells.

    python r16_unsup.py --method scMDC --datasets CellMix SLN111 SLN206 --seeds 3 4 5 6 7
Results: results/R16_unsup.csv (resumable).
"""
import argparse, os, sys, time, json, contextlib, io, tempfile
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_data  # noqa: E402
import r2_core as C  # noqa: E402

EXT = os.environ.get('EXT_DIR', os.path.join(HERE, 'ext'))


def dense(v):
    return v.toarray() if sp.issparse(v) else np.asarray(v)


def seed_all(seed):
    import random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_scmdc(x1, x2, K, seed, atac, device='cpu'):
    sys.path.insert(0, os.path.join(EXT, 'scMDC', 'src'))
    import torch
    import scanpy as sc
    from scMDC import scMultiCluster
    from preprocess import read_dataset, normalize
    from utils import geneSelection
    seed_all(seed)
    if atac:   # authors' RNA + ATAC command: --filter1 --filter2 --f1 2000 --f2 2000 -el 256 128 64 ...
        f1, f2, filt2 = 2000, 2000, True
        el, dl1, dl2 = [256, 128, 64], [64, 128, 256], [64, 128, 256]
        phi1 = phi2 = 0.005; sigma2 = 2.5; tau = 0.1
    else:      # authors' CITE-seq command: --filter1 (f1 = 1000 by default), default layers
        f1, f2, filt2 = 1000, 2000, False
        el, dl1, dl2 = [256, 64, 32, 16], [16, 64, 256], [16, 20]
        phi1 = phi2 = 0.001; sigma2 = 1.5; tau = 1.0
    x1 = x1[:, geneSelection(x1, n=f1, plot=False)]
    if filt2:
        x2 = x2[:, geneSelection(x2, n=f2, plot=False)]
    ads = []
    for x in (x1, x2):
        a = read_dataset(sc.AnnData(x.astype(np.float32)), transpose=False, test_split=False, copy=True)
        ads.append(normalize(a, size_factors=True, normalize_input=True, logtrans_input=True))
    a1, a2 = ads
    model = scMultiCluster(input_dim1=a1.n_vars, input_dim2=a2.n_vars, tau=tau, encodeLayer=el, decodeLayer1=dl1,
                           decodeLayer2=dl2, activation='elu', sigma1=2.5, sigma2=sigma2, gamma=0.1, cutoff=0.5,
                           phi1=phi1, phi2=phi2, device=device).to(device)
    wfile = os.path.join(tempfile.gettempdir(), f'scmdc_ae_{os.getpid()}.pth.tar')
    model.pretrain_autoencoder(X1=a1.X, X_raw1=a1.raw.X, sf1=a1.obs.size_factors.values, X2=a2.X, X_raw2=a2.raw.X,
                               sf2=a2.obs.size_factors.values, batch_size=256, epochs=400, ae_weights=wfile)
    y_pred, _ = model.fit(X1=a1.X, X_raw1=a1.raw.X, sf1=a1.obs.size_factors.values, X2=a2.X, X_raw2=a2.raw.X,
                          sf2=a2.obs.size_factors.values, y=None, n_clusters=K, batch_size=256, num_epochs=5000,
                          update_interval=1, tol=0.001, lr=1.0, save_dir=os.path.join(tempfile.gettempdir(), f'scmdc_{os.getpid()}'))
    sys.path.remove(os.path.join(EXT, 'scMDC', 'src'))
    for m in ['scMDC', 'preprocess', 'utils', 'layers']:
        sys.modules.pop(m, None)
    return np.asarray(y_pred)


def run_scemc(x1, x2, K, seed, device='cpu'):
    sys.path.insert(0, os.path.join(EXT, 'scEMC', 'scEMC'))
    import torch
    import scanpy as sc
    from sklearn.preprocessing import MinMaxScaler
    from network import scEMC
    from preprocess import read_dataset, normalize
    seed_all(seed)
    X = [MinMaxScaler().fit_transform(x.astype(np.float32)) for x in (x1, x2)]   # scEMC load_data.py

    def prep(x, size_factors=True, normalize_input=True, logtrans_input=True):   # scEMC.prepare_data
        a = read_dataset(sc.AnnData(np.array(x)), transpose=False, test_split=False, copy=True)
        return normalize(a, size_factors=size_factors, normalize_input=normalize_input, logtrans_input=logtrans_input)
    a1 = prep(X[0]); a2 = prep(X[1], False, False, False)
    model = scEMC(input_dim1=a1.n_vars, input_dim2=a2.n_vars, tau=1.0, encodeLayer1=[256, 64, 32, 8],
                  encodeLayer2=[256, 64, 32, 8], decodeLayer1=[24, 64, 256], decodeLayer2=[24, 20], activation='elu',
                  sigma1=2.5, sigma2=1.5, gamma=0.1, cutoff=0.5, phi1=0.001, phi2=0.001, device=device).to(device)
    wfile = os.path.join(tempfile.gettempdir(), f'scemc_ae_{os.getpid()}.pth.tar')
    model.pretrain_autoencoder(X1=a1.X, X_raw1=a1.raw.X, sf1=a1.obs.size_factors.values, X2=a2.X, X_raw2=a2.raw.X,
                               sf2=a2.obs.size_factors.values, batch_size=256, epochs=400, ae_weights=wfile)
    y_pred, _, _, _ = model.fit(X1=a1.X, X_raw1=a1.raw.X, sf1=a1.obs.size_factors.values, X2=a2.X, X_raw2=a2.raw.X,
                                sf2=a2.obs.size_factors.values, y=None, n_clusters=K, batch_size=256, num_epochs=500,
                                update_interval=1, tol=0.001, lr=1.0, save_dir=os.path.join(tempfile.gettempdir(), f'scemc_{os.getpid()}'),
                                lam1=1, lam2=1)
    sys.path.remove(os.path.join(EXT, 'scEMC', 'scEMC'))
    for m in ['network', 'preprocess', 'utils', 'layers']:
        sys.modules.pop(m, None)
    return np.asarray(y_pred)


def lognorm_hvg(x, n):
    """log-normalised, top-n variable features (input for scEMC on the count data sets)."""
    import scanpy as sc
    a = sc.AnnData(x.astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    if a.n_vars > n:
        sc.pp.highly_variable_genes(a, n_top_genes=n)
        a = a[:, a.var.highly_variable]
    return np.asarray(a.X)


def main(a):
    W = C.Writer(a.out)
    for ds in a.datasets:
        d = r2_data.load(ds)
        y = d['y']; K = int(len(np.unique(y)))
        x1, x2 = dense(d['views'][0]), dense(d['views'][1])
        counts = np.allclose(x1[:200], np.round(x1[:200])) and np.allclose(x2[:200], np.round(x2[:200]))
        atac = d['names'][1] == 'ATAC'
        for seed in a.seeds:
            st = C.make_setting(y, seed, 0.05)
            r0 = dict(exp='R16', ds=ds, seed=seed, frac=0.05, noise=0.0, hide=0, miss=0.0, param='', n_cells=len(y),
                      K=K, T=st['T'], n_labelled=int(st['lab'].sum()), encoder='own', head=a.method, time_encoder_s=0.0)
            if W.done(**r0):
                continue
            t0 = time.time()
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                if a.method == 'scMDC':
                    if not counts:
                        raise ValueError('scMDC needs counts')
                    pred = run_scmdc(x1, x2, K, seed, atac, a.device)
                else:
                    if counts:
                        x1p, x2p = lognorm_hvg(x1, 2000), lognorm_hvg(x2, 2000)
                    else:
                        x1p, x2p = x1, x2
                    pred = run_scemc(x1p, x2p, K, seed, a.device)
            row = dict(r0, time_head_s=time.time() - t0, **C.scores(st, pred, None))
            from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
            cnt = np.bincount(np.asarray(pred, int))
            row['extra'] = json.dumps(dict(ARI_all=float(ari(y, pred)), NMI_all=float(nmi(y, pred)),
                                           n_pred_clusters=int((cnt > 0).sum()), largest_cluster=float(cnt.max() / len(pred))))
            W.write(row)
            print(time.strftime('%H:%M:%S'), a.method, ds, seed, round(row['ARI'], 4), round(row['time_head_s']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.environ.get('EVPFC_DATA', os.path.join(HERE, '..', 'datasets')))
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--method', choices=['scMDC', 'scEMC'], required=True)
    ap.add_argument('--datasets', nargs='*', required=True)
    ap.add_argument('--seeds', nargs='*', type=int, default=[3, 4, 5, 6, 7])
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--device', default='cpu', help='cuda if a GPU is available')
    ap.add_argument('--out', default='R16_unsup.csv', help='result file inside --results')
    a = ap.parse_args()
    r2_data.set_root(a.data_root); C.RESULTS = a.results
    import torch; torch.set_num_threads(a.threads)
    main(a)
