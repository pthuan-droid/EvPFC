"""R7: EvPFC (picture fuzzy clustering with an evidential prior) on the cached latent spaces of round 2.

    python r7_evpfc.py                    # all settings, seeds 3-7, results/R7_evpfc.csv
    python r7_evpfc.py --datasets SLN111 --settings frac1 hide2

Uses only cached, label-free latents (cache/latents/*.npy), so no scvi-tools model is retrained.
Every setting also re-runs the evidential k-NN (eKNN) and label spreading (LS) heads on the same latent
and the same labelled cells, so the comparison is paired.
Pre-registered success criteria (written before running):
    (a) 1 % labels: EvPFC ARI > eKNN ARI
    (b) 2 hidden types: share of hidden-type cells in free clusters > 30 % and ARI > eKNN
    (c) 5-10 % labels and 20 % label noise: EvPFC ARI >= eKNN ARI - 0.01
"""
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_data  # noqa: E402
import r2_core as C  # noqa: E402
from evpfc import select_and_fit  # noqa: E402
from r2_run import novelty_cols, base_row  # noqa: E402

ENC = {'SLN111': 'totalvi', 'SLN206': 'totalvi', 'CellMix': 'multivi', 'ASAP_PBMC': 'multivi',
       'BMMC_CITE': 'totalvi', 'BMMC_Multiome': 'multivi', 'PBMC3k_Multiome': 'multivi',
       'BMNC': 'pca', 'Ma2020_1': 'pca', 'PBMC10k_Multiome_feat': 'pca', 'SMAGE3K': 'pca'}
ABL = {'EvPFC-constw': dict(const_w=True), 'EvPFC-nopair': dict(use_pair=False),
       'EvPFC-norefusal': dict(use_refusal=False), 'EvPFC-plain': dict(use_pair=False, use_refusal=False)}
SETTINGS = {'frac1': dict(frac=0.01), 'frac5': dict(frac=0.05), 'frac10': dict(frac=0.10),
            'noise20': dict(frac=0.05, noise=0.2), 'hide2': dict(frac=0.05, hide=2)}


def labels(ds):
    f = os.path.join(C.CACHE, f'y_{ds}.npy')
    if os.path.exists(f):
        return np.load(f)
    y = r2_data.load(ds)['y']
    np.save(f, y)
    return y


def main(a):
    W = C.Writer(a.out)
    for ds in a.datasets:
        try:
            y = labels(ds)
        except Exception as e:
            print('[skipped]', ds, e); continue
        encs = a.encoders or ([ENC[ds]] + (['pca'] if a.with_pca and ENC[ds] != 'pca' else []))
        dfull = None
        for enc in encs:
            for seed in a.seeds:
                lseed = seed if seed <= 7 else seed - 5          # seeds 8-12 reuse the label-free latents of seeds 3-7
                f = os.path.join(C.CACHE, 'latents', f'{ds}__{enc}__s{lseed if not a.miss else seed}__m{a.miss if a.miss else 0.0}.npy')
                if not os.path.exists(f):
                    if enc in ('ae', 'pca'):                       # cheap: compute on the fly (cached afterwards)
                        dfull = dfull or r2_data.load(ds)
                        C.latent(ds, dfull, enc, lseed)
                    else:
                        print('[latent missing]', f); continue
                Z = np.load(f).astype(np.float64)
                for sname in a.settings:
                    s = SETTINGS[sname]
                    hide = s.get('hide', 0)
                    if hide and len(np.unique(y)) < 6:
                        continue
                    st = C.make_setting(y, seed, s['frac'], noise=s.get('noise', 0.0), hide=hide, miss=a.miss)
                    T, K, yk = st['T'], st['K'], st['yk']
                    cols = novelty_cols if hide else None
                    for head in a.heads:
                        r0 = base_row('R7', ds, st, seed, s['frac'], s.get('noise', 0.0), hide, a.miss, encoder=enc,
                                      head=head, time_encoder_s=0.0)
                        if W.done(**r0):
                            continue
                        t0 = time.time()
                        ex = {}
                        if head == 'EvPFC':
                            m = select_and_fit(Z, yk, T, K, seed)
                            P = m.betp(); pred = P.argmax(1)
                            ex = dict(novelty=P[:, T:].sum(1) + m.xi if K > T else m.xi)
                            extra = dict(beta=m.selected['beta'], tau_scale=m.selected['tau_scale'],
                                         mean_xi=float(m.xi.mean()), mean_eta=float(m.eta.mean()), n_iter=m.n_iter)
                        elif head == 'EvPFC-noprior':          # ablation: same model, beta = 0 (no knowledge)
                            m = select_and_fit(Z, yk, T, K, seed, betas=(0.0,))
                            P = m.betp(); pred = P.argmax(1)
                            ex = dict(novelty=P[:, T:].sum(1) + m.xi if K > T else m.xi)
                            extra = {}
                        elif head in ABL:                       # ablations of EvPFC (same selection grid)
                            m = select_and_fit(Z, yk, T, K, seed, **ABL[head])
                            P = m.betp(); pred = P.argmax(1)
                            ex = dict(novelty=P[:, T:].sum(1) + m.xi if K > T else m.xi)
                            extra = dict(beta=m.selected['beta'], tau_scale=m.selected['tau_scale'], n_iter=m.n_iter)
                        elif head == 'SSFCM':                    # Pedrycz & Waletzky (1997), same latent and selection protocol
                            from ssfcm import select_and_fit_ssfcm
                            P, info = select_and_fit_ssfcm(Z, yk, T, K, seed); pred = P.argmax(1)
                            ex = dict(novelty=P[:, T:].sum(1) if K > T else 1 - P.max(1)); extra = info
                        elif head == 'kNN':
                            pred, P = C.head_knn(Z, yk, T); ex = dict(novelty=1 - P.max(1)); extra = {}
                        elif head == 'eKNN':
                            pred, P, th, _ = C.head_eknn(Z.astype(np.float32), yk, T, seed); ex = dict(novelty=th); extra = {}
                        else:
                            pred, P = C.head_ls(Z, yk, T); ex = dict(novelty=1 - P.max(1)); extra = {}
                        row = dict(r0, time_head_s=time.time() - t0, **C.scores(st, pred, P), **extra)
                        if a.miss:
                            inc = st['drop']
                            row['ARI_incomplete_cells'] = C.scores(st, pred, None, inc)['ARI']
                            row['ARI_complete_cells'] = C.scores(st, pred, None, ~inc)['ARI']
                        if cols:
                            row.update(cols(st, pred, P, ex))
                        W.write(row)
                    print(time.strftime('%H:%M:%S'), ds, enc, seed, sname, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.environ.get('EVPFC_DATA', os.path.join(HERE, '..', 'datasets')))
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache'))
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--out', default='R7_evpfc.csv')
    ap.add_argument('--datasets', nargs='*', default=['SLN111', 'SLN206', 'CellMix', 'ASAP_PBMC'])
    ap.add_argument('--settings', nargs='*', default=list(SETTINGS))
    ap.add_argument('--seeds', nargs='*', type=int, default=[3, 4, 5, 6, 7])
    ap.add_argument('--heads', nargs='*', default=['EvPFC', 'EvPFC-noprior', 'eKNN', 'LS'])
    ap.add_argument('--miss', type=float, default=0.0, help='missing-view rate (latents cached by R4)')
    ap.add_argument('--with-pca', action='store_true')
    ap.add_argument('--encoders', nargs='*', default=None, help='override the encoder (e.g. ae)')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    r2_data.set_root(a.data_root)
    C.CACHE, C.RESULTS = a.cache, a.results
    C.OPTS['threads'] = a.threads
    main(a)
