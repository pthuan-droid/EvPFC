"""R18: additional experiments requested in the internal review (development seeds 3-7, main latent of each data set).

  ls       label spreading and the evidential k-NN with their parameters selected on the same 20 % held-out
           labelled cells as EvPFC (LS: alpha x graph degree; eKNN: k)               -> heads LS-tuned, eKNN-tuned
  conv     EvPFC refitted with the selected (beta, tau) and a cap of 500 iterations instead of 60 (5 % labels)
  freek    two hidden types, EvPFC given 1, 2 (true) or 4 free clusters
  sens     sensitivity of EvPFC to fixed parameters: k and eps of the evidential k-NN (1 %, 5 %, 20 % noise);
           lambda quantile and rho (5 %)

Only cached, label-free latents are used (cache/latents). Results: results/R18_review.csv (resumable).

    python r18_review.py                                   # all parts, the 8 data sets in the cache
    python r18_review.py --parts conv freek --datasets BMMC_CITE BMMC_Multiome
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_core as C  # noqa: E402
from evpfc import EvPFC, eknn_knowledge  # noqa: E402
from r2_run import novelty_cols, base_row  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

MAIN = {'SLN111': 'totalvi', 'SLN206': 'totalvi', 'CellMix': 'multivi', 'ASAP_PBMC': 'multivi',
        'BMMC_CITE': 'totalvi', 'BMMC_Multiome': 'multivi',
        'BMNC': 'ae', 'Ma2020_1': 'ae', 'PBMC10k_Multiome_feat': 'ae', 'SMAGE3K': 'ae'}
SETTINGS = {'frac1': dict(frac=0.01), 'frac5': dict(frac=0.05), 'frac10': dict(frac=0.10),
            'noise20': dict(frac=0.05, noise=0.2), 'hide2': dict(frac=0.05, hide=2)}
SENS = {'k5': (dict(knn_labeled=5), {}), 'k20': (dict(knn_labeled=20), {}), 'k30': (dict(knn_labeled=30), {}),
        'eps0.01': (dict(eps=0.01), {}), 'eps0.3': (dict(eps=0.3), {}),
        'lamq0.9': ({}, dict(lam_quantile=0.90)), 'lamq0.99': ({}, dict(lam_quantile=0.99)),
        'rho0': ({}, dict(rho=0.0)), 'rho2': ({}, dict(rho=2.0))}
SENS_SETTINGS = {'k5': 'frac1 frac5 noise20', 'k20': 'frac1 frac5 noise20', 'k30': 'frac1 frac5 noise20',
                 'eps0.01': 'frac1 frac5 noise20', 'eps0.3': 'frac1 frac5 noise20',
                 'lamq0.9': 'frac5', 'lamq0.99': 'frac5', 'rho0': 'frac5', 'rho2': 'frac5'}


def holdout(yk, seed, frac=0.2):
    """The same held-out labelled cells as evpfc.select_and_fit."""
    rng = np.random.RandomState(seed + 101)
    lab = np.where(yk >= 0)[0]
    return lab[rng.rand(lab.size) < frac]


def select_fit(Z, yk, T, K, seed, know=None, betas=(0.5, 1, 2, 4), taus=(0.1, 0.3, 1.0), **kw):
    """evpfc.select_and_fit with optional settings of the evidential k-NN (know)."""
    know = know or {}
    ho = holdout(yk, seed)
    lab = np.where(yk >= 0)[0]
    best = None
    if ho.size >= 5 and (lab.size - ho.size) >= T:
        ytr = yk.copy(); ytr[ho] = -1
        S, Th = eknn_knowledge(Z, ytr, T, seed, **know)
        for b in betas:
            for t in taus:
                m = EvPFC(K, T, beta=b, tau_scale=t, seed=seed, **kw).fit(Z, ytr, S, Th)
                P = m.betp()[ho]
                sc = (np.mean(P.argmax(1) == yk[ho]), np.mean(np.log(P[np.arange(ho.size), yk[ho]] + 1e-12)))
                if best is None or sc > best[0]:
                    best = (sc, b, t)
    b, t = (best[1], best[2]) if best else (1.0, 0.3)
    S, Th = eknn_knowledge(Z, yk, T, seed, **know)
    m = EvPFC(K, T, beta=b, tau_scale=t, seed=seed, **kw).fit(Z, yk, S, Th)
    m.knowledge, m.selected = (S, Th), dict(beta=b, tau_scale=t)
    return m


def ls_tuned(Z, yk, T, seed, alphas=(0.01, 0.1, 0.2, 0.5, 0.8, 0.99), ks=(5, 10, 15, 30)):
    from sklearn.semi_supervised import LabelSpreading
    ho = holdout(yk, seed)
    ytr = yk.copy(); ytr[ho] = -1
    best = None
    for a in alphas:
        for k in ks:
            try:
                ls = LabelSpreading(kernel='knn', n_neighbors=k, alpha=a, max_iter=100).fit(Z, ytr)
            except Exception:
                continue
            Pd = ls.label_distributions_[ho]
            cls = list(ls.classes_[ls.classes_ >= 0])
            idx = np.array([cls.index(c) if c in cls else -1 for c in yk[ho]])
            ll = np.mean(np.log(np.where(idx >= 0, Pd[np.arange(ho.size), np.maximum(idx, 0)], 0) + 1e-12))
            sc = (np.mean(ls.transduction_[ho] == yk[ho]), ll)    # accuracy, ties broken by log-likelihood
            if best is None or sc > best[0]:
                best = (sc, a, k)
    _, a, k = best
    ls = LabelSpreading(kernel='knn', n_neighbors=k, alpha=a, max_iter=100).fit(Z, yk)
    P = C._full_proba(ls.label_distributions_, ls.classes_[ls.classes_ >= 0], T)
    return P.argmax(1), P, dict(ls_alpha=a, ls_k=k)


def eknn_tuned(Z, yk, T, seed, ks=(5, 10, 20, 30)):
    ho = holdout(yk, seed)
    ytr = yk.copy(); ytr[ho] = -1
    best = None
    for k in ks:
        S, Th = eknn_knowledge(Z, ytr, T, seed, knn_labeled=k)
        P = S + Th[:, None] / T
        sc = (np.mean(P[ho].argmax(1) == yk[ho]), np.mean(np.log(P[ho, yk[ho]] + 1e-12)))
        if best is None or sc > best[0]:
            best = (sc, k)
    S, Th = eknn_knowledge(Z, yk, T, seed, knn_labeled=best[1])
    P = S + Th[:, None] / T
    return P.argmax(1), P, Th, dict(eknn_k=best[1])


def main(a):
    W = C.Writer(a.out)
    for ds in a.datasets:
        fy = os.path.join(C.CACHE, f'y_{ds}.npy')
        if not os.path.exists(fy):
            import r2_data
            r2_data.set_root(a.data_root)
            np.save(fy, r2_data.load(ds)['y'])
        y = np.load(fy)
        enc = MAIN[ds]
        for seed in a.seeds:
            f = os.path.join(C.CACHE, 'latents', f'{ds}__{enc}__s{seed}__m0.0.npy')
            if not os.path.exists(f):
                print('[latent missing]', f, flush=True); continue
            Z = np.load(f).astype(np.float64)
            jobs = []
            if 'ls' in a.parts:
                jobs += [(s, h, '') for s in SETTINGS for h in ('LS-tuned', 'eKNN-tuned')]
            if 'conv' in a.parts:
                jobs += [('frac5', 'EvPFC-cap500', '')]
            if 'freek' in a.parts:
                jobs += [('hide2', 'EvPFC', f'free{nf}') for nf in (1, 2, 4)]
            if 'sens' in a.parts:
                jobs += [(s, 'EvPFC', v) for v in SENS for s in SENS_SETTINGS[v].split()]
                jobs += [(s, 'eKNN', v) for v in ('k5', 'k20', 'k30', 'eps0.01', 'eps0.3') for s in SENS_SETTINGS[v].split()]
            for sname, head, param in jobs:
                s = SETTINGS[sname]
                hide = s.get('hide', 0)
                if hide and len(np.unique(y)) < 6:
                    continue
                st = C.make_setting(y, seed, s['frac'], noise=s.get('noise', 0.0), hide=hide)
                T, K, yk = st['T'], st['K'], st['yk']
                r0 = base_row('R18', ds, st, seed, s['frac'], s.get('noise', 0.0), hide, 0.0, encoder=enc,
                              head=head, time_encoder_s=0.0)
                r0['param'] = param
                if W.done(**r0):
                    continue
                t0 = time.time()
                ex, extra, P = {}, {}, None
                if head == 'LS-tuned':
                    pred, P, extra = ls_tuned(Z, yk, T, seed); ex = dict(novelty=1 - P.max(1))
                elif head == 'eKNN-tuned':
                    pred, P, th, extra = eknn_tuned(Z, yk, T, seed); ex = dict(novelty=th)
                elif head == 'EvPFC-cap500':
                    m = select_fit(Z, yk, T, K, seed)
                    m2 = EvPFC(K, T, beta=m.selected['beta'], tau_scale=m.selected['tau_scale'], seed=seed,
                               max_iter=500).fit(Z, yk, *m.knowledge)
                    p1 = m.predict(); pred = m2.predict()
                    extra = dict(it60=m.n_iter, it500=m2.n_iter, ari_between=float(ari(p1[st['ev']], pred[st['ev']])),
                                 ARI60=C.scores(st, p1)['ARI'], changed=float((p1[st['ev']] != pred[st['ev']]).mean()))
                elif head == 'EvPFC' and param.startswith('free'):
                    K2 = T + int(param[4:])
                    m = select_fit(Z, yk, T, K2, seed)
                    P = m.betp(); pred = P.argmax(1)
                    ex = dict(novelty=P[:, T:].sum(1) + m.xi)
                    extra = dict(K_used=K2, beta=m.selected['beta'], tau_scale=m.selected['tau_scale'])
                elif head == 'EvPFC':
                    know, kw = SENS[param]
                    m = select_fit(Z, yk, T, K, seed, know=know, **kw)
                    pred = m.predict()
                    extra = dict(beta=m.selected['beta'], tau_scale=m.selected['tau_scale'], n_iter=m.n_iter)
                elif head == 'eKNN':
                    know, _ = SENS[param]
                    S, Th = eknn_knowledge(Z, yk, T, seed, **know)
                    pred = (S + Th[:, None] / T).argmax(1)
                row = dict(r0, time_head_s=time.time() - t0, **C.scores(st, pred, None), **extra)
                if hide:
                    row.update(novelty_cols(st, pred, P, ex))
                    known = st['ev'] & (st['y'] < st['T'])
                    row['known_to_free'] = float(np.mean(pred[known] >= T))
                W.write(row)
            print(time.strftime('%H:%M:%S'), ds, seed, 'done', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.environ.get('EVPFC_DATA', os.path.join(HERE, '..', 'datasets')))
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache'))
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--out', default='R18_review.csv')
    ap.add_argument('--datasets', nargs='*', default=['SLN111', 'SLN206', 'CellMix', 'ASAP_PBMC', 'BMNC', 'Ma2020_1',
                                                      'PBMC10k_Multiome_feat', 'SMAGE3K'])
    ap.add_argument('--parts', nargs='*', default=['ls', 'conv', 'freek', 'sens'])
    ap.add_argument('--seeds', nargs='*', type=int, default=[3, 4, 5, 6, 7])
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    C.CACHE, C.RESULTS = a.cache, a.results
    C.OPTS['threads'] = a.threads
    import torch; torch.set_num_threads(a.threads)
    main(a)
