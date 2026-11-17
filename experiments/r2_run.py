"""Round 2 experiments (direction B: scPIKA as an evidential layer on top of any multi-omics latent space).

    python r2_run.py --exp R1            # main comparison: label budget 1 / 5 / 10 %
    python r2_run.py --exp R2            # corrupted labels: accuracy + noisy-label detection
    python r2_run.py --exp R3            # unseen cell types (2 types hidden from the labels)
    python r2_run.py --exp R4            # missing views (30 / 50 / 70 % of cells without ATAC) + mosaic ASAP
    python r2_run.py --exp R5            # hyper-parameter sensitivity
    python r2_run.py --exp R6            # running time vs number of cells
    python r2_run.py --exp all
Options: --datasets ... --seeds 3 4 5 6 7 --max-epochs N (cap for scvi-tools) --threads 4 --fast (smoke test)

Seeds 3-7 are used by default: seeds 0-2 were used while developing the method and are not reported.
Results are appended to results/<exp>_*.csv; finished rows are skipped, so the script can be stopped
and restarted at any time. Latent spaces are cached in cache/latents (unsupervised, label-free).
"""
import argparse, os, sys, time, json, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_data  # noqa: E402
import r2_core as C  # noqa: E402

FRACS = [0.01, 0.05, 0.10]
NOISES = [0.1, 0.2, 0.3]
MISSES = [0.3, 0.5, 0.7]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def err(r0, e):
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(os.path.join(C.RESULTS, 'errors.log'), 'a', encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + json.dumps({k: r0.get(k) for k in C.KEY}, default=str)
                + ' ' + repr(e) + '\n' + traceback.format_exc() + '\n')


def base_row(exp, ds, st, seed, frac, noise=0.0, hide=0, miss=0.0, **kw):
    r = dict(exp=exp, ds=ds, seed=seed, frac=frac, noise=noise, hide=hide, miss=miss, param='',
             n_cells=int(len(st['y'])), K=st['K'], T=st['T'], n_labelled=int(st['lab'].sum()))
    r.update(kw)
    return r


def run_heads(W, exp, ds, d, st, seed, frac, enc, Z, t_enc, heads, noise=0.0, hide=0, miss=0.0, extra_cols=None):
    """Run the requested heads on one latent space and write one row per head."""
    T, K, yk = st['T'], st['K'], st['yk']
    out = {}
    for h in heads:
        r0 = base_row(exp, ds, st, seed, frac, noise, hide, miss, encoder=enc, head=h, time_encoder_s=t_enc)
        if W.done(**r0):
            continue
        t0 = time.time()
        try:
            ex = {}
            if h == 'LS':
                pred, P = C.head_ls(Z, yk, T); ex['novelty'] = 1 - P.max(1)
            elif h == 'kNN':
                pred, P = C.head_knn(Z, yk, T); ex['novelty'] = 1 - P.max(1)
            elif h == 'eKNN':
                pred, P, th, _ = C.head_eknn(Z, yk, T, seed); ex['novelty'] = th
            elif h == 'eKNN-open':
                pred, P, th, _ = C.head_eknn(Z, yk, T, seed, know_open=True); ex['novelty'] = th
            elif h in ('scPIKA-L', 'scPIKA-L-open'):
                pred, P, U = C.head_scpika_latent(Z, yk, T, K, seed, mode='open' if h.endswith('open') else 'closed')
                ex['novelty'] = P[:, T:].sum(1) if K > T else U['know_theta']
                ex['kappa_K'] = U['kappa_K']; ex['U'] = U
            elif h == 'seeded-kmeans':
                pred = C.B.seeded_kmeans(Z, K, yk, seed=seed); P = None
            elif h == 'scANVI':
                C.SCANVI_FALLBACK.clear()
                pred, P = C.scanvi(ds, d, seed, yk, miss); ex['novelty'] = 1 - P.max(1)
                r0['scanvi_no_validation_split'] = bool(C.SCANVI_FALLBACK)
            else:
                raise ValueError(h)
            row = dict(r0, time_head_s=time.time() - t0, **C.scores(st, pred, P))
            if extra_cols:
                row.update(extra_cols(st, pred, P, ex))
            W.write(row)
            out[h] = (pred, P, ex)
        except Exception as e:
            log('ERROR', ds, enc, h, repr(e)); traceback.print_exc(); err(r0, e)
    return out


def own_model(W, exp, ds, d, st, seed, frac, noise=0.0, hide=0, miss=0.0, variants=('scPIKA',), extra_cols=None):
    """scPIKA with its own encoder (as in the paper). 'scPIKA-zero' = naive zero-filling of missing views."""
    for v in variants:
        r0 = base_row(exp, ds, st, seed, frac, noise, hide, miss, encoder='own', head=v, time_encoder_s=0.0)
        if W.done(**r0):
            continue
        cfg = {'scPIKA': {}, 'scPIKA-open': dict(C.CFG_OPEN), 'scPIKA-zero': dict(missing='zero')}[v]
        t0 = time.time()
        try:
            pred, P, U, m = C.head_scpika_own(d, st['yk'], st['T'], st['K'], seed, **cfg)
            row = dict(r0, time_head_s=time.time() - t0, **C.scores(st, pred, P))
            if extra_cols:
                ex = dict(U=U, novelty=(P[:, st['T']:].sum(1) if st['K'] > st['T'] else U['know_theta']),
                          kappa_K=U['kappa_K'])
                row.update(extra_cols(st, pred, P, ex))
            W.write(row)
        except Exception as e:
            log('ERROR', ds, 'own', v, repr(e)); traceback.print_exc(); err(r0, e)


def datasets_for(args, default):
    names = args.datasets or default
    return [n for n in names if n in r2_data.COUNT_DATASETS + r2_data.PROCESSED_DATASETS]


def load_or_skip(ds):
    try:
        return r2_data.load(ds)
    except Exception as e:
        log(f'[skipped {ds}]', e)
        return None


# ============================================================================ R1
def R1(args):
    W = C.Writer('R1_main.csv')
    for ds in datasets_for(args, r2_data.COUNT_DATASETS + r2_data.PROCESSED_DATASETS):
        d = load_or_skip(ds)
        if d is None:
            continue
        processed = ds in r2_data.PROCESSED_DATASETS
        for seed in args.seeds:
            for enc in C.encoders_for(d, processed):
                if args.encoders and enc not in args.encoders:
                    continue
                log('R1', ds, 'seed', seed, 'latent', enc)
                try:
                    Z, t = C.latent(ds, d, enc, seed)
                except Exception as e:
                    log('ERROR latent', ds, enc, repr(e)); traceback.print_exc(); continue
                for frac in args.fracs:
                    st = C.make_setting(d['y'], seed, frac)
                    heads = ['LS', 'kNN', 'eKNN']
                    if enc == 'pca':
                        heads += ['seeded-kmeans']
                    if enc in ('totalvi', 'multivi') or (processed and enc == 'pca'):
                        heads += ['scPIKA-L']
                    if enc == 'scvi':
                        heads += ['scANVI']
                    run_heads(W, 'R1', ds, d, st, seed, frac, enc, Z, t, heads)
            for frac in args.fracs:
                st = C.make_setting(d['y'], seed, frac)
                log('R1', ds, 'seed', seed, 'frac', frac, 'scPIKA (own encoder)')
                own_model(W, 'R1', ds, d, st, seed, frac)


# ============================================================================ R2
def R2(args):
    W = C.Writer('R2_noise.csv')
    WD = C.Writer('R2_detection.csv')
    for ds in datasets_for(args, ['SLN111', 'CellMix', 'PBMC3k_Multiome', 'BMMC_CITE', 'BMMC_Multiome', 'SLN206']):
        d = load_or_skip(ds)
        if d is None:
            continue
        for seed in args.seeds:
            for enc in [e for e in C.encoders_for(d) if e in ('pca', 'totalvi', 'multivi')]:
                Z, t = C.latent(ds, d, enc, seed)
                for noise in args.noises:
                    st = C.make_setting(d['y'], seed, 0.05, noise=noise)
                    log('R2', ds, seed, enc, 'noise', noise)
                    res = run_heads(W, 'R2', ds, d, st, seed, 0.05, enc, Z, t, ['LS', 'kNN', 'eKNN', 'scPIKA-L'], noise=noise)
                    lab = np.where(st['lab'])[0]
                    is_noisy = st['noisy'][lab]
                    # ---- detection of corrupted labels
                    r0 = base_row('R2', ds, st, seed, 0.05, noise, encoder=enc, head='audit', time_encoder_s=t)
                    t0 = time.time()
                    A = C.audit_scores(Z, st['yk'], st['T'], seed)
                    t_audit = time.time() - t0
                    if not WD.done(**r0):
                        row = dict(r0, time_head_s=t_audit, n_noisy=int(is_noisy.sum()))
                        for k, v in A.items():
                            row[f'auroc_{k}'] = C.M.auroc(is_noisy, v)
                        if 'scPIKA-L' in res:
                            row['auroc_kappaK_scPIKA-L'] = C.M.auroc(is_noisy, res['scPIKA-L'][2]['kappa_K'][lab])
                        WD.write(row)
                    # ---- audit-then-train: drop labels whose leave-one-out reliability is below 0.3
                    for h in ('LS', 'eKNN'):
                        r1 = base_row('R2', ds, st, seed, 0.05, noise, encoder=enc, head=h + '-clean', time_encoder_s=t)
                        if W.done(**r1):
                            continue
                        t0 = time.time() - t_audit
                        yk2 = st['yk'].copy()
                        yk2[lab[A['eknn_loo'] > 0.7]] = -1
                        pred, P = (C.head_ls(Z, yk2, st['T']) if h == 'LS' else C.head_eknn(Z, yk2, st['T'], seed)[:2])
                        W.write(dict(r1, time_head_s=time.time() - t0, n_dropped=int((yk2 < 0).sum() - (st['yk'] < 0).sum()),
                                     **C.scores(st, pred, P)))
            for noise in args.noises:
                st = C.make_setting(d['y'], seed, 0.05, noise=noise)
                own_model(W, 'R2', ds, d, st, seed, 0.05, noise=noise)


# ============================================================================ R3
def novelty_cols(st, pred, P, ex):
    hid = st['ev'] & (st['y'] >= st['T'])
    out = dict(n_hidden=int(hid.sum()))
    if 'novelty' in ex:
        out['auroc_novel'] = C.M.auroc(hid[st['ev']], np.asarray(ex['novelty'])[st['ev']])
    out['hidden_to_free'] = float(np.mean(pred[hid] >= st['T'])) if hid.any() else np.nan
    out['ARI_hidden_cells'] = C.M.clustering_scores(st['y'][hid], pred[hid])['ARI'] if hid.sum() > 10 else np.nan
    known = st['ev'] & (st['y'] < st['T'])
    out['ACC_known_cells'] = float(np.mean(pred[known] == st['y'][known]))
    return out


def R3(args):
    W = C.Writer('R3_novel.csv')
    for ds in datasets_for(args, ['SLN111', 'CellMix', 'PBMC3k_Multiome', 'BMMC_CITE', 'BMMC_Multiome', 'SLN206']):
        d = load_or_skip(ds)
        if d is None:
            continue
        if len(np.unique(d['y'])) < 5:
            log('R3 skipped', ds, '(too few cell types)'); continue
        for seed in args.seeds:
            st = C.make_setting(d['y'], seed, 0.05, hide=2)
            for enc in [e for e in C.encoders_for(d) if e in ('pca', 'totalvi', 'multivi')]:
                Z, t = C.latent(ds, d, enc, seed)
                log('R3', ds, seed, enc)
                run_heads(W, 'R3', ds, d, st, seed, 0.05, enc, Z, t,
                          ['LS', 'kNN', 'eKNN', 'eKNN-open', 'scPIKA-L', 'scPIKA-L-open'], hide=2, extra_cols=novelty_cols)
            own_model(W, 'R3', ds, d, st, seed, 0.05, hide=2, variants=('scPIKA', 'scPIKA-open'), extra_cols=novelty_cols)


# ============================================================================ R4
def R4(args):
    W = C.Writer('R4_missing.csv')
    for ds in datasets_for(args, ['CellMix', 'PBMC3k_Multiome', 'BMMC_Multiome', 'ASAP_PBMC']):
        d0 = load_or_skip(ds)
        if d0 is None:
            continue
        misses = [0.0] if ds == 'ASAP_PBMC' else args.misses
        mv = C.miss_view(d0)
        for seed in args.seeds:
            for miss in misses:
                st = C.make_setting(d0['y'], seed, 0.05, miss=miss)
                d = C.apply_missing(d0, st['drop'], mv)
                incomplete = ~d['mask'].all(1)

                def cols(st_, pred, P, ex, inc=incomplete):
                    return dict(ARI_incomplete_cells=C.scores(st_, pred, None, inc)['ARI'] if (st_['ev'] & inc).sum() > 10 else np.nan,
                                ARI_complete_cells=C.scores(st_, pred, None, ~inc)['ARI'] if (st_['ev'] & ~inc).sum() > 10 else np.nan,
                                frac_incomplete=float(inc.mean()))
                for enc in ('pca', 'multivi'):
                    try:
                        Z, t = C.latent(ds, d, enc, seed, miss=miss)
                    except Exception as e:
                        log('ERROR latent', ds, enc, repr(e)); continue
                    log('R4', ds, seed, 'miss', miss, enc)
                    heads = ['LS', 'eKNN', 'scPIKA-L'] + (['seeded-kmeans'] if enc == 'pca' else [])
                    run_heads(W, 'R4', ds, d, st, seed, 0.05, enc, Z, t, heads, miss=miss, extra_cols=cols)
                own_model(W, 'R4', ds, d, st, seed, 0.05, miss=miss, variants=('scPIKA', 'scPIKA-zero'), extra_cols=cols)


# ============================================================================ R5
GRID = {
    'eKNN': dict(knn_labeled=[5, 10, 20, 30], eps=[0.01, 0.05, 0.1, 0.2, 0.3]),
    'scPIKA-L': dict(w_know=[0.5, 0.6, 0.75, 0.9, 1.0], tau_scale=[0.05, 0.1, 0.2, 0.4], alpha=[0.01, 0.03, 0.1, 0.3]),
    'eKNN-open': dict(open_q=[0.8, 0.9, 0.95, 0.99]),
}


def R5(args):
    W = C.Writer('R5_sensitivity.csv')
    best_enc = {'SLN111': 'totalvi', 'BMMC_CITE': 'totalvi', 'CellMix': 'multivi', 'PBMC3k_Multiome': 'multivi',
                'BMMC_Multiome': 'multivi', 'SLN206': 'totalvi'}
    for ds in datasets_for(args, ['SLN111', 'CellMix', 'PBMC3k_Multiome']):
        d = load_or_skip(ds)
        if d is None:
            continue
        enc = best_enc.get(ds, 'pca')
        for seed in args.seeds[:3]:
            Z, t = C.latent(ds, d, enc, seed)
            for head, grid in GRID.items():
                hide = 2 if head == 'eKNN-open' else 0
                st = C.make_setting(d['y'], seed, 0.05, hide=hide)
                for p, values in grid.items():
                    for v in values:
                        r0 = base_row('R5', ds, st, seed, 0.05, hide=hide, encoder=enc, head=head, time_encoder_s=t)
                        r0['param'] = f'{p}={v}'
                        if W.done(**r0):
                            continue
                        log('R5', ds, seed, head, p, v)
                        t0 = time.time()
                        if head == 'scPIKA-L':
                            pred, P, U = C.head_scpika_latent(Z, st['yk'], st['T'], st['K'], seed, **{p: v})
                            ex = {}
                        else:
                            kw = {p: v}
                            if head == 'eKNN-open':
                                kw['know_open'] = True
                            pred, P, th, _ = C.head_eknn(Z, st['yk'], st['T'], seed, **kw)
                            ex = dict(novelty=th)
                        row = dict(r0, time_head_s=time.time() - t0, **C.scores(st, pred, P))
                        if hide:
                            row.update(novelty_cols(st, pred, P, ex))
                        W.write(row)


# ============================================================================ R6
def R6(args):
    W = C.Writer('R6_runtime.csv')
    ds = (args.datasets or ['BMMC_CITE'])[0]
    d = load_or_skip(ds)
    if d is None and ds != 'SLN111':
        ds = 'SLN111'; d = load_or_skip(ds)
    if d is None:
        return
    seed = args.seeds[0]
    n = len(d['y'])
    enc = 'totalvi' if 'ADT' in d['names'] and 'ATAC' not in d['names'] else 'multivi'
    for size in [s for s in (2000, 5000, 10000, 20000) if s < n] + [n]:
        idx = np.sort(np.random.RandomState(seed).choice(n, size, replace=False))
        ds_s = f'{ds}_n{size}'
        dsub = C.subsample(d, idx)
        st = C.make_setting(dsub['y'], seed, 0.05)
        for e in ('pca', enc):
            log('R6', ds_s, e)
            Z, t = C.latent(ds_s, dsub, e, seed)
            run_heads(W, 'R6', ds_s, dsub, st, seed, 0.05, e, Z, t, ['LS', 'eKNN', 'scPIKA-L'])
        own_model(W, 'R6', ds_s, dsub, st, seed, 0.05)


EXPS = dict(R1=R1, R2=R2, R3=R3, R4=R4, R5=R5, R6=R6)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='R1', help='R1..R6 or all')
    ap.add_argument('--data-root', default=os.environ.get('EVPFC_DATA', os.path.join(HERE, '..', 'datasets')))
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--encoders', nargs='*', default=None, help='R1 only: restrict encoders (pca scvi totalvi multivi)')
    ap.add_argument('--seeds', nargs='*', type=int, default=[3, 4, 5, 6, 7])
    ap.add_argument('--fracs', nargs='*', type=float, default=FRACS)
    ap.add_argument('--noises', nargs='*', type=float, default=NOISES)
    ap.add_argument('--misses', nargs='*', type=float, default=MISSES)
    ap.add_argument('--max-epochs', type=int, default=None, help='cap on scvi-tools epochs (default: their heuristic)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--bmmc-batches', nargs='*', default=None)
    ap.add_argument('--fast', action='store_true', help='smoke test: tiny epochs, results in results_fast/')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    r2_data.set_root(a.data_root)
    if a.bmmc_batches:
        r2_data.BMMC_BATCHES = a.bmmc_batches
    C.OPTS.update(max_epochs=a.max_epochs, threads=a.threads, verbose=a.verbose)
    if a.fast:
        C.OPTS['max_epochs'] = 3
        C.OPTS['cfg_extra'] = dict(pre_epochs=2, epochs=2)
        C.RESULTS = os.path.join(HERE, 'results_fast'); C.CACHE = os.path.join(HERE, 'cache_fast')
        os.environ['SCPIKA_FAST'] = '1'
    import torch
    torch.set_num_threads(a.threads)
    for e in (list(EXPS) if a.exp == 'all' else [a.exp]):
        log('==== bat dau', e)
        EXPS[e](a)
        log('==== done', e)
