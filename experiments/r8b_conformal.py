"""R8b: free clusters with a split-conformal novelty threshold (see PREREG_R8.md).

    python r8b_conformal.py            # seeds 8-12, results/R8b_conformal.csv

Settings: 'hide2' (2 cell types hidden from the labels, K = T + 2) and 'falsealarm' (no hidden type, but the
model is still given K = T + 2 clusters: the cost of the mechanism when nothing is novel).
Heads: EvPFC (as R7), EvPFC-conformal, eKNN, LS. Latents are the label-free latents of seeds 3-7 (seed - 5).
"""
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_data  # noqa: E402
import r2_core as C  # noqa: E402
from evpfc import select_and_fit, conformal_open_set_fit  # noqa: E402
from r2_run import base_row  # noqa: E402
from r7_evpfc import ENC, labels  # noqa: E402


def cols(st, pred, novelty, K_model):
    T = st['T']
    ev, y = st['ev'], st['y']
    hid = ev & (y >= T)
    known = ev & (y < T)
    out = dict(K_model=K_model,
               known_to_free=float(np.mean(pred[known] >= T)) if known.any() else np.nan,
               ACC_known_cells=float(np.mean(pred[known] == y[known])) if known.any() else np.nan)
    if hid.any():
        out.update(hidden_to_free=float(np.mean(pred[hid] >= T)),
                   ARI_hidden_cells=C.M.clustering_scores(y[hid], pred[hid])['ARI'] if hid.sum() > 10 else np.nan)
        if novelty is not None:
            out['auroc_novel'] = C.M.auroc(hid[ev], np.asarray(novelty)[ev])
    return out


def main(a):
    W = C.Writer(a.out)
    for ds in a.datasets:
        y = labels(ds)
        if len(np.unique(y)) < 6:
            print('[skipped, < 6 types]', ds); continue
        encs = a.encoders or [ENC[ds]]
        for enc in encs:
            for seed in a.seeds:
                f = os.path.join(C.CACHE, 'latents', f'{ds}__{enc}__s{seed - 5}__m0.0.npy')
                if not os.path.exists(f):
                    print('[latent missing]', f); continue
                Z = np.load(f).astype(np.float64)
                for setting in ('hide2', 'falsealarm'):
                    st = C.make_setting(y, seed, 0.05, hide=2 if setting == 'hide2' else 0)
                    T = st['T']
                    K_model = T + 2
                    st_eval = dict(st)
                    for head in ('EvPFC-conformal', 'EvPFC', 'eKNN', 'LS'):
                        r0 = base_row('R8b', ds, st, seed, 0.05, 0.0, 2 if setting == 'hide2' else 0,
                                      encoder=enc, head=head, time_encoder_s=0.0)
                        r0['param'] = setting
                        if W.done(**r0):
                            continue
                        t0 = time.time()
                        nov, extra = None, {}
                        if head == 'EvPFC-conformal':
                            pred, nov, extra = conformal_open_set_fit(Z, st['yk'], T, K_model, seed, alpha=0.05)
                        elif head == 'EvPFC':
                            m = select_and_fit(Z, st['yk'], T, K_model, seed)
                            P = m.betp(); pred = P.argmax(1); nov = P[:, T:].sum(1) + m.xi
                        elif head == 'eKNN':
                            pred, P, nov, _ = C.head_eknn(Z.astype(np.float32), st['yk'], T, seed)
                        else:
                            pred, P = C.head_ls(Z, st['yk'], T); nov = 1 - P.max(1)
                        row = dict(r0, time_head_s=time.time() - t0, **C.scores(st_eval, pred, None),
                                   **cols(st, pred, nov, K_model), **extra)
                        W.write(row)
                    print(time.strftime('%H:%M:%S'), ds, enc, seed, setting, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.environ.get('EVPFC_DATA', os.path.join(HERE, '..', 'datasets')))
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache'))
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--out', default='R8b_conformal.csv')
    ap.add_argument('--datasets', nargs='*', default=['SLN111', 'SLN206', 'BMNC', 'Ma2020_1', 'PBMC10k_Multiome_feat', 'SMAGE3K'])
    ap.add_argument('--encoders', nargs='*', default=None)
    ap.add_argument('--seeds', nargs='*', type=int, default=[8, 9, 10, 11, 12])
    a = ap.parse_args()
    r2_data.set_root(a.data_root)
    C.CACHE, C.RESULTS = a.cache, a.results
    main(a)
