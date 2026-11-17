"""R14: fill the gaps of the comparison (development seeds 3-7, 5 % labels).

  * E2E (scPIKA with its own label-fitted encoder) under 20 % label noise on the processed data sets and ASAP-PBMC,
    and with two hidden types on the processed data sets;
  * scANVI under 20 % label noise (and with two hidden types) on the data sets with RNA counts in every cell.

    python r14_fill.py --data-root ../../datasets --what e2e scanvi
Results: results/R14_fill.csv (resumable).
"""
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_data  # noqa: E402
import r2_core as C  # noqa: E402
from r2_run import own_model, run_heads, novelty_cols, log  # noqa: E402


def main(a):
    W = C.Writer(a.out)
    if 'e2e' in a.what:
        for ds in a.e2e_datasets:
            d = r2_data.load(ds)
            for seed in a.seeds:
                st = C.make_setting(d['y'], seed, 0.05, noise=0.2)
                log('R14 E2E noise', ds, seed); own_model(W, 'R14', ds, d, st, seed, 0.05, noise=0.2)
                if len(np.unique(d['y'])) >= 6:
                    st = C.make_setting(d['y'], seed, 0.05, hide=2)
                    log('R14 E2E hide2', ds, seed)
                    own_model(W, 'R14', ds, d, st, seed, 0.05, hide=2, extra_cols=novelty_cols)
    if 'scanvi' in a.what:
        for ds in a.scanvi_datasets:
            d = r2_data.load(ds)
            for seed in a.seeds:
                st = C.make_setting(d['y'], seed, 0.05, noise=0.2)
                log('R14 scANVI noise', ds, seed)
                run_heads(W, 'R14', ds, d, st, seed, 0.05, 'scvi', None, 0.0, ['scANVI'], noise=0.2)
                if a.scanvi_hide and len(np.unique(d['y'])) >= 6:
                    st = C.make_setting(d['y'], seed, 0.05, hide=2)
                    log('R14 scANVI hide2', ds, seed)
                    run_heads(W, 'R14', ds, d, st, seed, 0.05, 'scvi', None, 0.0, ['scANVI'], hide=2, extra_cols=novelty_cols)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.environ.get('EVPFC_DATA', os.path.join(HERE, '..', 'datasets')))
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache'))
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--what', nargs='*', default=['e2e', 'scanvi'])
    ap.add_argument('--e2e-datasets', nargs='*', default=['SMAGE3K', 'Ma2020_1', 'PBMC10k_Multiome_feat', 'ASAP_PBMC', 'BMNC'])
    ap.add_argument('--scanvi-datasets', nargs='*', default=['CellMix', 'SLN111', 'SLN206'])
    ap.add_argument('--scanvi-hide', action='store_true')
    ap.add_argument('--seeds', nargs='*', type=int, default=[3, 4, 5, 6, 7])
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', default='R14_fill.csv')
    a = ap.parse_args()
    r2_data.set_root(a.data_root)
    C.CACHE, C.RESULTS = a.cache, a.results
    C.OPTS.update(threads=a.threads, max_epochs=None)
    try:
        import torch; torch.set_num_threads(a.threads)
    except Exception:
        pass
    main(a)
