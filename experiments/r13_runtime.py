"""R13: run time of the clustering step (EvPFC with parameter selection, eKNN, LS) against the number of cells.

Uses the label-free autoencoder latent of SLN111 (seed 3) and random subsamples of 2,000-15,802 cells, 5 % labels.
The encoder is not timed here (see R6 for totalVI). Two CPU threads.

    python r13_runtime.py --cache cache --results results
"""
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_core as C  # noqa: E402
from evpfc import select_and_fit  # noqa: E402


def main(a):
    import pandas as pd, torch
    torch.set_num_threads(a.threads)
    y = np.load(os.path.join(a.cache, f'y_{a.dataset}.npy'))
    Z = np.load(os.path.join(a.cache, 'latents', f'{a.dataset}__ae__s3__m0.0.npy')).astype(np.float64)
    rows = []
    for n in a.sizes:
        for rep in range(a.reps):
            rng = np.random.RandomState(rep)
            idx = np.sort(rng.choice(len(y), min(n, len(y)), replace=False))
            st = C.make_setting(y[idx], rep, 0.05)
            T, K, yk, Zs = st['T'], st['K'], st['yk'], Z[idx]
            for head in ['EvPFC', 'eKNN', 'LS']:
                t0 = time.time()
                if head == 'EvPFC':
                    m = select_and_fit(Zs, yk, T, K, rep)
                elif head == 'eKNN':
                    C.head_eknn(Zs.astype(np.float32), yk, T, rep)
                else:
                    C.head_ls(Zs, yk, T)
                rows.append(dict(n=len(idx), rep=rep, method=head, time_s=time.time() - t0))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.results, 'R13_runtime.csv'), index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache'))
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--dataset', default='SLN111')
    ap.add_argument('--sizes', nargs='*', type=int, default=[2000, 5000, 10000, 15802])
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--threads', type=int, default=2)
    main(ap.parse_args())
