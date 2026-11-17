"""R12: does the 60-iteration cap of EvPFC change the partition?

For each data set and seed (5 % labels) EvPFC is selected and fitted as usual (cap 60). It is then refitted with
the selected (beta, tau) and a cap of 500 iterations. We report the number of iterations, the ARI between the two
partitions, and the ARI of each with the annotation.

    python r12_converge.py --cache cache --results results
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_core as C  # noqa: E402
from evpfc import select_and_fit, EvPFC  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402


def main(a):
    import pandas as pd
    rows = []
    for ds in a.datasets:
        y = np.load(os.path.join(a.cache, f'y_{ds}.npy'))
        for seed in a.seeds:
            Z = np.load(os.path.join(a.cache, 'latents', f'{ds}__ae__s{seed}__m0.0.npy')).astype(np.float64)
            st = C.make_setting(y, seed, 0.05)
            T, K, yk, ev = st['T'], st['K'], st['yk'], st['ev']
            m = select_and_fit(Z, yk, T, K, seed)
            S, Th = m.knowledge
            m2 = EvPFC(K, T, beta=m.selected['beta'], tau_scale=m.selected['tau_scale'], seed=seed,
                       max_iter=500).fit(Z, yk, S, Th)
            p1, p2 = m.predict(), m2.predict()
            rows.append(dict(ds=ds, seed=seed, it60=m.n_iter, it500=m2.n_iter, ari_between=ari(p1[ev], p2[ev]),
                             ari60=ari(st['y'][ev], p1[ev]), ari500=ari(st['y'][ev], p2[ev]),
                             changed=float((p1[ev] != p2[ev]).mean())))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.results, 'R12_converge.csv'), index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache'))
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--datasets', nargs='*', default=['SMAGE3K', 'Ma2020_1', 'PBMC10k_Multiome_feat'])
    ap.add_argument('--seeds', nargs='*', type=int, default=[3, 4, 5, 6, 7])
    main(ap.parse_args())
