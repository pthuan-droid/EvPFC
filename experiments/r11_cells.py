"""R11: per-cell analysis of EvPFC (where does the gain over the evidential k-NN come from?).

For each processed data set (autoencoder latent), seeds 3-7, 5 % labels:
  * unlabelled cells are split into five equal-size bins of the knowledge ignorance m_i(Theta)
  * in every bin we record the accuracy of EvPFC and of the evidential k-NN (eKNN) on that bin
  * per-cell masses (m(Theta), mu_max, eta, xi, max BetP) are saved for every seed (latent Z for the first seed)

    python r11_cells.py --cache cache --results results
Outputs: results/R11_bins.csv, results/R11_cells_<ds>_s<seed>.npz
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r2_core as C  # noqa: E402
from evpfc import select_and_fit  # noqa: E402


ENC = {'SLN111': 'totalvi', 'SLN206': 'totalvi', 'CellMix': 'multivi', 'ASAP_PBMC': 'multivi',
       'BMMC_CITE': 'totalvi', 'BMMC_Multiome': 'multivi'}


def main(a):
    rows = []
    for ds in a.datasets:
        y = np.load(os.path.join(a.cache, f'y_{ds}.npy'))
        for seed in a.seeds:
            enc = a.encoder or ENC.get(ds, 'ae')
            Z = np.load(os.path.join(a.cache, 'latents', f'{ds}__{enc}__s{seed}__m0.0.npy')).astype(np.float64)
            st = C.make_setting(y, seed, 0.05)
            T, K, yk = st['T'], st['K'], st['yk']
            m = select_and_fit(Z, yk, T, K, seed)
            pe = m.predict()
            pk, Pk, th, _ = C.head_eknn(Z.astype(np.float32), yk, T, seed)
            ev = st['ev']
            yt = st['y']
            q = np.quantile(th[ev], [0.2, 0.4, 0.6, 0.8])
            b = np.digitize(th, q)
            for j in range(5):
                s = ev & (b == j)
                rows.append(dict(ds=ds, enc=enc, seed=seed, bin=j, n=int(s.sum()), theta_mean=float(th[s].mean()),
                                 acc_evpfc=float((pe[s] == yt[s]).mean()), acc_eknn=float((pk[s] == yt[s]).mean()),
                                 eta_mean=float(m.eta[s].mean()), xi_mean=float(m.xi[s].mean())))
            P = m.betp()
            extra = dict(Z=Z.astype(np.float32)) if seed == a.seeds[0] else {}
            np.savez_compressed(os.path.join(a.results, f'R11_cells_{ds}_s{seed}.npz'), y=yt, yk=yk, ev=ev,
                                theta=th, pred_evpfc=pe, pred_eknn=pk, mu=m.mu.max(1), pmax=P.max(1),
                                eta=m.eta, xi=m.xi, Peknn_max=Pk.max(1), **extra)
            print(ds, seed, flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(a.results, a.out), index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache'))
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--datasets', nargs='*', default=['BMNC', 'Ma2020_1', 'PBMC10k_Multiome_feat', 'SMAGE3K'])
    ap.add_argument('--seeds', nargs='*', type=int, default=[3, 4, 5, 6, 7])
    ap.add_argument('--encoder', default=None, help='override; default: main latent of each data set')
    ap.add_argument('--out', default='R11_bins.csv')
    a = ap.parse_args()
    main(a)
