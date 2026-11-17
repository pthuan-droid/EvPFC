"""Append the BMMC runs (R17 stage files and the BMMC rows of R1 / R2 / R3 / R16 made on the BMMC machine) to
results_tidy.csv, with the conventions of collect.py (main latent, method names, settings, phases)."""
import json, os
import numpy as np
import pandas as pd
from collect import MAIN, setting

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import BMMC, OUT
KQ = os.environ.get('KQ', BMMC)
BM = ['BMMC_CITE', 'BMMC_Multiome']
KEYS = ['hidden_to_free', 'known_to_free', 'auroc_novel', 'ARI_hidden_cells', 'ACC_known_cells', 'alpha', 'n_iter',
        'beta', 'tau_scale', 'mean_xi', 'mean_eta']
REN = {'scPIKA': 'E2E', 'seeded-kmeans': 'Seeded k-means', 'EvPFC-noprior': 'EvPFC w/o prior'}


def load(f, keep):
    p = os.path.join(KQ, f)
    if not os.path.exists(p):
        print('missing', f); return pd.DataFrame()
    d = pd.read_csv(p)
    d = d[d.ds.isin(BM)]
    d = d[keep(d)].copy()
    d['src'] = 'R17' if f.startswith('R17') else f.split('_')[0]
    return d


def main():
    T = pd.read_csv(os.path.join(OUT, 'results_tidy_rebuilt.csv'))
    T = T[~T.ds.isin(BM)]
    mainlat = lambda d: d.encoder == d.ds.map(MAIN)
    base = lambda d: d['head'].isin(['scANVI', 'seeded-kmeans']) | ((d.encoder == 'own') & (d['head'] == 'scPIKA'))
    new = [load('R17_evpfc.csv', mainlat), load('R17_confirm.csv', mainlat), load('R17_ablation.csv', mainlat),
           load('R17_scanvi.csv', lambda d: d['head'] == 'scANVI'),
           load('R1_main.csv', base), load('R2_noise.csv', base), load('R3_novel.csv', base),
           load('R16_unsup.csv', lambda d: d['head'].isin(['scMDC', 'scEMC']))]
    N = pd.concat([d for d in new if len(d)], ignore_index=True)
    N['setting'] = N.apply(setting, axis=1)
    N['method'] = N['head'].replace(REN)
    N['phase'] = np.where(N.seed >= 8, 'confirm', 'dev')
    ex = N['extra'].fillna('{}').map(lambda s: json.loads(s) if isinstance(s, str) and s.startswith('{') else {})
    for k in KEYS:
        v = ex.map(lambda z: z.get(k, np.nan))
        N[k] = N[k].where(N[k].notna(), v) if k in N else v
    N = N.drop_duplicates(subset=['ds', 'seed', 'setting', 'method'], keep='first')
    A = pd.concat([T, N], ignore_index=True)
    A.to_csv(os.path.join(OUT, 'results_tidy_rebuilt.csv'), index=False)
    print(N.groupby(['phase', 'setting', 'method']).size().unstack(0).fillna(0).astype(int).to_string())


if __name__ == '__main__':
    main()
