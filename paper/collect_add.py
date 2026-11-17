"""Append the results of the gap-filling runs (R14 kNN / E2E / scANVI, R15 SSFCM) to results_tidy.csv.

results_tidy.csv was produced by collect.py from the round-2 and EvPFC result files; this script adds the later runs
with the same conventions (main latent per data set, method names, setting labels, development / confirmation phase).
"""
import json, os
import numpy as np
import pandas as pd
from collect import MAIN, setting

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import RAW, OUT
RES = os.environ.get('RES', RAW)
KEYS = ['hidden_to_free', 'known_to_free', 'auroc_novel', 'ARI_hidden_cells', 'ACC_known_cells', 'alpha', 'n_iter']


def load(f, keep):
    p = os.path.join(RES, f)
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_csv(p)
    d = d[keep(d)].copy()
    d['src'] = f.split('_')[0]
    return d


def main():
    T = pd.read_csv(os.path.join(OUT, 'results_tidy_rebuilt.csv'))
    T = T[T.src != 'R14'] if 'src' in T else T
    T = T[~T['head'].isin(['SSFCM', 'scMDC', 'scEMC'])]
    new = [load('R14_knn_ae.csv', lambda d: d.encoder == d.ds.map(MAIN)),
           load('R14_knn_count.csv', lambda d: d.encoder == d.ds.map(MAIN)),
           load('R14_fill.csv', lambda d: d['head'].isin(['scPIKA', 'scANVI'])),
           load('R15_ssfcm.csv', lambda d: d.encoder == d.ds.map(MAIN)),
           load('R16_unsup.csv', lambda d: d['head'].isin(['scMDC', 'scEMC']))]
    N = pd.concat([d for d in new if len(d)], ignore_index=True)
    N['setting'] = N.apply(setting, axis=1)
    N['method'] = N['head'].replace({'scPIKA': 'E2E'})
    N['phase'] = np.where(N.seed >= 8, 'confirm', 'dev')
    ex = N['extra'].fillna('{}').map(lambda s: json.loads(s) if isinstance(s, str) and s.startswith('{') else {})
    for k in KEYS:
        v = ex.map(lambda z: z.get(k, np.nan))
        N[k] = N[k].where(N[k].notna(), v) if k in N else v
    A = pd.concat([T, N], ignore_index=True)
    A = A.drop_duplicates(subset=['ds', 'seed', 'setting', 'method'], keep='first')
    A.to_csv(os.path.join(OUT, 'results_tidy_rebuilt.csv'), index=False)
    print(N.groupby(['setting', 'method']).ds.nunique().unstack().fillna(0).astype(int))


if __name__ == '__main__':
    main()
