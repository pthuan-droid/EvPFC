"""Collect all result CSVs into one tidy table (results_tidy.csv) used by the figures and tables of the paper.

Main latent per data set: totalVI (SLN111, SLN206), MultiVI (CellMix, ASAP-PBMC), autoencoder pre-training (others).
"""
import json, os
import numpy as np
import pandas as pd

from paths import RAW, OUT
R_EV = os.environ.get('EVPFC_RESULTS', RAW)
R_R2 = os.environ.get('R2_RESULTS', R_EV)
MAIN = {'SLN111': 'totalvi', 'SLN206': 'totalvi', 'CellMix': 'multivi', 'ASAP_PBMC': 'multivi',
        'BMMC_CITE': 'totalvi', 'BMMC_Multiome': 'multivi',
        'BMNC': 'ae', 'Ma2020_1': 'ae', 'PBMC10k_Multiome_feat': 'ae', 'SMAGE3K': 'ae'}
DSNAME = {'SLN111': 'SLN111', 'SLN206': 'SLN206', 'CellMix': 'CellMix', 'ASAP_PBMC': 'ASAP-PBMC',
          'BMMC_CITE': 'BMMC-CITE', 'BMMC_Multiome': 'BMMC-Multiome',
          'BMNC': 'BMNC', 'Ma2020_1': 'Ma-2020', 'PBMC10k_Multiome_feat': 'PBMC10k', 'SMAGE3K': 'SMAGE-3K'}
ORDER = ['SLN111', 'SLN206', 'CellMix', 'ASAP_PBMC', 'BMMC_CITE', 'BMMC_Multiome', 'BMNC', 'Ma2020_1', 'PBMC10k_Multiome_feat', 'SMAGE3K']
COUNT = ORDER[:6]


def setting(r):
    if r.miss > 0:
        return f'miss{int(round(r.miss * 100))}'
    if r.hide > 0:
        return 'hide2'
    if r.noise > 0:
        return f'noise{int(round(r.noise * 100))}'
    return f'frac{int(round(r.frac * 100))}'


def read(path, exp):
    if not os.path.exists(path):
        return pd.DataFrame()
    d = pd.read_csv(path)
    d['src'] = exp
    return d


def tidy():
    parts = []
    # EvPFC family, eKNN and LS on the main latents (development seeds 3-7)
    for f in ['R7_evpfc.csv', 'R8a_evpfc_ae.csv', 'R9_confirm.csv', 'R9_missing.csv', 'R10_ablation.csv']:
        d = read(os.path.join(R_EV, f), f.split('_')[0])
        if len(d):
            d = d[d.encoder == d.ds.map(MAIN)]
            parts.append(d)
    ev = pd.concat(parts)
    # baselines from round 2 (same label draws: label rng = RandomState(1000 + seed))
    r1 = read(os.path.join(R_R2, 'R1_main.csv'), 'R1')
    r2 = read(os.path.join(R_R2, 'R2_noise.csv'), 'R2')
    r3 = read(os.path.join(R_R2, 'R3_novel.csv'), 'R3')
    r4 = read(os.path.join(R_R2, 'R4_missing.csv'), 'R4')
    base = []
    for d in (r1, r2, r3, r4):
        b = d[(d['head'].isin(['scANVI', 'seeded-kmeans'])) |
              ((d.encoder == 'own') & d['head'].isin(['scPIKA', 'scPIKA-zero'])) |
              ((d['head'] == 'kNN') & (d.encoder == d.ds.map(MAIN)))]
        base.append(b)
    base = pd.concat(base)
    allr = pd.concat([ev, base], ignore_index=True)
    allr['setting'] = allr.apply(setting, axis=1)
    ren = {'scPIKA': 'E2E', 'scPIKA-zero': 'E2E (zero-fill)', 'seeded-kmeans': 'Seeded k-means',
           'EvPFC-noprior': 'EvPFC w/o prior'}
    allr['method'] = allr['head'].replace(ren)
    allr['phase'] = np.where(allr.seed >= 8, 'confirm', 'dev')
    ex = allr['extra'].fillna('{}').map(json.loads)
    for k in ['hidden_to_free', 'known_to_free', 'auroc_novel', 'ARI_hidden_cells', 'ACC_known_cells',
              'beta', 'tau_scale', 'n_iter', 'mean_xi', 'mean_eta', 'ARI_incomplete_cells', 'ARI_complete_cells']:
        v = ex.map(lambda z: z.get(k, np.nan))
        if k in allr:
            allr[k] = allr[k].where(allr[k].notna(), v)
        else:
            allr[k] = v
    # drop exact duplicates (same ds/seed/setting/method from two files): keep the first (EvPFC files first)
    allr = allr.drop_duplicates(subset=['ds', 'seed', 'setting', 'method'], keep='first')
    return allr


if __name__ == '__main__':
    t = tidy()
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(os.path.join(OUT, 'results_tidy_rebuilt.csv'), index=False)
    print(t.groupby(['phase', 'setting', 'method']).size().unstack(0).to_string())
