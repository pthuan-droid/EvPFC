"""Evaluate the pre-registered criteria of docs/PREREG.md (steps 2a and 2b)."""
import argparse, os, json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = ['BMNC', 'Ma2020_1', 'PBMC10k_Multiome_feat', 'SMAGE3K']


def load(res, name):
    p = os.path.join(res, name)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    ex = pd.DataFrame([json.loads(x) if isinstance(x, str) and x else {} for x in d['extra']], index=d.index)
    return pd.concat([d.drop(columns='extra'), ex], axis=1)


def wx(a, b):
    j = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    if len(j) < 5 or np.allclose(j['a'], j['b']):
        return len(j), np.nan, j['a'].mean(), j['b'].mean()
    return len(j), wilcoxon(j['a'], j['b']).pvalue, j['a'].mean(), j['b'].mean()


def step2a(res, out):
    a = load(res, 'R8a_evpfc_ae.csv'); r7 = load(res, 'R7_evpfc.csv')
    r1 = load(res, 'R1_main.csv'); r2 = load(res, 'R2_noise.csv')
    if a is None:
        return
    a['setting'] = np.select([a['hide'] > 0, a['noise'] > 0], ['hide2', 'noise20'], 'frac' + (a['frac'] * 100).round().astype(int).astype(str))
    g = a.groupby(['setting', 'ds', 'head'])['ARI']
    t = (g.mean().round(3).astype(str) + ' ± ' + g.std(ddof=0).round(3).astype(str)).unstack('head')
    t.to_csv(os.path.join(out, 'R8a_table_ARI.csv'), encoding='utf-8-sig'); print(t.to_string())
    rows = []
    key = ['ds', 'seed']
    ev_ae = a[(a['head'] == 'EvPFC') & (a['frac'] == 0.05) & (a['noise'] == 0) & (a['hide'] == 0)].set_index(key)['ARI']
    # A1: vs own scPIKA (R1, same seeds and labelled cells)
    if r1 is not None:
        own = r1[(r1['encoder'] == 'own') & (r1['head'] == 'scPIKA') & (r1['frac'] == 0.05)].set_index(key)['ARI']
        for scope, dss in (('4 processed data sets', PROC), ('all data sets', None)):
            x = ev_ae[ev_ae.index.get_level_values(0).isin(dss)] if dss else ev_ae
            n, p, ma, mb = wx(x, own)
            rows.append(dict(criterion=f'A1 ({scope}) 5% labels: EvPFC-ae >= scPIKA(own) - 0.01', EvPFC_ae=ma, comparison=mb, n=n, p=p,
                             met=(ma >= mb - 0.01) if scope != 'all data sets' else np.nan))
    # A2: vs EvPFC on PCA (R7)
    if r7 is not None:
        pc = r7[(r7['head'] == 'EvPFC') & (r7['encoder'] == 'pca') & (r7['frac'] == 0.05) & (r7['noise'] == 0) & (r7['hide'] == 0)].set_index(key)['ARI']
        x = ev_ae[ev_ae.index.get_level_values(0).isin(PROC)]
        n, p, ma, mb = wx(x, pc)
        rows.append(dict(criterion='A2 (4 processed data sets) 5% labels: EvPFC-ae > EvPFC-PCA', EvPFC_ae=ma, comparison=mb, n=n, p=p, met=ma > mb))
    # A3: noise vs own scPIKA (R2)
    if r2 is not None:
        own2 = r2[(r2['encoder'] == 'own') & (r2['head'] == 'scPIKA') & (r2['noise'] == 0.2)].set_index(key)['ARI']
        x = a[(a['head'] == 'EvPFC') & (a['noise'] == 0.2)].set_index(key)['ARI']
        n, p, ma, mb = wx(x, own2)
        rows.append(dict(criterion='A3 20% noisy labels: EvPFC-ae > scPIKA(own)', EvPFC_ae=ma, comparison=mb, n=n, p=p, met=ma > mb))
    # extra: EvPFC-ae vs eKNN/LS on same latent, per setting
    for s in sorted(a['setting'].unique()):
        P = a[a['setting'] == s].pivot_table(index=key, columns='head', values='ARI')
        for o in ('eKNN', 'LS'):
            n, p, ma, mb = wx(P['EvPFC'], P[o])
            rows.append(dict(criterion=f'(secondary) {s}: EvPFC-ae vs {o}-ae', EvPFC_ae=ma, comparison=mb, n=n, p=p, met=np.nan))
    r = pd.DataFrame(rows); r.round(4).to_csv(os.path.join(out, 'R8a_criteria.csv'), index=False, encoding='utf-8-sig')
    print(r.round(4).to_string())


def step2b(res, out):
    b = load(res, 'R8b_conformal.csv')
    if b is None:
        return
    b['setting'] = b['param']
    g = b.groupby(['setting', 'ds', 'encoder', 'head'])
    cols = [c for c in ('ARI', 'hidden_to_free', 'known_to_free', 'auroc_novel', 'ACC_known_cells', 'ARI_hidden_cells') if c in b]
    t = g[cols].mean().round(3)
    t.to_csv(os.path.join(out, 'R8b_table.csv'), encoding='utf-8-sig'); print(t.to_string())
    h = b[b['setting'] == 'hide2']
    pair = h.groupby(['ds', 'encoder', 'head'])[['ARI', 'hidden_to_free', 'known_to_free']].mean().reset_index()
    c = pair[pair['head'] == 'EvPFC-conformal']; e = pair[pair['head'] == 'EvPFC']
    m = c.merge(e, on=['ds', 'encoder'], suffixes=('_conf', '_evpfc'))
    rows = [dict(criterion='B1 cells of hidden types sent to free clusters > 30%', value=m['hidden_to_free_conf'].mean(), threshold=0.30,
                 met=m['hidden_to_free_conf'].mean() > 0.30),
            dict(criterion='B2 ARI conformal >= EvPFC - 0.01', value=m['ARI_conf'].mean(), threshold=m['ARI_evpfc'].mean() - 0.01,
                 met=m['ARI_conf'].mean() >= m['ARI_evpfc'].mean() - 0.01),
            dict(criterion='B3 cells of known types sent to free clusters <= 0.07', value=m['known_to_free_conf'].mean(), threshold=0.07,
                 met=m['known_to_free_conf'].mean() <= 0.07)]
    fa = b[b['setting'] == 'falsealarm'].groupby('head')[['ARI', 'known_to_free']].mean()
    for hd in fa.index:
        rows.append(dict(criterion=f'B4 (reported) false alarms, {hd}: ARI / share sent to free clusters',
                         value=fa.loc[hd, 'ARI'], threshold=fa.loc[hd, 'known_to_free'], met=np.nan))
    r = pd.DataFrame(rows); r.round(4).to_csv(os.path.join(out, 'R8b_criteria.csv'), index=False, encoding='utf-8-sig')
    print(m.round(3).to_string()); print(r.round(4).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    a = ap.parse_args()
    out = os.path.join(a.results, 'summary'); os.makedirs(out, exist_ok=True)
    step2a(a.results, out); step2b(a.results, out)
