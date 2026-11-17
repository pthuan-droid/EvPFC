"""Summary of R7 (EvPFC) with the pre-registered criteria and paired Wilcoxon tests (Holm)."""
import argparse, os, json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj


def main(res):
    d = pd.read_csv(os.path.join(res, 'R7_evpfc.csv'))
    ex = pd.DataFrame([json.loads(x) if isinstance(x, str) and x else {} for x in d['extra']], index=d.index)
    d = pd.concat([d.drop(columns='extra'), ex], axis=1)
    d['setting'] = np.select([d['hide'] > 0, d['noise'] > 0], ['hide2', 'noise20'], 'frac' + (d['frac'] * 100).round().astype(int).astype(str))
    d['group'] = np.where(d['encoder'] == 'pca', 'processed (PCA)', 'count data (totalVI/MultiVI)')
    out = os.path.join(res, 'summary'); os.makedirs(out, exist_ok=True)
    g = d.groupby(['setting', 'ds', 'head'])['ARI']
    tab = (g.mean().round(3).astype(str) + ' ± ' + g.std(ddof=0).round(3).astype(str)).unstack('head')
    tab.to_csv(os.path.join(out, 'R7_table_ARI.csv'), encoding='utf-8-sig'); print(tab.to_string())
    if 'auroc_novel' in d:
        h = d[d['setting'] == 'hide2'].groupby(['ds', 'head'])[['ARI', 'auroc_novel', 'hidden_to_free', 'ACC_known_cells']].mean().round(3)
        h.to_csv(os.path.join(out, 'R7_table_hide2.csv'), encoding='utf-8-sig'); print(h.to_string())
    rows = []
    for grp in d['group'].unique():
        for s in sorted(d['setting'].unique()):
            dd = d[(d['group'] == grp) & (d['setting'] == s)]
            P = dd.pivot_table(index=['ds', 'seed'], columns='head', values='ARI')
            for other in ('eKNN', 'LS', 'EvPFC-noprior'):
                if 'EvPFC' not in P or other not in P:
                    continue
                j = P[['EvPFC', other]].dropna()
                if len(j) < 5:
                    continue
                diff = j['EvPFC'] - j[other]
                p = wilcoxon(j['EvPFC'], j[other]).pvalue if not np.allclose(diff, 0) else 1.0
                rows.append(dict(group=grp, setting=s, A='EvPFC', B=other, n_pairs=len(j), mean_A=j['EvPFC'].mean(),
                                 mean_B=j[other].mean(), mean_diff=diff.mean(), wins=int((diff > 0).sum()),
                                 losses=int((diff < 0).sum()), p=p))
    w = pd.DataFrame(rows)
    w['p_holm'] = holm(w['p'])
    w.round(4).to_csv(os.path.join(out, 'R7_wilcoxon.csv'), index=False, encoding='utf-8-sig')
    print(w.round(4).to_string())
    # ---- pre-registered criteria (count-data latents)
    c = d[d['group'].str.startswith('count')]
    m = c.groupby(['setting', 'head'])['ARI'].mean().unstack('head')
    crit = []
    if 'frac1' in m.index:
        crit.append(('(a) 1% labels: EvPFC > eKNN', m.loc['frac1', 'EvPFC'], m.loc['frac1', 'eKNN'], m.loc['frac1', 'EvPFC'] > m.loc['frac1', 'eKNN']))
    for s in ('frac5', 'frac10', 'noise20'):
        if s in m.index:
            crit.append((f'(c) {s}: EvPFC >= eKNN - 0.01', m.loc[s, 'EvPFC'], m.loc[s, 'eKNN'], m.loc[s, 'EvPFC'] >= m.loc[s, 'eKNN'] - 0.01))
    if 'hide2' in m.index:
        hf = c[(c['setting'] == 'hide2') & (c['head'] == 'EvPFC')]['hidden_to_free'].mean()
        crit.append(('(b) 2 hidden types: ARI EvPFC > eKNN', m.loc['hide2', 'EvPFC'], m.loc['hide2', 'eKNN'], m.loc['hide2', 'EvPFC'] > m.loc['hide2', 'eKNN']))
        crit.append(('(b) 2 hidden types: cells of hidden types sent to free clusters > 30%', hf, 0.30, hf > 0.30))
    cr = pd.DataFrame(crit, columns=['criterion', 'EvPFC', 'comparison', 'met'])
    cr.to_csv(os.path.join(out, 'R7_criteria.csv'), index=False, encoding='utf-8-sig'); print(cr.to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    main(ap.parse_args().results)
