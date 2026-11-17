"""Tables 2, 3 and 4 (development seeds 3-7) from results_tidy.csv, including the gap-filling runs.

Table 2 cell format: mean ARI of the baseline; data sets on which EvPFC has the higher mean / data sets;
(data set, draw) pairs on which EvPFC is better / pairs; Holm-corrected two-sided Wilcoxon p over the pairs
(Holm over all settings x baselines of the table).
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from collect import ORDER, DSNAME

os.makedirs('tables', exist_ok=True)
from paths import TIDY
T = pd.read_csv(TIDY)
DEV = T[T.phase == 'dev']
SETS = [('frac1', '1 % labels'), ('frac5', '5 % labels'), ('frac10', '10 % labels'),
        ('noise20', '5 %, 20 % wrong'), ('hide2', '5 %, 2 types hidden')]
BASE = ['eKNN', 'LS', 'kNN', 'SSFCM']


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def main_table(setting='frac5', metric='ARI', name='tab_main'):
    meths = ['scMDC', 'scEMC', 'Seeded k-means', 'SSFCM', 'scANVI', 'E2E', 'kNN', 'LS', 'eKNN', 'EvPFC w/o prior', 'EvPFC']
    d = DEV[DEV.setting == setting]
    g = d.groupby(['method', 'ds'])[metric].agg(['mean', 'std'])
    rows = []
    for m in meths:
        r = [m]
        for ds in ORDER:
            if (m, ds) in g.index:
                mu, sd = g.loc[(m, ds)]
                gg = g.xs(ds, level='ds')['mean']
                best = mu >= gg[gg.index.isin(meths)].max() - 1e-9
                s = f'{mu:.3f} ({int(round(sd * 1000))})'
                r.append(f'**{s}**' if best else s)
            else:
                r.append('–')
        rows.append(r)
    hdr = '| Method | ' + ' | '.join(DSNAME[x] for x in ORDER) + ' |'
    md = '\n'.join([hdr, '|' + '---|' * (len(ORDER) + 1)] + ['| ' + ' | '.join(r) + ' |' for r in rows])
    open(f'tables/{name}.md', 'w').write(md)
    return g


def hboot(j, B=10000, seed=0):
    """Hierarchical bootstrap of the mean difference: resample data sets, then draws within each data set."""
    rng = np.random.default_rng(seed)
    groups = [g.values for _, g in (j.a - j.b).groupby(level=0)]
    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(groups), len(groups))
        stats[b] = np.mean([rng.choice(groups[i], len(groups[i])).mean() for i in idx])
    return np.percentile(stats, [2.5, 97.5])


def summary():
    res = []
    for s, _ in SETS:
        d = DEV[DEV.setting == s]
        a = d[d.method == 'EvPFC'].set_index(['ds', 'seed']).ARI
        for o in BASE:
            b = d[d.method == o].set_index(['ds', 'seed']).ARI
            j = pd.concat([a, b], axis=1, keys=['a', 'b']).dropna()
            if len(j) < 5:
                continue
            per = j.groupby(level=0).mean()
            res.append(dict(setting=s, other=o, ev=per.a.mean(), oth=per.b.mean(), n_ds=len(per),
                            ds_wins=int((per.a > per.b).sum()), n=len(j), wins=int((j.a > j.b).sum()),
                            p=wilcoxon(j.a, j.b).pvalue, p_ds=wilcoxon(per.a, per.b).pvalue,
                            delta=(per.a - per.b).mean(), ci=hboot(j)))
    R = pd.DataFrame(res)
    R['p_holm'] = holm(R.p)
    R['p_ds_holm'] = holm(R.p_ds)
    R['ci_lo'] = R.ci.map(lambda c: c[0]); R['ci_hi'] = R.ci.map(lambda c: c[1]); R = R.drop(columns='ci')
    R.to_csv('tables/summary_dev.csv', index=False)
    ev_all = DEV[DEV.method == 'EvPFC'].groupby(['setting', 'ds']).ARI.mean()
    lines = ['| Setting | EvPFC | ' + ' | '.join(BASE) + ' |', '|' + '---|' * (len(BASE) + 2)]
    for s, lab in SETS:
        cells = [f'{ev_all.loc[s].mean():.3f}']
        for o in BASE:
            r = R[(R.setting == s) & (R.other == o)]
            if not len(r):
                cells.append('–'); continue
            r = r.iloc[0]
            fm = lambda v: f'{v:+.3f}'.replace('-', '\u2212')
            cells.append(f'{r.oth:.3f}; {fm(r.delta)} [{fm(r.ci_lo)},\u00a0{fm(r.ci_hi)}]; {r.ds_wins}/{r.n_ds}')
        lines.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    open('tables/tab_summary.md', 'w').write('\n'.join(lines))
    return R


if __name__ == '__main__':
    main_table()
    main_table(metric='NMI', name='tab_main_nmi')
    R = summary()
    print(R.round(4).to_string())
    print(open('tables/tab_summary.md').read())
