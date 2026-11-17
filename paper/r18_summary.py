"""Summaries of R18 (review experiments) for Appendix B.9 and Table 7."""
import glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from collect import ORDER, DSNAME, setting

from paths import R18, TIDY
FILES = sorted(glob.glob(os.path.join(R18, 'R18_*.csv')))
R = pd.concat([pd.read_csv(f) for f in FILES], ignore_index=True)
R['setting'] = R.apply(setting, axis=1)
R['param'] = R['param'].fillna('')
ex = R['extra'].fillna('{}').map(json.loads)
for k in ['it60', 'it500', 'ari_between', 'ARI60', 'changed', 'hidden_to_free', 'known_to_free', 'ACC_known_cells',
          'ls_alpha', 'ls_k', 'eknn_k']:
    R[k] = ex.map(lambda z: z.get(k, np.nan))
R = R.drop_duplicates(['ds', 'seed', 'setting', 'head', 'param'])
T = pd.read_csv(TIDY); T = T[T.phase == 'dev']
ev = T[T.method == 'EvPFC'].set_index(['setting', 'ds', 'seed']).ARI
SETS = ['frac1', 'frac5', 'frac10', 'noise20', 'hide2']
out = {}

# ---------------------------------------------------------------- tuned baselines
rows = []
for h, base in [('LS-tuned', 'LS'), ('eKNN-tuned', 'eKNN')]:
    b0 = T[T.method == base].set_index(['setting', 'ds', 'seed']).ARI
    x = R[R['head'] == h].set_index(['setting', 'ds', 'seed']).ARI
    for s in SETS:
        if s not in x.index.get_level_values(0):
            continue
        j = pd.concat([ev.loc[s], x.loc[s], b0.loc[s]], axis=1, keys=['ev', 't', 'f']).dropna()
        per = j.groupby(level=0).mean()
        rows.append(dict(head=h, setting=s, n_ds=len(per), evpfc=per.ev.mean(), tuned=per.t.mean(), fixed=per.f.mean(),
                         ds_wins=int((per.ev > per.t).sum()), pair_wins=int((j.ev > j.t).sum()), n=len(j),
                         p=wilcoxon(j.ev, j.t).pvalue if len(j) > 5 else np.nan))
TB = pd.DataFrame(rows); out['tuned'] = TB

# ---------------------------------------------------------------- convergence
c = R[R['head'] == 'EvPFC-cap500']
out['conv'] = dict(n=len(c), capped=int((c.it60 >= 60).sum()), reached500=int((c.it500 >= 500).sum()),
                   identical=int((c.ari_between >= 0.999999).sum()), min_ari_between=c.ari_between.min(),
                   max_changed=c.changed.max(), mean_dARI=(c.ARI - c.ARI60).mean(), max_abs_dARI=(c.ARI - c.ARI60).abs().max(),
                   per_ds=c.groupby('ds')[['it60', 'it500', 'ari_between', 'changed']].mean())

# ---------------------------------------------------------------- free clusters
f = R[(R['head'] == 'EvPFC') & R.param.str.startswith('free')].copy()
_h = T[(T.method == 'EvPFC') & (T.setting == 'hide2')][['ds', 'seed', 'ARI', 'hidden_to_free']].copy()
_h['param'] = 'free2'
_have = set(zip(f.param, f.ds, f.seed))
_h = _h[[('free2', d, s_) not in _have for d, s_ in zip(_h.ds, _h.seed)]]
f = pd.concat([f, _h], ignore_index=True)
out['freek'] = f.groupby(['param', 'ds'])[['ARI', 'hidden_to_free', 'known_to_free']].mean().groupby(level=0).agg(['mean'])
out['freek_ds'] = f.pivot_table(index='ds', columns='param', values='ARI')

# ---------------------------------------------------------------- sensitivity
s = R[(R['head'] == 'EvPFC') & ~R.param.str.startswith('free') & (R.param != '')]
k = R[(R['head'] == 'eKNN') & (R.param != '')]
rows = []
for (v, st), g in s.groupby(['param', 'setting']):
    j = g.set_index(['ds', 'seed']).ARI
    b = ev.loc[st]
    d = pd.concat([j, b], axis=1, keys=['v', 'b']).dropna()
    per = (d.v - d.b).groupby(level=0).mean()
    e = k[(k.param == v) & (k.setting == st)].set_index(['ds', 'seed']).ARI
    gap = pd.concat([j, e], axis=1, keys=['v', 'e']).dropna()
    rows.append(dict(variant=v, setting=st, n_ds=len(per), d_mean=per.mean(), d_min=per.min(), d_max=per.max(),
                     evpfc=d.v.groupby(level=0).mean().mean(),
                     gap_eknn=(gap.v - gap.e).groupby(level=0).mean().mean() if len(gap) else np.nan))
out['sens'] = pd.DataFrame(rows)

out['coverage'] = R.groupby(['ds', 'head']).size().unstack(fill_value=0)

if __name__ == '__main__':
    pd.set_option('display.width', 220)
    for k_, v in out.items():
        print('==', k_); print(v if not isinstance(v, pd.DataFrame) else v.round(4).to_string())


def tables():
    tb = out['tuned']
    lab = {'frac1': '1 % labels', 'frac5': '5 % labels', 'frac10': '10 % labels', 'noise20': '5 %, 20 % wrong',
           'hide2': '5 %, 2 types hidden'}
    L = ['| Setting | EvPFC | LS (fixed) | LS (tuned) | eKNN (fixed) | eKNN (tuned) | EvPFC ahead of LS (tuned) | EvPFC ahead of eKNN (tuned) |',
         '|---|---|---|---|---|---|---|---|']
    for s in SETS:
        a = tb[(tb['head'] == 'LS-tuned') & (tb.setting == s)].iloc[0]
        b = tb[(tb['head'] == 'eKNN-tuned') & (tb.setting == s)].iloc[0]
        L.append(f"| {lab[s]} | {a.evpfc:.3f} | {a.fixed:.3f} | {a.tuned:.3f} | {b.fixed:.3f} | {b.tuned:.3f} | "
                 f"{a.ds_wins}/{a.n_ds}; {a.pair_wins}/{a.n} | {b.ds_wins}/{b.n_ds}; {b.pair_wins}/{b.n} |")
    open('tables/tab_tuned.md', 'w').write('\n'.join(L))
    se = out['sens']
    names = {'k5': '$k = 5$', 'k20': '$k = 20$', 'k30': '$k = 30$', 'eps0.01': r'$\varepsilon = 0.01$',
             'eps0.3': r'$\varepsilon = 0.3$', 'lamq0.9': r'$\lambda$: 0.90 quantile', 'lamq0.99': r'$\lambda$: 0.99 quantile',
             'rho0': r'$\rho = 0$', 'rho2': r'$\rho = 2$'}
    L = ['| Variant (default) | 1 % labels | 5 % labels | 5 %, 20 % wrong |', '|---|---|---|---|']
    for v in names:
        cells = []
        for s in ['frac1', 'frac5', 'noise20']:
            r = se[(se.variant == v) & (se.setting == s)]
            cells.append('–' if not len(r) else f"{round(r.iloc[0].d_mean, 3) + 0:+.3f} ({round(r.iloc[0].d_min, 3) + 0:+.3f}, {round(r.iloc[0].d_max, 3) + 0:+.3f})")
        L.append(f"| {names[v]} | " + ' | '.join(cells) + ' |')
    open('tables/tab_sens.md', 'w').write('\n'.join(L))


tables()
