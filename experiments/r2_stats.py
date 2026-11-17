"""Summaries and statistical tests of the round-2 CSV files.

    python r2_stats.py                 # reads results/*.csv, writes results/summary/*.csv
    python r2_stats.py --results results_fast

Outputs (all CSV):
    <exp>_summary.csv          mean, sd, n of every metric per setting x method
    R1_table_frac<f>.csv       paper-style table (method x dataset, 'mean ± sd') for each label fraction
    R1_friedman_frac<f>.csv    Friedman test over (dataset, seed) blocks + average rank of every method
    R1_wilcoxon_frac<f>.csv    paired Wilcoxon signed-rank tests of the planned comparisons, Holm-adjusted
    R2_detection_table.csv     AUROC of the noisy-label scores; R2_detection_wilcoxon.csv
    R3_novel_table.csv, R4_missing_table.csv, R5_sensitivity_table.csv, R6_runtime_table.csv
"""
import argparse, os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SET = ['exp', 'ds', 'frac', 'noise', 'hide', 'miss']
METH = ['encoder', 'head', 'param']


def load(res, name):
    p = os.path.join(res, name)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    if 'extra' in d and d['extra'].notna().any():
        ex = pd.DataFrame([json.loads(x) if isinstance(x, str) and x else {} for x in d['extra']], index=d.index)
        d = pd.concat([d.drop(columns='extra'), ex], axis=1)
    d['param'] = d['param'].fillna('')
    d['method'] = d['encoder'] + '|' + d['head'] + np.where(d['param'] != '', '|' + d['param'], '')
    return d


def summary(d):
    num = [c for c in d.columns if c not in SET + METH + ['seed', 'method'] and pd.api.types.is_numeric_dtype(d[c])]
    g = d.groupby(SET + ['method'], dropna=False)[num]
    out = g.mean().add_suffix('_mean').join(g.std(ddof=0).add_suffix('_sd')).join(g.size().rename('n_seeds'))
    return out.reset_index()


def pm(m, s):
    return '' if pd.isna(m) else f'{m:.3f} ± {0 if pd.isna(s) else s:.3f}'


def table(d, metric, rows='method', cols='ds'):
    g = d.groupby([rows, cols])[metric]
    t = (g.mean().combine(g.std(ddof=0), pm)).unstack(cols)
    return t


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def paired(d, a, b, metric='ARI', block=('ds', 'seed')):
    A = d[d['method'] == a].set_index(list(block))[metric]
    Bv = d[d['method'] == b].set_index(list(block))[metric]
    j = A.to_frame('a').join(Bv.to_frame('b'), how='inner').dropna()
    return j


def wilcoxon_rows(d, pairs, metric='ARI', block=('ds', 'seed')):
    from scipy.stats import wilcoxon
    rows = []
    for a, b in pairs:
        j = paired(d, a, b, metric, block)
        if len(j) < 5 or np.allclose(j['a'], j['b']):
            continue
        st = wilcoxon(j['a'], j['b'])
        rows.append(dict(A=a, B=b, metric=metric, n_pairs=len(j), mean_A=j['a'].mean(), mean_B=j['b'].mean(),
                         mean_diff=(j['a'] - j['b']).mean(), wins_A=int((j['a'] > j['b']).sum()),
                         losses_A=int((j['a'] < j['b']).sum()), p=st.pvalue))
    out = pd.DataFrame(rows)
    if len(out):
        out['p_holm'] = holm(out['p'])
    return out


def friedman(d, metric='ARI'):
    from scipy.stats import friedmanchisquare
    P = d.pivot_table(index=['ds', 'seed'], columns='method', values=metric)
    P = P.dropna(axis=1, thresh=int(0.8 * len(P))).dropna(axis=0)
    if P.shape[1] < 3 or P.shape[0] < 3:
        return pd.DataFrame()
    st = friedmanchisquare(*[P[c] for c in P.columns])
    ranks = P.rank(axis=1, ascending=False).mean().sort_values()
    out = ranks.rename('avg_rank').reset_index()
    out['n_blocks'] = len(P); out['friedman_chi2'] = st.statistic; out['friedman_p'] = st.pvalue
    return out


def main(res):
    out = os.path.join(res, 'summary'); os.makedirs(out, exist_ok=True)
    w = lambda df, name: (df.to_csv(os.path.join(out, name), index=name.endswith('table.csv') or 'table_' in name,
                                    encoding='utf-8-sig'), print('->', name))
    # ---------------- R1
    d = load(res, 'R1_main.csv')
    if d is not None:
        w(summary(d), 'R1_summary.csv')
        for f in sorted(d['frac'].unique()):
            df = d[d['frac'] == f]
            w(table(df, 'ARI'), f'R1_table_frac{f}.csv')
            w(table(df, 'NMI'), f'R1_table_NMI_frac{f}.csv')
            w(friedman(df), f'R1_friedman_frac{f}.csv')
            pairs = []
            for enc in df['encoder'].unique():
                pairs += [(f'{enc}|eKNN', f'{enc}|LS'), (f'{enc}|eKNN', f'{enc}|kNN'), (f'{enc}|scPIKA-L', f'{enc}|LS'),
                          (f'{enc}|scPIKA-L', f'{enc}|eKNN')]
            for enc in ('totalvi', 'multivi'):
                pairs += [(f'{enc}|eKNN', 'scvi|scANVI'), (f'{enc}|scPIKA-L', 'own|scPIKA'), (f'{enc}|eKNN', 'own|scPIKA')]
            pairs += [('own|scPIKA', 'pca|LS'), ('own|scPIKA', 'pca|seeded-kmeans')]
            w(wilcoxon_rows(df, pairs), f'R1_wilcoxon_frac{f}.csv')
    # ---------------- R2
    d = load(res, 'R2_noise.csv')
    if d is not None:
        w(summary(d), 'R2_summary.csv')
        for nz in sorted(d['noise'].unique()):
            w(table(d[d['noise'] == nz], 'ARI'), f'R2_table_noise{nz}.csv')
        pairs = [(f'{e}|{h}-clean', f'{e}|{h}') for e in d['encoder'].unique() for h in ('LS', 'eKNN')]
        pairs += [(f'{e}|eKNN', f'{e}|LS') for e in d['encoder'].unique()]
        w(wilcoxon_rows(d, pairs, block=('ds', 'seed', 'noise')), 'R2_wilcoxon.csv')
    dd = load(res, 'R2_detection.csv')
    if dd is not None:
        cols = [c for c in dd.columns if c.startswith('auroc_')]
        long = dd.melt(id_vars=['ds', 'seed', 'noise', 'encoder'], value_vars=cols, var_name='score', value_name='AUROC')
        long['method'] = long['encoder'] + '|' + long['score']
        g = long.groupby(['noise', 'method', 'ds'])['AUROC']
        t = g.mean().combine(g.std(ddof=0), pm).unstack('ds')
        w(t, 'R2_detection_table.csv')
        long2 = long.copy(); long2['seed_noise'] = long2['seed'].astype(str) + '_' + long2['noise'].astype(str)
        pairs = []
        for e in dd['encoder'].unique():
            pairs += [(f'{e}|auroc_eknn_loo', f'{e}|auroc_knn_disagree'), (f'{e}|auroc_eknn_loo', f'{e}|auroc_cl_selfconf'),
                      (f'{e}|auroc_eknn_loo', f'{e}|auroc_kappaK_scPIKA-L')]
        w(wilcoxon_rows(long2, pairs, 'AUROC', block=('ds', 'seed_noise')), 'R2_detection_wilcoxon.csv')
    # ---------------- R3 .. R6
    for name, key in (('R3_novel.csv', 'R3'), ('R4_missing.csv', 'R4'), ('R5_sensitivity.csv', 'R5'), ('R6_runtime.csv', 'R6')):
        d = load(res, name)
        if d is None:
            continue
        w(summary(d), f'{key}_summary.csv')
        if key == 'R3':
            for m in ('ARI', 'auroc_novel', 'hidden_to_free', 'ACC_known_cells'):
                if m in d:
                    w(table(d, m), f'R3_table_{m}.csv')
            pairs = []
            for e in d['encoder'].unique():
                pairs += [(f'{e}|scPIKA-L-open', f'{e}|LS'), (f'{e}|eKNN-open', f'{e}|LS'), (f'{e}|scPIKA-L-open', f'{e}|scPIKA-L')]
            if 'auroc_novel' in d:
                w(wilcoxon_rows(d, pairs, 'auroc_novel'), 'R3_wilcoxon_auroc.csv')
            w(wilcoxon_rows(d, pairs, 'ARI'), 'R3_wilcoxon_ARI.csv')
        if key == 'R4':
            d['setting'] = d['ds'] + ' miss=' + d['miss'].astype(str)
            for m in ('ARI', 'ARI_incomplete_cells'):
                if m in d:
                    w(table(d, m, cols='setting'), f'R4_table_{m}.csv')
        if key == 'R5':
            d['setting'] = d['ds'] + ' ' + d['head']
            w(table(d, 'ARI', rows='param', cols='setting'), 'R5_table_ARI.csv')
            if 'auroc_novel' in d:
                w(table(d[d['head'] == 'eKNN-open'], 'auroc_novel', rows='param', cols='ds'), 'R5_table_auroc_novel.csv')
        if key == 'R6':
            d['total_s'] = d['time_encoder_s'].fillna(0) + d['time_head_s'].fillna(0)
            w(d.pivot_table(index='method', columns='n_cells', values='total_s'), 'R6_runtime_table.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    a = ap.parse_args()
    main(a.results if os.path.isabs(a.results) else os.path.join(HERE, a.results))
