"""Figures 7-10 of the paper (ablation and encoder, per-cell analysis, confirmation, run time).

Inputs (all produced by the scripts in scpika/round2):
  R10_ablation_ae.csv (r7_evpfc.py with the ablation heads), R8a_evpfc_ae.csv, R7_evpfc.csv, R8b_conformal.csv,
  R11_bins.csv + R11_cells_<ds>.npz (r11_cells.py), R12_converge.csv, R13_runtime.csv, results_tidy.csv
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fig_results import COL, MK, INK, QUIET, GRID, grid, save
from collect import DSNAME

from paths import RAW, BMMC, TIDY as TIDY_PATH, CELLS
RES = os.environ.get('RES', RAW)
R7 = os.path.join(RAW, 'R7_evpfc.csv')
R8A = os.path.join(RAW, 'R8a_evpfc_ae.csv')
R8B = os.path.join(RAW, 'R8b_conformal.csv')
PROC = ['BMNC', 'Ma2020_1', 'PBMC10k_Multiome_feat', 'SMAGE3K']
KQ = os.environ.get('KQ', BMMC)
R17E, R17A, R17B = (os.path.join(KQ, f) for f in ['R17_evpfc.csv', 'R17_ablation.csv', 'R17_cells_bins.csv'])
ORDER8 = ['SLN111', 'SLN206', 'CellMix', 'ASAP_PBMC', 'BMMC_CITE', 'BMMC_Multiome', 'BMNC', 'Ma2020_1', 'PBMC10k_Multiome_feat', 'SMAGE3K']
MAIN = {'SLN111': 'totalvi', 'SLN206': 'totalvi', 'CellMix': 'multivi', 'ASAP_PBMC': 'multivi',
        'BMMC_CITE': 'totalvi', 'BMMC_Multiome': 'multivi', 'BMNC': 'ae', 'Ma2020_1': 'ae', 'PBMC10k_Multiome_feat': 'ae', 'SMAGE3K': 'ae'}
TIDY = pd.read_csv(TIDY_PATH)


def setting(r):
    if r.hide > 0:
        return 'hide2'
    if r.noise > 0:
        return f'noise{int(round(r.noise * 100))}'
    return f'frac{int(round(r.frac * 100))}'


def load(path):
    d = pd.read_csv(path)
    d['setting'] = d.apply(setting, axis=1)
    return d


# ----------------------------------------------------------------------------------------------- Fig. 5
ABL = [('EvPFC-constw', '$w_i\\!=\\!1$'), ('EvPFC-nopair', 'no $\\eta$'),
       ('EvPFC-norefusal', 'no $\\xi$'), ('EvPFC-plain', 'no $\\eta,\\xi$'),
       ('EvPFC-noprior', '$\\beta\\!=\\!0$')]
COUNT_DS = ['SLN111', 'SLN206', 'CellMix', 'ASAP_PBMC', 'BMMC_CITE', 'BMMC_Multiome']
SETS = [('frac1', '1 % labels'), ('frac5', '5 % labels'), ('noise20', '5 %, 20 % wrong'), ('hide2', '2 types hidden')]


R10C = os.path.join(RES, 'R10_ablation_count.csv')


def ablation_table():
    """Variants from R10 (processed: autoencoder latent; count data: totalVI / MultiVI latent); full EvPFC and
    EvPFC without prior from R7 (count data) and R8a (processed), i.e. the same runs as in Tables 2-3."""
    parts = [load(os.path.join(RES, 'R10_ablation_ae.csv')), load(R10C), load(R7), load(R8A), load(R17A), load(R17E)]
    d = pd.concat(parts)
    d = d[(d.encoder == d.ds.map(MAIN)) & (d.seed <= 7) & d['head'].str.startswith('EvPFC')]
    piv = d.pivot_table(index=['setting', 'ds', 'seed'], columns='head', values='ARI')
    rows = []
    for s, _ in SETS:
        x = piv.loc[s]
        for v, _ in ABL:
            if v not in x:
                continue
            dl = (x[v] - x['EvPFC']).dropna()
            per = dl.groupby(level=0).mean()
            from scipy.stats import wilcoxon
            p = wilcoxon(dl).pvalue if (dl != 0).sum() > 5 else np.nan
            rows.append(dict(setting=s, variant=v, mean=dl.mean(), n_pairs=len(dl), worse_pairs=int((dl < 0).sum()),
                             better_pairs=int((dl > 0).sum()), ds_worse=int((per < 0).sum()), n_ds=len(per), p=p,
                             **{f'd_{k}': per.get(k, np.nan) for k in MAIN}, evpfc=x['EvPFC'].mean()))
    T = pd.DataFrame(rows)
    ok = T.p.notna() & (T.variant != 'EvPFC-noprior')
    from stats2 import holm
    T['p_holm'] = np.nan
    T.loc[ok, 'p_holm'] = holm(T.loc[ok, 'p'])
    T.to_csv('tables/ablation.csv', index=False)
    return piv, T


def fig7():
    piv, T = ablation_table()
    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=0.62, wspace=0.28)
    for j, (s, title) in enumerate(SETS):
        ax = fig.add_subplot(gs[0, j])
        x = piv.loc[s]
        for i, (v, _) in enumerate(ABL):
            if v not in x:
                continue
            per = (x[v] - x['EvPFC']).groupby(level=0).mean()
            xs = i + np.linspace(-0.2, 0.2, len(per))
            ax.scatter(xs, per.values, s=13, color=['#52514e' if q in COUNT_DS else '#b9b8b3' for q in per.index],
                       edgecolors='white', linewidths=0.4, zorder=2)
            ax.plot([i - 0.3, i + 0.3], [per.mean()] * 2, color=COL['EvPFC'], lw=2.0, zorder=3)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xticks(range(len(ABL))); ax.set_xticklabels([l for _, l in ABL], fontsize=7.5)
        grid(ax); ax.set_title(f'({"abcd"[j]}) {title}', loc='left')
        if j == 0:
            ax.set_ylabel('ARI(variant) − ARI(EvPFC)')
            ax.text(0.03, 0.03, 'dark: count data (totalVI/MultiVI)\nlight: processed data (AE)\nbar: mean over data sets',
                    transform=ax.transAxes, fontsize=6.2, color=QUIET, va='bottom')
    # encoder panels
    r7 = load(R7); r8 = load(R8A)
    e2e = TIDY[(TIDY.method == 'E2E') & (TIDY.phase == 'dev')]
    for j, s in enumerate(['frac1', 'frac5']):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        series = [('EvPFC on PCA', r7[(r7['head'] == 'EvPFC') & (r7.encoder == 'pca') & (r7.setting == s)], '#8a8984'),
                  ('EvPFC on AE latent', r8[(r8['head'] == 'EvPFC') & (r8.setting == s)], COL['EvPFC']),
                  ('E2E (encoder fitted to labels)', e2e[e2e.setting == s], COL['E2E'])]
        w = 0.26
        for i, (name, d, c) in enumerate(series):
            g = d[d.ds.isin(PROC)].groupby('ds').ARI.agg(['mean', 'std']).reindex(PROC)
            xs = np.arange(len(PROC)) + (i - 1) * w
            ax.bar(xs, g['mean'], w * 0.92, color=c, label=name, zorder=2)
            ax.errorbar(xs, g['mean'], yerr=g['std'], fmt='none', ecolor=INK, elinewidth=0.6, zorder=3)
        ax.set_xticks(range(len(PROC))); ax.set_xticklabels([DSNAME[d] for d in PROC])
        ax.set_ylim(0.0, 1.0); grid(ax); ax.set_ylabel('ARI')
        ax.set_title(f'({"ef"[j]}) choice of latent space, {"1" if s == "frac1" else "5"} % labels', loc='left')
        if j == 0:
            h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.12)
    save(fig, 'fig7_ablation')
    return T


# ----------------------------------------------------------------------------------------------- Fig. 7
def fig8(ds='Ma2020_1'):
    from sklearn.manifold import TSNE
    z = np.load(os.path.join(CELLS, f'R11_cells_{ds}_s3.npz'))
    cache = os.path.join(CELLS, f'tsne_{ds}.npy')
    if os.path.exists(cache):
        E = np.load(cache)
    else:
        E = TSNE(n_components=2, init='pca', perplexity=30, random_state=0).fit_transform(z['Z'])
        np.save(cache, E)
    ev, y = z['ev'], z['y']
    ok_e, ok_k = z['pred_evpfc'] == y, z['pred_eknn'] == y
    fig = plt.figure(figsize=(11.5, 8.8))
    G = fig.add_gridspec(2, 1, height_ratios=[1, 1.62], hspace=0.28)
    gs = G[0].subgridspec(1, 4, wspace=0.36)
    gb = G[1].subgridspec(2, 5, hspace=0.55, wspace=0.34)
    kw = dict(s=2.2, linewidths=0, rasterized=True)

    def emb(ax, title):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, loc='left')
        for s in ax.spines.values():
            s.set_visible(False)
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(E[:, 0], E[:, 1], c=z['theta'], cmap='Blues', vmin=0, vmax=np.quantile(z['theta'], 0.99), **kw)
    lab = z['yk'] >= 0
    ax.scatter(E[lab, 0], E[lab, 1], s=3.5, color=INK, linewidths=0, label='labelled cell')
    emb(ax, '(a) ignorance $m_i(\\Theta)$'); fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
    ax.legend(fontsize=6.4, loc='upper left', bbox_to_anchor=(0, 0), markerscale=2, handletextpad=0.1, frameon=False)
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(E[:, 0], E[:, 1], color='#e6e5e1', **kw)
    a = ev & ok_e & ~ok_k; b = ev & ~ok_e & ok_k
    ax.scatter(E[a, 0], E[a, 1], s=3.2, color=COL['EvPFC'], linewidths=0, label=f'EvPFC right, eKNN wrong ({a.sum()})')
    ax.scatter(E[b, 0], E[b, 1], s=3.2, color=COL['eKNN'], linewidths=0, label=f'eKNN right, EvPFC wrong ({b.sum()})')
    emb(ax, '(b) where the two disagree'); ax.legend(fontsize=6.4, loc='upper left', bbox_to_anchor=(0, 0), markerscale=2.5, handletextpad=0.1, frameon=False)
    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(E[:, 0], E[:, 1], c=z['xi'], cmap='Oranges', vmin=0, vmax=np.quantile(z['xi'], 0.99), **kw)
    emb(ax, '(c) refusal mass $\\xi_i$'); fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
    B = pd.concat([pd.read_csv(os.path.join(RES, 'R11_bins.csv')), pd.read_csv(os.path.join(RES, 'R11_bins_count.csv')),
                   pd.read_csv(R17B).drop(columns=['enc'])])
    ax = fig.add_subplot(gs[0, 3])
    x = np.arange(1, 6)
    for col, c, ls, nm in [('xi_mean', COL['EvPFC'], '-', 'refusal $\\xi_i$'), ('eta_mean', '#8a8984', '--', 'neutral $\\eta_i$')]:
        g = B.groupby(['ds', 'bin'])[col].mean().unstack('ds')
        ax.fill_between(x, g.min(1), g.max(1), color=c, alpha=0.15, lw=0)
        ax.plot(x, g.mean(1), color=c, ls=ls, marker='o', ms=3.5, lw=1.4, label=nm)
    ax.set_xticks(x); ax.set_xticklabels(['low', '2', '3', '4', 'high']); grid(ax)
    ax.set_xlabel('quintile of $m_i(\\Theta)$')
    ax.set_title(f'(d) mean mass (band: range over {B.ds.nunique()} data sets)', loc='left'); ax.legend(fontsize=6.8, loc='upper left')
    letters = 'efghijklmn'
    for j, d in enumerate(ORDER8):
        ax = fig.add_subplot(gb[j // 5, j % 5])
        g = B[B.ds == d].groupby('bin')[['acc_evpfc', 'acc_eknn']].agg(['mean', 'std'])
        for col, m in [('acc_eknn', 'eKNN'), ('acc_evpfc', 'EvPFC')]:
            ax.errorbar(x, g[(col, 'mean')], yerr=g[(col, 'std')], color=COL[m], marker=MK[m], ms=4, lw=1.5 if m == 'EvPFC' else 1.0,
                        elinewidth=0.6, label=m, mec='white', mew=0.5)
        ax.set_xticks(x); ax.set_xticklabels(['low', '2', '3', '4', 'high'])
        grid(ax); ax.set_title(f'({letters[j]}) {DSNAME[d]}', loc='left')
        if j >= 5:
            ax.set_xlabel('quintile of $m_i(\\Theta)$')
        if j % 5 == 0:
            ax.set_ylabel('accuracy, unlabelled cells')
        if j == 0:
            ax.legend(fontsize=7, loc='lower left')
    save(fig, 'fig8_cells')
    out = B.groupby(['ds', 'bin'])[['acc_evpfc', 'acc_eknn', 'theta_mean', 'eta_mean', 'xi_mean']].mean()
    out['gain'] = out.acc_evpfc - out.acc_eknn
    out.to_csv('tables/r11_bins_summary.csv')
    return out


# ----------------------------------------------------------------------------------------------- Fig. 10
def confirmation_pairs():
    """Gain of EvPFC over eKNN and LS per (setting, data set) on development and on confirmation draws.
    Development: results_tidy (main latent of each data set). Confirmation: results_tidy (R9: count data sets) and,
    for 5 % labels and two hidden types on the six data sets with >= 6 types, R8b (main latent)."""
    rows = []
    r8b = load(R8B)
    r8b['setting'] = np.where(r8b.hide > 0, 'hide2', 'frac5')
    r8b = r8b[r8b.encoder == r8b.ds.map(MAIN)]
    six = set(r8b.ds)
    for s in ['frac1', 'frac5', 'frac10', 'noise20', 'hide2']:
        for ph in ['dev', 'confirm']:
            if ph == 'confirm' and s in ('frac5', 'hide2'):
                d = r8b[r8b.setting == s].rename(columns={'head': 'method'})
                d = pd.concat([d, TIDY[(TIDY.phase == ph) & (TIDY.setting == s) & ~TIDY.ds.isin(six)]])
            else:
                d = TIDY[(TIDY.phase == ph) & (TIDY.setting == s)]
            x = d.pivot_table(index='ds', columns='method', values='ARI')
            for ds in x.index:
                for o in ['eKNN', 'LS']:
                    if {'EvPFC', o} <= set(x.columns) and not np.isnan(x.loc[ds, 'EvPFC'] - x.loc[ds, o]):
                        rows.append(dict(setting=s, ds=ds, other=o, phase=ph, gain=x.loc[ds, 'EvPFC'] - x.loc[ds, o]))
    P = pd.DataFrame(rows).pivot_table(index=['setting', 'ds', 'other'], columns='phase', values='gain').dropna()
    P.to_csv('tables/confirmation_gains.csv')
    return P


def fig9():
    P = confirmation_pairs().reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2))
    mk = {'frac1': 'o', 'frac5': 's', 'frac10': 'D', 'noise20': '^', 'hide2': 'v'}
    nm = {'frac1': '1 % labels', 'frac5': '5 % labels', 'frac10': '10 % labels', 'noise20': '20 % wrong labels',
          'hide2': '2 types hidden'}
    for ax, o in zip(axs, ['eKNN', 'LS']):
        d = P[P.other == o]
        lim = [min(d.dev.min(), d.confirm.min()) - 0.02, max(d.dev.max(), d.confirm.max()) + 0.02]
        ax.plot(lim, lim, color=QUIET, lw=0.7, ls='--', zorder=1)
        ax.axhline(0, color=INK, lw=0.6); ax.axvline(0, color=INK, lw=0.6)
        for s in mk:
            e = d[d.setting == s]
            ax.scatter(e.dev, e.confirm, marker=mk[s], s=30, color=COL['EvPFC'], edgecolors='white', linewidths=0.5,
                       label=nm[s], zorder=3)
        ax.set_xlim(lim); ax.set_ylim(lim); grid(ax, 'both')
        ax.set_xlabel(f'ARI(EvPFC) − ARI({o}), development draws')
        ax.set_ylabel(f'ARI(EvPFC) − ARI({o}), confirmation draws')
        pos = (d.confirm > 0).sum()
        ax.set_title(f'({"ab"[o == "LS"]}) EvPFC vs {o}: {pos}/{len(d)} gains positive on new draws', loc='left')
    axs[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    save(fig, 'fig9_confirmation')
    return P


# ----------------------------------------------------------------------------------------------- Fig. 10
def fig10():
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    ax = axs[0]
    r = pd.read_csv(os.path.join(RES, 'R13_runtime.csv'))
    for m in ['EvPFC', 'eKNN', 'LS']:
        g = r[r.method == m].groupby('n').time_s.agg(['median', 'min', 'max'])
        ax.errorbar(g.index, g['median'], yerr=[g['median'] - g['min'], g['max'] - g['median']], marker=MK[m], color=COL[m],
                    lw=1.4 if m == 'EvPFC' else 1.0, ms=4.5, label=m + (' (incl. selection)' if m == 'EvPFC' else ''),
                    mec='white', mew=0.5, elinewidth=0.6)
    ax.set_xscale('log'); ax.set_yscale('log'); grid(ax, 'both'); ax.minorticks_off()
    ax.set_xticks([2000, 5000, 10000, 15802]); ax.set_xticklabels(['2k', '5k', '10k', '15.8k'])
    ax.set_xlabel('number of cells (SLN111 subsamples)'); ax.set_ylabel('time (s), 2 CPU threads')
    ax.set_title('(a) run time of the clustering step', loc='left'); ax.legend(fontsize=6.8, loc='upper left')
    ax = axs[1]
    c = pd.read_csv(os.path.join(RES, 'R12_converge.csv'))
    capped = c[c.it60 >= 60]
    ax.scatter(c.ari60, c.ari500, s=22, color=[COL['EvPFC'] if v >= 60 else '#8a8984' for v in c.it60],
               edgecolors='white', linewidths=0.4, zorder=3)
    lo, hi = min(c.ari60.min(), c.ari500.min()) - 0.02, max(c.ari60.max(), c.ari500.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color=QUIET, lw=0.7, ls='--'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); grid(ax, 'both')
    ax.set_xlabel('ARI, cap of 60 iterations'); ax.set_ylabel('ARI, cap of 500 iterations')
    ax.set_title(f'(b) iteration cap ({len(capped)}/{len(c)} fits capped, blue)', loc='left')
    ax = axs[2]
    e = TIDY[(TIDY.method == 'EvPFC') & TIDY.beta.notna()]
    tab = pd.crosstab(e.tau_scale, e.beta).reindex(index=[0.1, 0.3, 1.0], columns=[0.5, 1.0, 2.0, 4.0]).fillna(0)
    ax.imshow(tab.values, cmap='Blues', aspect='auto', origin='lower')
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = int(tab.values[i, j])
            ax.text(j, i, v, ha='center', va='center', fontsize=8, color='white' if v > tab.values.max() * 0.55 else INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(['0.5', '1', '2', '4']); ax.set_yticks(range(3)); ax.set_yticklabels(['0.1', '0.3', '1'])
    ax.set_xlabel(r'$\beta$'); ax.set_ylabel(r'$\tau_s$'); ax.set_title(f'(c) selected parameters ({len(e)} fits)', loc='left')
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    save(fig, 'fig10_runtime')
    return r.groupby(['method', 'n']).time_s.median(), c, tab


if __name__ == '__main__':
    for w in sys.argv[1:]:
        print('fig', w, globals()['fig' + w]())
