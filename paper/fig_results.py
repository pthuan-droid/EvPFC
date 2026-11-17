"""Figures 3-6 of the paper from results_tidy.csv and R8b_conformal.csv (development seeds 3-7 unless stated).

    fig3() -> Fig. 2 (label fraction)   fig4() -> Fig. 3 (label noise)
    fig6() -> Fig. 4 (open-world mode)  fig7() -> Fig. 9 (missing views)
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collect import ORDER, DSNAME, COUNT

from paths import TIDY, RAW
T = pd.read_csv(TIDY)
DEV = T[T.phase == 'dev']
COL = {'EvPFC': '#2a78d6', 'eKNN': '#eb6834', 'LS': '#1baf7a', 'kNN': '#eda100', 'E2E': '#e87ba4',
       'scANVI': '#008300', 'EvPFC w/o prior': '#6250d6', 'Seeded k-means': '#e34948', 'E2E (zero-fill)': '#e34948',
       'EvPFC-conformal': '#6250d6', 'SSFCM': '#8a8984'}
MK = {'EvPFC': 'o', 'eKNN': 's', 'LS': '^', 'kNN': 'D', 'E2E': 'v', 'scANVI': 'P', 'EvPFC w/o prior': 'X',
      'Seeded k-means': '*', 'E2E (zero-fill)': 'x', 'EvPFC-conformal': 'X', 'SSFCM': 'h'}
INK, QUIET, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': '#b9b8b3', 'axes.linewidth': 0.7,
                     'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'axes.labelcolor': INK, 'xtick.color': QUIET,
                     'ytick.color': QUIET, 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f'fig/{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'fig/{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig3():
    meths = ['EvPFC', 'eKNN', 'LS', 'kNN', 'SSFCM', 'E2E', 'scANVI']
    fr = {'frac1': 1, 'frac5': 5, 'frac10': 10}
    fig, axs = plt.subplots(2, 5, figsize=(12.5, 5.0), sharex=True)
    for ax, ds in zip(axs.ravel(), ORDER):
        d = DEV[(DEV.ds == ds) & DEV.setting.isin(fr)]
        for i, m in enumerate(meths):
            g = d[d.method == m].groupby('setting').ARI.agg(['mean', 'std'])
            if not len(g):
                continue
            x = np.array([fr[s] for s in g.index]); o = np.argsort(x)
            xs = x[o] * (1 + (i - 3) * 0.022)
            ax.errorbar(xs, g['mean'].values[o], yerr=g['std'].values[o], color=COL[m], marker=MK[m], ms=4.2,
                        lw=1.6 if m == 'EvPFC' else 1.0, elinewidth=0.7, capsize=0, label=m,
                        zorder=3 if m == 'EvPFC' else 2, mec='white', mew=0.5)
        ax.set_xscale('log'); ax.set_xticks([1, 5, 10]); ax.set_xticklabels(['1', '5', '10'])
        ax.minorticks_off(); grid(ax)
        ax.set_title(DSNAME[ds] + ('' if ds in COUNT else '  (AE latent)'), loc='left')
    for ax in axs[1]:
        ax.set_xlabel('labelled cells (%)')
    for ax in axs[:, 0]:
        ax.set_ylabel('ARI')
    h, l = [], []
    for ax in axs.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc='lower center', ncol=len(l), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'fig3_labelfrac')


def fig4():
    meths = ['EvPFC', 'eKNN', 'LS', 'kNN', 'SSFCM', 'E2E', 'scANVI']
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    ys = np.arange(len(ORDER))[::-1]
    for i, m in enumerate(meths):
        g = DEV[(DEV.setting == 'noise20') & (DEV.method == m)].groupby('ds').ARI.mean()
        yy = [ys[ORDER.index(ds)] + (i - 3) * 0.11 for ds in g.index if ds in ORDER]
        ax.scatter(g.values, yy, color=COL[m], marker=MK[m], s=26, label=m, zorder=3, edgecolors='white', linewidths=0.5)
    ax.set_yticks(ys); ax.set_yticklabels([DSNAME[d] for d in ORDER]); grid(ax, 'x')
    ax.set_xlabel('ARI with 20 % wrong labels (5 % labelled)'); ax.set_title('(a) ARI under label noise', loc='left')
    ax.legend(loc='lower left', fontsize=7, ncol=1)
    ax = axs[1]
    # ARI lost to noise: clean 5 % minus 20 % noise, paired by data set and seed
    rows = []
    for m in meths:
        a = DEV[(DEV.setting == 'frac5') & (DEV.method == m)].set_index(['ds', 'seed']).ARI
        b = DEV[(DEV.setting == 'noise20') & (DEV.method == m)].set_index(['ds', 'seed']).ARI
        j = (a - b).dropna()
        for ds, v in j.groupby(level=0):
            rows.append((m, ds, v.mean()))
    L = pd.DataFrame(rows, columns=['m', 'ds', 'loss'])
    for i, m in enumerate(meths):
        v = L[L.m == m]
        if not len(v):
            continue
        x = np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v))
        ax.scatter(x, v.loss, color=COL[m], marker=MK[m], s=16, alpha=0.75, edgecolors='white', linewidths=0.4)
        ax.plot([i - 0.28, i + 0.28], [v.loss.mean()] * 2, color=INK, lw=1.4)
    ax.axhline(0, color='#b9b8b3', lw=0.7)
    ax.set_xticks(range(len(meths))); ax.set_xticklabels(meths); grid(ax)
    ax.set_ylabel('ARI(clean) − ARI(20 % noise)'); ax.set_title('(b) ARI lost to wrong labels', loc='left')
    ax.text(0.02, 0.97, 'dots: data sets; bar: mean', transform=ax.transAxes, ha='left', va='top', fontsize=7, color=QUIET)
    fig.tight_layout()
    save(fig, 'fig4_noise')
    return L.groupby('m').loss.agg(['mean', 'count'])


def clip(g, hi=None):
    lo = np.minimum(g['std'].values, g['mean'].values)
    up = g['std'].values if hi is None else np.minimum(g['std'].values, hi - g['mean'].values)
    return np.vstack([lo, up])


def fig6():
    d = pd.read_csv(os.path.join(RAW, 'R8b_conformal.csv'))
    from collect import MAIN
    d = d[d.encoder == d.ds.map(MAIN)].copy()
    ex = d.extra.map(json.loads)
    for k in ['hidden_to_free', 'known_to_free', 'auroc_novel']:
        d[k] = ex.map(lambda z: z.get(k, np.nan))
    d['method'] = d['head']
    dss = [x for x in ORDER if x in set(d.ds)]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))
    meths = ['EvPFC', 'EvPFC-conformal']
    w = 0.36
    x = np.arange(len(dss))
    ax = axs[0]
    for i, m in enumerate(meths):
        g = d[(d.hide == 0) & (d.method == m)].groupby('ds').known_to_free.agg(['mean', 'std']).reindex(dss)
        ax.bar(x + (i - 0.5) * w, g['mean'], w * 0.92, yerr=clip(g), color=COL[m], label=m,
               error_kw=dict(lw=0.7, ecolor=QUIET))
    ax.axhline(0.05, color=INK, lw=0.9, ls='--'); ax.text(len(dss) - 0.5, 0.053, r'$\alpha$ = 0.05', ha='right', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([DSNAME[s] for s in dss], rotation=30, ha='right'); grid(ax)
    ax.set_ylabel('share of known-type cells'); ax.set_title('(a) known-type cells sent to free clusters\n(no hidden type, 2 free clusters)', loc='left')
    ax.legend(fontsize=7, loc='upper left')
    ax = axs[1]
    for i, m in enumerate(meths):
        g = d[(d.hide == 2) & (d.method == m)].groupby('ds').hidden_to_free.agg(['mean', 'std']).reindex(dss)
        ax.bar(x + (i - 0.5) * w, g['mean'], w * 0.92, yerr=clip(g), color=COL[m], label=m, error_kw=dict(lw=0.7, ecolor=QUIET))
    ax.axhline(0.30, color=INK, lw=0.9, ls=':'); ax.text(-0.4, 0.31, 'pre-registered target 30 %', fontsize=7, va='bottom')
    ax.set_xticks(x); ax.set_xticklabels([DSNAME[s] for s in dss], rotation=30, ha='right'); grid(ax)
    ax.set_ylabel('share of hidden-type cells'); ax.set_title('(b) hidden-type cells found\n(2 types hidden)', loc='left')
    ax = axs[2]
    meths3 = ['eKNN', 'EvPFC', 'EvPFC-conformal']
    w3 = 0.27
    for i, m in enumerate(meths3):
        g = d[(d.hide == 2) & (d.method == m)].groupby('ds').auroc_novel.agg(['mean', 'std']).reindex(dss)
        ax.bar(x + (i - 1) * w3, g['mean'], w3 * 0.92, yerr=clip(g, 1.0), color=COL[m], label=m, error_kw=dict(lw=0.7, ecolor=QUIET))
    ax.axhline(0.5, color='#b9b8b3', lw=0.7)
    ax.set_ylim(0.4, 1.0); ax.set_xticks(x); ax.set_xticklabels([DSNAME[s] for s in dss], rotation=30, ha='right'); grid(ax)
    ax.set_ylabel('AUROC'); ax.set_title('(c) novelty score: hidden vs known cells', loc='left'); ax.legend(fontsize=7, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.27))
    fig.tight_layout()
    save(fig, 'fig5_openworld')
    s = d.groupby(['hide', 'method'])[['known_to_free', 'hidden_to_free', 'auroc_novel', 'ARI']].mean()
    return s


def fig7():
    meths = ['EvPFC', 'eKNN', 'LS', 'kNN', 'E2E', 'E2E (zero-fill)']
    d = DEV[DEV.ds == 'CellMix']
    lv = {'frac5': 0, 'miss30': 30, 'miss50': 50, 'miss70': 70}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in meths:
        g = d[(d.method == m) & d.setting.isin(lv)].groupby('setting').ARI.agg(['mean', 'std'])
        if not len(g):
            continue
        xs = np.array([lv[s] for s in g.index]); o = np.argsort(xs)
        ax.errorbar(xs[o], g['mean'].values[o], yerr=g['std'].values[o], color=COL[m], marker=MK[m], ms=4.5,
                    lw=1.6 if m == 'EvPFC' else 1.0, elinewidth=0.7, label=m, mec='white', mew=0.5,
                    ls='--' if m == 'E2E (zero-fill)' else '-')
    ax.set_xticks([0, 30, 50, 70]); ax.set_xlabel('cells with one view removed (%)'); ax.set_ylabel('ARI')
    grid(ax); ax.legend(fontsize=7, ncol=2, loc='lower left')
    ax.set_title('CellMix, 5 % labels', loc='left')
    fig.tight_layout()
    save(fig, 'fig6_missing')


if __name__ == '__main__':
    which = sys.argv[1:] or ['3', '4', '6', '7']   # fig6() draws Fig. 4 (open world), fig7() draws Fig. 9 (missing views)
    for w in which:
        print('fig', w, globals()['fig' + w]())
