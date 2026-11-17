"""Figure 2: what the evidential prior does, on a 2-D toy problem (computed with the actual EvPFC code)."""
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(_R, 'experiments')); sys.path.insert(0, _R)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evpfc import EvPFC, eknn_knowledge
from sklearn.semi_supervised import LabelSpreading

rng = np.random.RandomState(7)
centres = np.array([[0, 0], [3.2, 0.3], [1.5, 2.9], [4.6, 3.1], [-0.6, 3.6]])   # last one: unseen type
sizes = [220, 220, 220, 220, 160]
X = np.vstack([c + rng.randn(n, 2) * s for c, n, s in zip(centres, sizes, [0.55, 0.55, 0.6, 0.5, 0.45])])
y = np.repeat(np.arange(5), sizes)
T, K = 4, 5
lab = (rng.rand(len(y)) < 0.04) & (y < T)
yk = np.full(len(y), -1); yk[lab] = y[lab]
S, Th = eknn_knowledge(X, yk, T, seed=0)
m = EvPFC(K, T, beta=2.0, tau_scale=0.3, seed=0).fit(X, yk, S, Th)
pred = m.predict()
ls = LabelSpreading(kernel='knn', n_neighbors=15, alpha=0.2).fit(X, yk).transduction_

COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6']
MK = ['o', 's', '^', 'D', 'P']
GREY = '#b9b8b3'; INK = '#0b0b0b'; QUIET = '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': '#c9c8c3',
                     'axes.linewidth': 0.8, 'axes.titlesize': 9, 'axes.titleweight': 'bold'})
fig, axs = plt.subplots(1, 4, figsize=(11, 3.0), dpi=300)


def clean(ax, t):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, loc='left', color=INK)
    ax.set_aspect('equal')


ax = axs[0]
for k in range(5):
    sel = y == k
    ax.scatter(X[sel, 0], X[sel, 1], s=5, c=GREY if k == 4 else COL[k], alpha=0.35 if k < 4 else 0.6, lw=0, marker=MK[k])
for k in range(T):
    sel = lab & (y == k)
    ax.scatter(X[sel, 0], X[sel, 1], s=34, c=COL[k], marker=MK[k], edgecolors='white', linewidths=1.0, zorder=3)
ax.text(centres[4, 0], centres[4, 1] + 1.25, 'unseen type\n(no label)', ha='center', fontsize=7, color=QUIET)
clean(ax, '(a) data, 4 % labelled')

ax = axs[1]
sc = ax.scatter(X[:, 0], X[:, 1], s=5, c=Th, cmap='Blues', vmin=0, vmax=1, lw=0)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
clean(ax, r'(b) ignorance $m_i(\Theta)$')

ax = axs[2]
for k in range(K):
    sel = ls == k
    if sel.any():
        ax.scatter(X[sel, 0], X[sel, 1], s=5, c=COL[k], marker=MK[k], lw=0, alpha=0.8)
clean(ax, '(c) label spreading')

ax = axs[3]
for k in range(K):
    sel = pred == k
    ax.scatter(X[sel, 0], X[sel, 1], s=5, c=COL[k] if k < T else COL[4], marker=MK[k], lw=0, alpha=0.8)
hi = m.eta > np.quantile(m.eta, 0.95)
ax.scatter(X[hi, 0], X[hi, 1], s=14, facecolors='none', edgecolors=INK, linewidths=0.5)
clean(ax, r'(d) EvPFC ($\circ$: top 5 % $\eta$)')
for a in axs:
    a.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5); a.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 1.5)
plt.tight_layout()
plt.savefig('fig/fig2_toy.png', bbox_inches='tight', facecolor='white')
from sklearn.metrics import adjusted_rand_score as ari
print('ARI LS', round(ari(y, ls), 3), 'ARI EvPFC', round(ari(y, pred), 3), 'novel->free', round(np.mean(pred[y == 4] == 4), 3),
      'mean Theta novel', round(Th[y == 4].mean(), 3), 'known', round(Th[y < 4].mean(), 3))
