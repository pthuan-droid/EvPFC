"""Figure 1: EvPFC pipeline (vector diagram, matplotlib)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = '#0b0b0b'; QUIET = '#52514e'; EDGE = '#8a8984'
BLUE = '#2a78d6'; BLUE_T = '#eaf2fc'; ORANGE = '#eb6834'; ORANGE_T = '#fdf0ea'; GREY_T = '#f4f3f0'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

W, H = 1000, 600
fig, ax = plt.subplots(figsize=(13.5, 8.1), dpi=250)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')


def box(x, y, w, h, title, lines=(), fc='white', ec=EDGE, lw=1.0, tsize=9.5, lsize=8.0, top=22, step=17):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=8', fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + top, title, ha='center', va='center', fontsize=tsize, fontweight='bold', color=INK)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + top + 22 + step * i, l, ha='center', va='center', fontsize=lsize, color=QUIET)


def arrow(p, q, c=EDGE, lw=1.2, ls='-', rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, color=c, lw=lw, linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


# ---- inputs
box(15, 25, 175, 95, 'Multi-omics profiles', ['RNA + ATAC / ADT counts', 'or processed features', 'some views may be missing'], fc=GREY_T)
box(15, 160, 175, 95, 'Partial annotation', ['labels for 1–10 % of cells', 'labels may be wrong', 'some types never labelled'], fc=GREY_T)

# ---- encoder and knowledge
box(230, 25, 210, 95, '(1) Label-free encoder', ['totalVI / MultiVI (count data)', 'or auto-encoder pre-training', r'$\rightarrow$ latent $z_i$, kept fixed'])
box(230, 160, 210, 95, '(2) Evidential k-NN', [r'masses $m_i(\{c\})$, $m_i(\Theta)$', r'pignistic prior $b_{ik}$', r'confidence $w_i = 1 - m_i(\Theta)$'],
    fc=ORANGE_T, ec=ORANGE, lw=1.5)
arrow((190, 72), (230, 72)); arrow((190, 207), (230, 207)); arrow((335, 120), (335, 160))

# ---- EvPFC core
box(480, 25, 300, 230, '(3) Picture fuzzy clustering', [], fc=BLUE_T, ec=BLUE, lw=1.8, tsize=9.3)
ax.text(630, 66, 'with an evidential prior', ha='center', va='center', fontsize=8.5, color=QUIET, style='italic')
core = [(r'$\mu_{ik} \propto \exp(-D_{ik}/\tau)\; b_{ik}^{\;\beta w_i}$', INK, 9.5),
        (r'$\eta_i$ : neutral mass on the nearest pair of clusters', QUIET, 8.0),
        (r'$\xi_i$ : refusal mass, reject cost $\lambda$', QUIET, 8.0),
        ('E-step: closed-form Gibbs solution', QUIET, 8.0),
        (r'M-step: closed-form centres $HC = R$', QUIET, 8.0),
        (r'$(\beta, \tau)$ selected on held-out labelled cells', QUIET, 8.0)]
for i, (t, c, s) in enumerate(core):
    ax.text(630, 94 + 25 * i, t, ha='center', va='center', fontsize=s, color=c)
arrow((440, 72), (480, 90)); arrow((440, 207), (480, 190), c=ORANGE, lw=1.6)
ax.text(630, 240, 'E- and M-steps alternate until the memberships settle', fontsize=7.6, color=BLUE, va='center', ha='center', style='italic')

# ---- outputs
box(840, 25, 145, 230, 'Outputs', ['cluster = arg max BetP', '', r'$\mu$  membership', r'$\eta$  boundary cell', r'$\xi$  refusal',
                                    r'$m(\Theta)$  novelty score'], fc=GREY_T, step=22)
arrow((780, 140), (840, 140))

# ---- open-world branch
box(230, 300, 755, 112, '(4) Optional open-world mode: split-conformal novelty threshold', [
    r'knowledge rebuilt from 80 % of the labels; $q$ = conformal $(1-\alpha)$ quantile of $m(\Theta)$ on the other 20 %',
    r'unlabelled cells with $m(\Theta) > q$ are declared novel and grouped into free clusters ($\beta = 0$)',
    r'guarantee: a cell of a known type is flagged with probability $\leq \alpha$'],
    fc='white', ec=ORANGE, lw=1.3, lsize=8.0, step=20)
arrow((335, 255), (335, 300), c=ORANGE, ls='--')

# ---- principles
ax.add_patch(FancyBboxPatch((15, 450), 970, 130, boxstyle='round,pad=0,rounding_size=8', fc='white', ec='#d9d8d3', lw=0.9))
ax.text(35, 472, 'Design principles', fontsize=9.2, fontweight='bold', color=INK, va='center')
notes = ['No network is trained on the labels, so wrong labels cannot be memorised by the representation.',
         r'Labels steer the partition only where the knowledge view has evidence ($w_i \approx 1$); where it abstains ($w_i \approx 0$),',
         'the picture fuzzy partition is driven by the data alone.',
         r'All updates are in closed form; one iteration costs $O(nKd)$.']
for i, l in enumerate(notes):
    ax.text(35 + (14 if i == 2 else 0), 498 + 20 * i, ('' if i == 2 else '•  ') + l, fontsize=8.2, color=QUIET, va='center')

plt.savefig('fig/fig1_evpfc.png', bbox_inches='tight', facecolor='white')
plt.savefig('fig/fig1_evpfc.pdf', bbox_inches='tight', facecolor='white')
print('ok')
