"""scPIKA: Picture-fuzzy Incomplete multi-view clustering with Knowledge Abstention.

Implements the model of the proposal:
  * per cell x view picture-fuzzy partition (singletons = mu, boundary pair = eta, Theta = xi)
    obtained in closed form (Gibbs) from an entropic energy with a rejection cost lambda;
  * missing views enter as vacuous masses (xi = 1), never imputed;
  * supervision (seed labels, optional marker evidence) is a knowledge view with abstention;
  * two-stage fusion: Dempster among data views, conflict-to-refusal with the knowledge view;
  * closed-form centre update (linear K x K system), networks trained with Adam.
"""
from dataclasses import dataclass, field, asdict
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, kmeans_plusplus

from .fusion import BBA, fuse_views, dempster, conflict_to_refusal
from .model import ViewAE
from .data import rows


@dataclass
class Config:
    d: int = 32
    hidden: tuple = (256, 128)
    pre_epochs: int = 30
    epochs: int = 40
    update_every: int = 4
    batch: int = 256
    lr: float = 1e-3
    alpha: float = 0.03      # free-energy (clustering) weight (chosen on the development sets)
    beta: float = 1.0        # cross-view self-training weight
    gamma: float = 1.0       # pairwise-constraint weight
    delta: float = 1.0       # seed-label weight
    rho: float = 1.0         # imprecision penalty of the boundary pair
    eps: float = 0.1         # label doubt: M0({t}) = 1 - eps, M0(Theta) = eps
    lam_quantile: float = 0.95
    tau_scale: float = 0.1   # tau = tau_scale * median_i min_k D_ik (chosen on the development sets)
    omega: float = 1.0       # quality shift of the rejection threshold
    stage2: str = 'refusal'  # 'refusal' (conflict -> Theta) or 'dempster'
    use_pair: bool = True
    use_refusal: bool = True
    global_reliability: bool = False
    missing: str = 'vacuous'  # 'vacuous' or 'zero' (naive zero-filling)
    kappa_weight: bool = True
    knowledge: str = 'eknn'  # 'eknn' (evidential k-NN from labelled cells) or 'self' (own label only)
    knn_labeled: int = 10
    pseudo: bool = False      # evidential self-training: confident cells become discounted evidence sources
    pseudo_conf: float = 0.9  # BetP max required
    pseudo_eps: float = 0.3   # doubt attached to pseudo-labels
    n_pair_batch: int = 128
    calibrate: bool = True    # calibrate the evidence of every source on the labelled cells
    align: bool = True        # Hungarian alignment of data clusters to labelled classes at each update
    decision: str = 'average' # final decision: 'average' | 'conjunctive' | 'fill' (knowledge, ignorance resolved by data)
                              # | 'cautious' (cautious rule within each omics, Dempster across omics)
    w_know: float = 0.75      # weight of the knowledge view in the average (chosen on the development sets)
    target: str = 'average'   # fusion used for the self-training target: 'dempster' (D (+) M0) or 'refusal'
    # ---- v2 options (defaults reproduce v1)
    know_open: bool = False   # distance-calibrated ignorance: neighbours beyond the class radius give little support
    open_q: float = 0.95      # quantile of leave-one-out same-class distances that defines the class radius
    label_discount: bool = False  # contextual discounting of each label by its leave-one-out agreement
    elp_k: int = 15           # graph degree of evidential label propagation (knowledge='elp')
    elp_alpha: float = 0.2    # propagation strength (same as the label-spreading baseline)
    discount_from: str = 'current'  # 'current' latent or 'pretrain' (unsupervised latent, frozen; not fitted to the labels)
    discount_apply: bool = True     # False: label reliability is only reported as a noisy-label score
    discount_filter: float = 0.0    # > 0: labels with reliability below this value are dropped, the rest kept intact
    per_omics: bool = False   # knowledge built separately in every omics space (for decision='cautious')
    free_init_theta: bool = False  # free clusters initialised from cells the knowledge view abstains on
    seed: int = 0
    threads: int = 1
    verbose: bool = False


class SCPIKA:
    def __init__(self, config: Config = None, **kw):
        self.cfg = config or Config()
        for k, v in kw.items():
            setattr(self.cfg, k, v)

    # ------------------------------------------------------------------ utilities
    def _log(self, *a):
        if self.cfg.verbose:
            print(*a, flush=True)

    def _batch_tensors(self, v, idx):
        x = torch.from_numpy(rows(v['x'], idx))
        t = torch.from_numpy(rows(v['target'], idx))
        return x, t

    def _encode_all(self):
        """Latent codes of all cells for every view (zeros where the view is missing)."""
        Z = []
        for ae, v in zip(self.aes, self.views):
            ae.eval()
            z = torch.zeros(self.n, self.cfg.d)
            idx = np.where(v['mask'])[0]
            with torch.no_grad():
                for s in range(0, idx.size, 2048):
                    b = idx[s:s + 2048]
                    z[b] = ae.encode(torch.from_numpy(rows(v['x'], b)))
            Z.append(z)
            ae.train()
        return Z

    # ------------------------------------------------------------------ per-view Gibbs masses
    def _gibbs(self, j, z, idx):
        """Picture-fuzzy BBA of view j for cells idx (z: latent of those cells)."""
        cfg, K = self.cfg, self.K
        C, tau = self.C[j], self.tau[j]
        D = torch.cdist(z, C).pow(2)                                   # n,K
        n = z.shape[0]
        logits = [-D / tau]
        Mk = torch.zeros(n, 1, K)
        if cfg.use_pair:
            near = D.topk(3, largest=False).indices                      # n,3
            cand = [(0, 1), (0, 2), (1, 2)]
            pcs = []
            for a, b in cand:
                ca, cb = C[near[:, a]], C[near[:, b]]
                pcs.append((2.0 ** cfg.rho) * (z - 0.5 * (ca + cb)).pow(2).sum(1))
            pcs = torch.stack(pcs, 1)                                    # n,3
            best = pcs.detach().argmin(1)
            pair_cost = pcs.gather(1, best[:, None]).squeeze(1)
            ai = torch.tensor([c[0] for c in cand])[best]
            bi = torch.tensor([c[1] for c in cand])[best]
            a_idx = near.gather(1, ai[:, None]).squeeze(1)
            b_idx = near.gather(1, bi[:, None]).squeeze(1)
            Mk[torch.arange(n), 0, a_idx] = 1.0
            Mk[torch.arange(n), 0, b_idx] = 1.0
            logits.append((-pair_cost / tau)[:, None])
        if cfg.use_refusal:
            lam_i = self.lam[j] - cfg.omega * tau * torch.from_numpy(self.views[j]['q'][idx])
            logits.append((-lam_i / tau)[:, None])
        L = torch.cat(logits, 1)
        m = torch.softmax(L / self.tmul[j], 1)                          # calibrated decision temperature
        phi = -tau * torch.logsumexp(L, 1)                              # free energy (training temperature)
        S = m[:, :K]
        P = m[:, K:K + 1] if cfg.use_pair else torch.zeros(n, 1)
        T = m[:, -1] if cfg.use_refusal else torch.zeros(n)
        return BBA(S, P, Mk, T), phi, D

    def _knowledge(self, idx):
        return BBA(self.S0[idx], torch.zeros(len(idx), 0), torch.zeros(len(idx), 0, self.K), self.T0[idx])

    def _view_bbas(self, Zb, idx, present):
        """Per-view BBAs for a batch; missing views are vacuous."""
        out, phis = [], []
        for j in range(self.V):
            b, phi, _ = self._gibbs(j, Zb[j], idx)
            vac = BBA.vacuous(len(idx), self.K, Q=1)
            b = b.where(present[j], vac)
            if self.cfg.global_reliability and self.xi_bar is not None:
                inf = 1.0 - b.T
                scale = (1.0 - self.xi_bar[j]) / inf.clamp(min=1e-6)
                b = BBA(b.S * scale[:, None], b.P * scale[:, None], b.Mk,
                        torch.full_like(b.T, float(self.xi_bar[j])))
                b = b.where(present[j], vac)
            out.append(b)
            phis.append(phi)
        return out, phis

    def _fuse(self, bbas, idx):
        D, kD = fuse_views(bbas, 'dempster') if self.V > 1 else (bbas[0], torch.zeros(len(idx)))
        K0 = self._knowledge(idx)
        if self.cfg.stage2 == 'refusal':
            M, kK = conflict_to_refusal(D, K0)
        else:
            M, kK = dempster(D, K0)
        return D, kD, M, kK

    # ------------------------------------------------------------------ knowledge view
    def _knn_labelled(self, Z, vids, lab, kk):
        """Squared distances (n, kk) to the kk nearest labelled cells, averaged over the views in vids
        that are observed in both cells (inf if none)."""
        n = self.n
        masks = [torch.from_numpy(self.views[j]['mask']).float() for j in vids]
        dists, nbrs = [], []
        for s0 in range(0, n, 4096):
            b = np.arange(s0, min(s0 + 4096, n))
            num = torch.zeros(len(b), lab.size); den = torch.zeros(len(b), lab.size)
            for mj, j in zip(masks, vids):
                both = mj[b][:, None] * mj[lab][None, :]
                num += torch.cdist(Z[j][b], Z[j][lab]).pow(2) * both
                den += both
            d2 = torch.where(den > 0, num / den.clamp(min=1), torch.full_like(num, float('inf')))
            dv, di = d2.topk(kk, largest=False)
            dists.append(dv); nbrs.append(di)
        return torch.cat(dists), torch.cat(nbrs)

    @staticmethod
    def _eknn_masses(dvv, dii, g, ylab, lab_eps, K, rad2=None):
        """Evidential k-NN (Denoeux, 1995). Neighbour j supports {y_j} with
        (1 - eps_j) exp(-g d^2) [x exp(-(d^2/r_{y_j}^2 - 1)_+) if class radii are given];
        supports of one class are pooled, classes are combined by Dempster's rule."""
        fin = torch.isfinite(dvv)
        d = torch.where(fin, dvv, torch.full_like(dvv, 1e6))
        sup = (1.0 - lab_eps[dii]) * torch.exp(-g * d)
        if rad2 is not None:
            r2 = rad2[ylab[dii]]
            sup = sup * torch.exp(-torch.relu(d / r2 - 1.0))
        sup = sup * fin
        lk = torch.zeros(dvv.shape[0], K)
        lk.scatter_add_(1, ylab[dii], torch.log1p(-sup.clamp(max=1 - 1e-6)))
        support = 1.0 - torch.exp(lk)
        l1 = torch.log1p(-support.clamp(max=1 - 1e-6))
        Sx = support * torch.exp(l1.sum(1, keepdim=True) - l1)
        Tx = torch.exp(l1.sum(1))
        z_ = Sx.sum(1) + Tx
        return Sx / z_[:, None], Tx / z_

    def _eknn_view(self, Z, vids, lab, ylab, lab_eps, calibrate_g=True):
        """Evidential k-NN knowledge computed in the latent space of the views vids.
        Returns S (n,K), T (n,), and the leave-one-out BetP of every labelled cell (lab.size, K)."""
        cfg, K, n = self.cfg, self.K, self.n
        k = min(cfg.knn_labeled, lab.size)
        kk = min(k + 1, lab.size)
        dv, di = self._knn_labelled(Z, vids, lab, kk)
        pos = np.full(n, -1); pos[lab] = np.arange(lab.size)
        dv_full, di_full = dv[:, :k], di[:, :k]
        not_self = di != torch.from_numpy(pos)[:, None]
        order = torch.argsort((~not_self).float(), dim=1, stable=True)       # push the self column to the end
        dv_loo, di_loo = dv.gather(1, order)[:, :k], di.gather(1, order)[:, :k]
        finite = torch.isfinite(dv_full)
        g0 = 1.0 / float(dv_full[finite].median().clamp(min=1e-6)) if finite.any() else 1.0
        rad2 = None
        if cfg.know_open:
            # class radius: quantile of the leave-one-out distance of labelled cells to their
            # nearest labelled neighbour of the same class
            same = (ylab[di_loo[lab]] == ylab[:, None]) & torch.isfinite(dv_loo[lab])
            dnn = torch.where(same, dv_loo[lab], torch.full_like(dv_loo[lab], float('inf'))).min(1).values
            ok = torch.isfinite(dnn)
            glob = float(torch.quantile(dnn[ok], cfg.open_q)) if ok.sum() > 2 else float(dv_full[finite].median())
            rad2 = torch.full((K,), glob)
            for c in range(K):
                sel = ok & (ylab == c)
                if sel.sum() >= 5:
                    rad2[c] = float(torch.quantile(dnn[sel], cfg.open_q))
            rad2 = rad2.clamp(min=1e-6)
            self.rad2 = rad2
        g = g0
        if calibrate_g and cfg.calibrate and lab.size > K and kk > k:
            best = None
            y_lab_t = ylab
            for f in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
                Sx, Tx = self._eknn_masses(dv_loo[lab], di_loo[lab], g0 * f, ylab, lab_eps, K, rad2)
                bp = Sx + Tx[:, None] / K
                nll = -torch.log(bp.gather(1, y_lab_t[:, None]).squeeze(1) + 1e-8).mean().item()
                if best is None or nll < best[0]:
                    best = (nll, f)
            g = g0 * best[1]
            self.g_mult = best[1]
        S, T = self._eknn_masses(dv_full, di_full, g, ylab, lab_eps, K, rad2)
        Sl, Tl = self._eknn_masses(dv_loo[lab], di_loo[lab], g, ylab, lab_eps, K, rad2)
        return S, T, Sl + Tl[:, None] / K

    # ---- evidential label propagation -------------------------------------------------
    def _graph(self, spaces, rows_mask):
        """Symmetric normalised union of kNN graphs built in every space (rows_mask: cells in the graph)."""
        import scipy.sparse as sp
        from sklearn.neighbors import NearestNeighbors
        idx = np.where(rows_mask)[0]
        m = idx.size
        k = min(self.cfg.elp_k, m - 1)
        R, C = [], []
        for X in spaces:
            Xi = np.asarray(X[idx], np.float32)
            nb = NearestNeighbors(n_neighbors=k + 1).fit(Xi).kneighbors(Xi, return_distance=False)
            R.append(np.repeat(np.arange(m), k)); C.append(nb[:, 1:].ravel())
        R, C = np.concatenate(R), np.concatenate(C)
        W = sp.csr_matrix((np.ones(R.size), (R, C)), shape=(m, m))
        W = ((W + W.T) > 0).astype(np.float64)
        d = np.asarray(W.sum(1)).ravel()
        Dm = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-12)))
        return idx, (Dm @ W @ Dm).tocsr()

    def _propagate(self, Sm, pos_lab, y_lab, eps_lab, gamma, iters=None):
        cfg, K = self.cfg, self.K
        m = Sm.shape[0]
        Y = np.zeros((m, K + 1)); Y[:, K] = gamma
        Y[pos_lab] = 0.0
        Y[pos_lab, y_lab] = 1.0 - eps_lab
        Y[pos_lab, K] = eps_lab
        a = cfg.elp_alpha
        iters = iters or (30 if a <= 0.5 else 200)
        F = Y.copy()
        for _ in range(iters):
            F = a * (Sm @ F) + (1 - a) * Y
        return F / F.sum(1, keepdims=True)

    def _elp_view(self, candidates, rows_mask, lab, ylab_np, eps_np):
        """Evidential label propagation: label spreading (Zhou et al., 2004) over the K singletons plus an
        ignorance channel Theta, which every unlabelled cell feeds with weight gamma. Mass that reaches a cell
        from labelled cells becomes belief in their classes; the rest stays on Theta.
        candidates: list of space sets (e.g. [[latent], [PCA], [latent, PCA]]). The graph (space set) and gamma
        are chosen together on held-out labelled cells (negative log pignistic likelihood)."""
        K, n = self.K, self.n
        S, T = torch.zeros(n, K), torch.ones(n)
        rng = np.random.RandomState(self.cfg.seed + 17)
        ho_all = rng.rand(lab.size) < 0.2
        best = None
        for ci, spaces in enumerate(candidates):
            idx, Sm = self._graph(spaces, rows_mask)
            pos = np.full(n, -1); pos[idx] = np.arange(idx.size)
            keep = pos[lab] >= 0
            labp, yl, el, ho = pos[lab][keep], ylab_np[keep], eps_np[keep], ho_all[keep]
            if labp.size == 0:
                return S, T
            if ho.sum() >= 5 and (~ho).sum() >= K:
                for gm in (1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
                    F = self._propagate(Sm, labp[~ho], yl[~ho], el[~ho], gm)
                    bp = F[labp[ho], :K] + F[labp[ho], K:K + 1] / K
                    nll = -np.log(bp[np.arange(ho.sum()), yl[ho]] + 1e-8).mean()
                    if best is None or nll < best[0]:
                        best = (nll, gm, ci, idx, Sm, labp, yl, el)
            elif best is None:
                best = (np.inf, 0.01, ci, idx, Sm, labp, yl, el)
        _, gm, ci, idx, Sm, labp, yl, el = best
        self.elp_gamma, self.elp_space = gm, ci
        F = self._propagate(Sm, labp, yl, el, gm)
        S[idx] = torch.from_numpy(F[:, :K]).float()
        T[idx] = torch.from_numpy(F[:, K]).float()
        return S, T

    def _pca_views(self):
        if getattr(self, '_pv', None) is None:
            from .baselines import joint_embedding
            self._pv = [joint_embedding([v], seed=self.cfg.seed) for v in self.views]
            self._pj = np.concatenate(self._pv, 1)
        return self._pv, self._pj

    def _build_knowledge(self, Z):
        """Knowledge view M0 with abstention.
        'self': labelled cells get M({t}) = 1 - eps, M(Theta) = eps; all other cells are vacuous.
        'eknn': evidential k-NN (Denoeux, 1995) on the k nearest labelled cells in the latent space.
        'elp' : evidential label propagation on a union kNN graph of the latent space and the PCA space.
        Optional (v2): class-radius ignorance (know_open), per-label discounting by leave-one-out
        agreement (label_discount), and one knowledge view per omics (per_omics).
        Marker evidence, if given, is combined with Dempster's rule."""
        cfg, K, n = self.cfg, self.K, self.n
        src_y = getattr(self, 'src_y', self.y_known)
        lab = np.where(src_y >= 0)[0]
        eps_np = np.full(lab.size, cfg.eps, np.float32)
        S0, T0 = torch.zeros(n, K), torch.ones(n)
        self.label_unrel = np.zeros(n, np.float32)
        self.lab_w = np.ones(n, np.float32)
        allv = list(range(self.V))
        if lab.size and cfg.knowledge == 'self':
            S0[lab, torch.from_numpy(self.y_known[lab])] = 1.0 - cfg.eps
            T0[lab] = cfg.eps
        elif lab.size:
            ylab = torch.from_numpy(src_y[lab])
            if cfg.label_discount:
                # contextual discounting: each label is a source whose reliability is its leave-one-out
                # agreement with the other labelled cells (two passes). With discount_from='pretrain' the
                # reliability is computed once in the latent space of the unsupervised pre-training, which
                # has not been fitted to the (possibly wrong) labels, and then kept fixed.
                if cfg.discount_from == 'pretrain' and getattr(self, '_frozen_r', None) is not None:
                    r = self._frozen_r
                else:
                    e_tmp = eps_np.copy()
                    for _ in range(2):
                        _, _, bl = self._eknn_view(Z, allv, lab, ylab, torch.from_numpy(e_tmp))
                        p_own = bl.gather(1, ylab[:, None]).squeeze(1)
                        r = (p_own / bl.max(1).values.clamp(min=1e-8)).clamp(0, 1).numpy()
                        e_tmp = (1.0 - (1.0 - cfg.eps) * r).astype(np.float32)
                    if cfg.discount_from == 'pretrain':
                        self._frozen_r = r
                if cfg.discount_apply:
                    r_use = (r >= cfg.discount_filter).astype(np.float32) if cfg.discount_filter > 0 else r
                    eps_np = (1.0 - (1.0 - cfg.eps) * r_use).astype(np.float32)
                    self.lab_w[lab] = r_use
                self.label_unrel[lab] = 1.0 - r
            lab_eps = torch.from_numpy(eps_np)
            if cfg.knowledge == 'elp':
                pv, pj = self._pca_views()
                Zc = torch.cat(Z, 1).numpy()
                S0, T0 = self._elp_view([[Zc], [pj], [Zc, pj]], np.ones(n, bool), lab, src_y[lab], eps_np)
            else:
                S0, T0, _ = self._eknn_view(Z, allv, lab, ylab, lab_eps)
            if cfg.per_omics:
                self.S0v, self.T0v = [], []
                for j in allv:
                    if self.V == 1:
                        Sj, Tj = S0, T0
                    elif cfg.knowledge == 'elp':
                        pv, _ = self._pca_views()
                        zj = Z[j].numpy()
                        Sj, Tj = self._elp_view([[zj], [pv[j]], [zj, pv[j]]], self.views[j]['mask'], lab, src_y[lab], eps_np)
                    else:
                        Sj, Tj, _ = self._eknn_view(Z, [j], lab, ylab, lab_eps)
                    self.S0v.append(Sj); self.T0v.append(Tj)
        K0 = BBA(S0, torch.zeros(n, 0), torch.zeros(n, 0, K), T0)
        if self.marker is not None:
            K0, _ = dempster(K0, self.marker)
        self.S0, self.T0 = K0.S, K0.T

    # ------------------------------------------------------------------ centres
    def _update_centres(self, Z):
        cfg, K = self.cfg, self.K
        for j, v in enumerate(self.views):
            idx = np.where(v['mask'])[0]
            z = Z[j][idx]
            with torch.no_grad():
                b, _, _ = self._gibbs(j, z, idx)
            S, w = b.S, b.P[:, 0]
            H = torch.diag(S.sum(0))
            R = S.t() @ z
            if cfg.use_pair:
                ab = b.Mk[:, 0, :]                                      # n,K indicator of pair
                H = H + (2.0 ** (cfg.rho - 2)) * (ab * w[:, None]).t() @ ab
                R = R + (2.0 ** (cfg.rho - 1)) * (ab * w[:, None]).t() @ z
            mass = S.sum(0)
            H = H + 1e-6 * torch.eye(K)
            Cn = torch.linalg.solve(H, R)
            keep_old = mass < 1e-3
            Cn[keep_old] = self.C[j][keep_old]
            self.C[j] = Cn

    def _align(self, betp_data):
        """Permute cluster indices so that the data partition agrees with the labelled classes.
        Data clusters are matched to classes 0..T-1 by the Hungarian algorithm on the labelled cells;
        unmatched clusters keep their relative order in T..K-1. Returns True if a permutation was applied."""
        from scipy.optimize import linear_sum_assignment
        lab = np.where(self.y_known >= 0)[0]
        if lab.size == 0 or self.T == 0:
            return False
        pred = betp_data.argmax(1).numpy()[lab]
        W = np.zeros((self.K, self.T))
        np.add.at(W, (pred, self.y_known[lab]), 1.0)
        r, c = linear_sum_assignment(-W)
        perm = [None] * self.K
        for k_old, t in zip(r, c):
            perm[t] = k_old
        rest = [k for k in range(self.K) if k not in set(r)]
        for i in range(self.T, self.K):
            perm[i] = rest[i - self.T]
        perm = np.array(perm)
        if np.all(perm == np.arange(self.K)):
            return False
        self.C = [Cj[torch.from_numpy(perm)] for Cj in self.C]
        return True

    def _calibrate(self, Z, grid=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)):
        """Decision temperature of every data view, chosen on the labelled cells by minimising the
        negative log pignistic probability of their labels (temperature scaling). An unreliable view is
        softened, so that its evidence weighs less in the fusion. Returns True if anything changed."""
        lab = np.where(self.y_known >= 0)[0]
        if lab.size == 0 or self.T == 0:
            return False
        old = list(self.tmul)
        for j, v in enumerate(self.views):
            idx = lab[v['mask'][lab]]
            if idx.size < 2:
                continue
            yt = torch.from_numpy(self.y_known[idx])
            best = None
            with torch.no_grad():
                for f in grid:
                    self.tmul[j] = f
                    b, _, _ = self._gibbs(j, Z[j][idx], idx)
                    nll = -torch.log(b.betp().gather(1, yt[:, None]).squeeze(1) + 1e-8).mean().item()
                    if best is None or nll < best[0]:
                        best = (nll, f)
            self.tmul[j] = best[1]
        return self.tmul != old

    # ------------------------------------------------------------------ full pass
    @staticmethod
    def _odds(S, P, Mk, T):
        """Singleton odds r_k = m'({k}) / m(Theta) of a BBA after splitting pair masses equally between
        their two members. A BBA on singletons and Theta with m(Theta) > 0 is separable: it is the
        Dempster combination of simple supports on {k} with weights w_k = 1 / (1 + r_k)."""
        a = S + (0.5 * (P.unsqueeze(-1) * Mk).sum(1) if P.shape[1] else 0.0)
        return a.double() / T.double().clamp(min=1e-6)[:, None]

    def _cautious_omics(self, bb, idx):
        """Cautious rule (Denoeux, 2008) between the data and knowledge evidence of the same omics
        (non-distinct sources, both live in that omics' space), Dempster's rule across omics.
        On separable BBAs: cautious = min of weights = max of odds; Dempster = product of weights,
        i.e. 1 + r = prod_j (1 + r_j). Returns BetP, m(Theta) and the Dempster conflict across omics."""
        logs = 0.0
        for j, b in enumerate(bb):
            rD = self._odds(b.S, b.P, b.Mk, b.T)
            if getattr(self, 'S0v', None) is not None:
                rK = self.S0v[j][idx].double() / self.T0v[j][idx].double().clamp(min=1e-6)[:, None]
            else:
                rK = self.S0[idx].double() / self.T0[idx].double().clamp(min=1e-6)[:, None]
            r = torch.maximum(rD, rK)
            logs = logs + torch.log1p(r)
        R = torch.expm1(logs)                         # combined odds
        z = 1.0 + R.sum(1, keepdim=True)
        m = R / z; t = (1.0 / z).squeeze(1)
        # conflict of the unnormalised combination across omics: 1 - prod_k w_k (1 + sum_k R_k)
        logw = -logs.sum(1)
        conflict = (1.0 - torch.exp(logw + torch.log(z.squeeze(1)))).clamp(min=0.0)
        K = m.shape[1]
        return (m + t[:, None] / K).float(), t.float(), conflict.float()

    def _full_pass(self, Z):
        out = dict(S=[], betp=[], xi=[], eta=[], kD=[], kK=[], view_xi=[], betp_views=[], betp_dem=[], betp_avg=[],
                   betp_fill=[], betp_caut=[], xi_caut=[], k_caut=[], know_theta=[])
        with torch.no_grad():
            for s in range(0, self.n, 4096):
                idx = np.arange(s, min(s + 4096, self.n))
                Zb = [z[idx] for z in Z]
                pres = [torch.from_numpy(v['mask'][idx]) for v in self.views]
                bb, _ = self._view_bbas(Zb, idx, pres)
                D, kD, M, kK = self._fuse(bb, idx)
                bK = self._knowledge(idx).betp()
                out['betp_avg'].append(self.cfg.w_know * bK + (1 - self.cfg.w_know) * D.betp())
                # knowledge decides where it has evidence; its ignorance is resolved by the data partition
                out['betp_fill'].append(self.S0[idx] + self.T0[idx][:, None] * D.betp())
                out['know_theta'].append(self.T0[idx])
                bc, tc, kc = self._cautious_omics(bb, idx)
                out['betp_caut'].append(bc); out['xi_caut'].append(tc); out['k_caut'].append(kc)
                out['betp'].append(M.betp()); out['xi'].append(M.T); out['eta'].append(M.eta())
                out['kD'].append(kD); out['kK'].append(kK)
                out['view_xi'].append(torch.stack([b.T for b in bb], 1))
                out['betp_views'].append(torch.stack([b.betp() for b in bb], 1))
                out['S'].append(D.betp())
                out['betp_dem'].append(dempster(D, self._knowledge(idx))[0].betp())
        return {k: torch.cat(v, 0) for k, v in out.items()}

    @staticmethod
    def _target(betp):
        f = betp.sum(0, keepdim=True)
        p = betp ** 2 / f
        return p / p.sum(1, keepdim=True)

    # ------------------------------------------------------------------ fit
    def fit(self, views, y_known=None, n_known=0, K=None, constraints=None, marker_bba=None):
        """
        views      : list of dicts from scpika.data.build
        y_known    : int array (n,), label in 0..n_known-1 for labelled cells, -1 otherwise
        n_known    : number of known cell types T (clusters 0..T-1 are tied to them)
        K          : total number of clusters (K >= T); clusters T..K-1 are free
        constraints: optional (i, j, is_must_link, weight) arrays
        marker_bba : optional (S_marker (n,T), T_marker (n,)) marker evidence
        """
        cfg = self.cfg
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        torch.set_num_threads(cfg.threads)
        t0 = time.time()
        self.views = [dict(v) for v in views]
        self._pv = None
        self._frozen_r = None
        if cfg.missing == 'zero':
            for v in self.views:
                v['mask'] = np.ones_like(v['mask'])
        self.V, self.n, self.K, self.T = len(views), views[0]['mask'].shape[0], K, n_known
        y_known = np.full(self.n, -1) if y_known is None else np.asarray(y_known)
        self.y_known = y_known
        lab = np.where(y_known >= 0)[0]

        # ---- marker evidence (fixed); label evidence is built from the latent space below
        self.marker = None
        if marker_bba is not None:
            Sm = torch.zeros(self.n, K); Sm[:, :n_known] = torch.as_tensor(marker_bba[0], dtype=torch.float32)
            self.marker = BBA(Sm, torch.zeros(self.n, 0), torch.zeros(self.n, 0, K),
                              torch.as_tensor(marker_bba[1], dtype=torch.float32))
        self.S0, self.T0 = torch.zeros(self.n, K), torch.ones(self.n)

        # ---- autoencoders
        self.aes = [ViewAE(v['x'].shape[1], v['lik'], cfg.d, cfg.hidden) for v in self.views]
        params = [p for ae in self.aes for p in ae.parameters()]
        opt = torch.optim.Adam(params, lr=cfg.lr)
        for ep in range(cfg.pre_epochs):
            tot = 0.0
            for idx in self._batches():
                loss = 0.0
                for ae, v in zip(self.aes, self.views):
                    b = idx[v['mask'][idx]]
                    if b.size < 2:
                        continue
                    x, t = self._batch_tensors(v, b)
                    loss = loss + ae.recon_loss(ae.encode(x), t, None)
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            self._log(f'pretrain {ep} loss {tot:.3f}')

        # ---- initial centres: seeded k-means on concatenated latents
        Z = self._encode_all()
        self._build_knowledge(Z)
        Zc = torch.cat(Z, 1).numpy()
        init = []
        for t in range(n_known):
            m = lab[y_known[lab] == t]
            init.append(Zc[m].mean(0) if m.size else None)
        rng = np.random.RandomState(cfg.seed)
        n_free = K - sum(c is not None for c in init)
        pool = Zc
        if cfg.free_init_theta and n_free > 0 and (y_known >= 0).any():
            th = self.T0.numpy()
            cand = np.where(th >= np.quantile(th, 0.8))[0]
            if cand.size >= 5 * n_free:
                pool = Zc[cand]                      # free clusters start where the knowledge view abstains
        extra = kmeans_plusplus(pool, n_free, random_state=rng)[0] if n_free > 0 else []
        it = iter(extra)
        init = [c if c is not None else next(it) for c in init] + [next(it) for _ in range(K - n_known)]
        km = KMeans(K, init=np.stack(init), n_init=1, max_iter=100, random_state=cfg.seed).fit(Zc)
        assign = km.labels_
        self.C = []
        for j, v in enumerate(self.views):
            Cj = torch.zeros(K, cfg.d)
            for k in range(K):
                m = np.where((assign == k) & v['mask'])[0]
                Cj[k] = Z[j][m].mean(0) if m.size else Z[j][np.random.choice(np.where(v['mask'])[0])]
            self.C.append(Cj)

        # ---- calibrate tau and lambda per view (data-driven)
        self.tau, self.lam = [], []
        for j, v in enumerate(self.views):
            idx = np.where(v['mask'])[0]
            D = torch.cdist(Z[j][idx], self.C[j]).pow(2)
            tau = cfg.tau_scale * float(D.min(1).values.median().clamp(min=1e-3))
            self.tau.append(tau)
            phit = -tau * torch.logsumexp(-D / tau, 1)
            self.lam.append(float(torch.quantile(phit, cfg.lam_quantile)))
        self.xi_bar = None
        self.tmul = [1.0] * self.V

        # ---- constraints
        if constraints is not None:
            ci, cj, ml, cw = [np.asarray(a) for a in constraints]
        # ---- main loop
        for ep in range(cfg.epochs):
            if ep % cfg.update_every == 0:
                Z = self._encode_all()
                self._build_knowledge(Z)
                self._update_centres(Z)
                full = self._full_pass(Z)
                if cfg.align and self._align(full['S']):
                    full = self._full_pass(Z)
                if cfg.calibrate and self._calibrate(Z):
                    full = self._full_pass(Z)
                P = self._target({'dempster': full['betp_dem'], 'average': full['betp_avg'], 'fill': full['betp_fill'],
                                  'cautious': full['betp_caut']}.get(cfg.target, full['betp']))
                if cfg.pseudo and cfg.knowledge == 'eknn' and (self.y_known >= 0).any():
                    bp = full['betp']
                    conf = (bp.max(1).values >= cfg.pseudo_conf) & (full['xi'] < 0.1) & (full['kK'] < 0.1)
                    conf = conf.numpy() & (self.y_known < 0)
                    src = self.y_known.copy(); eps_arr = np.full(self.n, cfg.eps, np.float32)
                    pl = bp.argmax(1).numpy()
                    keep = conf & (pl < self.T)          # only known classes are propagated
                    src[keep] = pl[keep]; eps_arr[keep] = cfg.pseudo_eps
                    self.src_y, self.src_eps = src, eps_arr
                self.xi_bar = full['view_xi'].mean(0).numpy() if cfg.global_reliability else None
                pred = full['betp'].argmax(1).numpy()
                if ep > 0:
                    changed = (pred != self._last_pred).mean()
                    self._log(f'epoch {ep}: changed {changed:.4f}, mean xi {float(full["xi"].mean()):.3f}')
                    if changed < 1e-3:
                        break
                self._last_pred = pred
            for idx in self._batches():
                if constraints is not None and len(ci):
                    sel = np.random.randint(0, len(ci), cfg.n_pair_batch)
                    idx = np.concatenate([idx, ci[sel], cj[sel]])
                loss = self._step_loss(idx, P, constraints is not None and len(ci) > 0,
                                       (sel if constraints is not None and len(ci) else None),
                                       (ml, cw) if constraints is not None else None)
                opt.zero_grad(); loss.backward(); opt.step()

        Z = self._encode_all()
        self._update_centres(Z)
        self.result_ = self._full_pass(Z)
        self.result_['Z'] = Z
        self.fit_time_ = time.time() - t0
        return self

    def _batches(self):
        perm = np.random.permutation(self.n)
        return [perm[s:s + self.cfg.batch] for s in range(0, self.n, self.cfg.batch)]

    def _step_loss(self, idx, P, use_pairs, sel, cons):
        cfg = self.cfg
        nb = len(idx)
        Zb, pres, rec = [], [], 0.0
        for j, (ae, v) in enumerate(zip(self.aes, self.views)):
            m = v['mask'][idx]
            z = torch.zeros(nb, cfg.d)
            b = idx[m]
            if b.size >= 2:
                x, t = self._batch_tensors(v, b)
                zz = ae.encode(x)
                rec = rec + ae.recon_loss(zz, t, None)
                z = z.index_put((torch.from_numpy(np.where(m)[0]),), zz)
            else:
                m = np.zeros_like(m)
            Zb.append(z); pres.append(torch.from_numpy(m))
        bbas, phis = self._view_bbas(Zb, idx, pres)
        D, kD, M, kK = self._fuse(bbas, idx)
        Pt = P[idx]
        loss = rec
        for j, b in enumerate(bbas):
            pm = pres[j].float()
            if pm.sum() == 0:
                continue
            w = (1.0 - b.T).detach() * pm
            loss = loss + cfg.alpha * (phis[j] * pm).sum() / pm.sum()
            kl = (Pt * (torch.log(Pt + 1e-8) - torch.log(b.betp() + 1e-8))).sum(1)
            loss = loss + cfg.beta * (w * kl).sum() / pm.sum()
            yk = self.y_known[idx]
            labm = torch.from_numpy(yk >= 0) & pres[j]
            if labm.any() and cfg.delta > 0:
                bp = b.betp()[labm]
                yt = torch.from_numpy(yk[labm.numpy()])
                wl = w[labm]
                if cfg.kappa_weight:
                    wl = wl * (1.0 - kK[labm]).detach()
                if cfg.label_discount:
                    wl = wl * torch.from_numpy(self.lab_w[idx][labm.numpy()])
                ce = -torch.log(bp.gather(1, yt[:, None]).squeeze(1) + 1e-8)
                loss = loss + cfg.delta * (wl * ce).sum() / labm.sum()
        if use_pairs:
            ml, cw = cons
            q = len(sel)
            bp = D.betp()
            bi, bj = bp[-2 * q:-q], bp[-q:]
            s = (bi * bj).sum(1).clamp(1e-6, 1 - 1e-6)
            wt = torch.from_numpy(cw[sel].astype(np.float32)) * ((1 - D.T[-2 * q:-q]) * (1 - D.T[-q:])).detach()
            isml = torch.from_numpy(ml[sel].astype(bool))
            lp = torch.where(isml, -torch.log(s), -torch.log(1 - s))
            loss = loss + cfg.gamma * (wt * lp).mean()
        return loss

    # ------------------------------------------------------------------ outputs
    def predict(self):
        key = {'average': 'betp_avg', 'fill': 'betp_fill', 'cautious': 'betp_caut'}.get(self.cfg.decision, 'betp')
        return self.result_[key].argmax(1).numpy()

    def uncertainty(self):
        r = self.result_
        return dict(xi=r['xi'].numpy(), eta=r['eta'].numpy(), kappa_D=r['kD'].numpy(),
                    kappa_K=r['kK'].numpy(), view_xi=r['view_xi'].numpy(),
                    maxp=r['betp'].max(1).values.numpy(),
                    decision_proba=r[{'average': 'betp_avg', 'fill': 'betp_fill', 'cautious': 'betp_caut'}.get(self.cfg.decision, 'betp')].numpy(),
                    xi_caut=r['xi_caut'].numpy(), kappa_caut=r['k_caut'].numpy(),
                    know_theta=r['know_theta'].numpy(), label_unrel=self.label_unrel)


def marker_evidence(scores, coverage, kappa_m=0.5, temperature=1.0):
    """Marker evidence with abstention (Sec. 4.3): M({t}) = kappa_m * r_i * softmax(s_it/T),
    M(Theta) = 1 - kappa_m * r_i. scores: (n,T) standardised marker scores; coverage r_i in [0,1]."""
    s = np.asarray(scores, np.float32) / temperature
    p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    r = np.clip(np.asarray(coverage, np.float32), 0, 1)
    S = kappa_m * r[:, None] * p
    return S, 1.0 - kappa_m * r
