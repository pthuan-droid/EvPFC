"""EvPFC: picture fuzzy clustering with an evidential prior (semi-supervised, on a fixed latent space).

For every cell i and cluster k (clusters 0..T-1 tied to the T labelled types, T..K-1 free):
    mass on {k}      ∝ exp(-D_ik / tau) * b_ik ** (beta * w_i)                    -> positive degree  mu_ik
    mass on {a, b}   ∝ exp(-2^rho ||z_i - (c_a + c_b)/2||^2 / tau) * bab ** (beta * w_i)  -> neutral degree eta_i
    mass on Theta    ∝ exp(-lambda / tau)                                          -> refusal degree   xi_i
normalised to sum to one (closed-form Gibbs solution of an entropy-regularised picture fuzzy energy).
    b_ik : pignistic probability of the evidential k-NN knowledge view (Denoeux 1995) over the K clusters
           (its ignorance m_i(Theta) is spread uniformly, so free clusters receive b_ik = m_i(Theta)/K)
    w_i  : 1 - m_i(Theta), the knowledge view's own confidence: labels steer the partition only where
           the knowledge view has evidence; where it abstains the partition is driven by the data alone.
Centres: closed-form weighted means (H C = R, as in scPIKA). No neural network is trained, so wrong labels
cannot be memorised by the representation.
tau (relative to the median nearest-centre distance) and beta are chosen on held-out labelled cells.
"""
import numpy as np
import torch
from sklearn.cluster import kmeans_plusplus


def _sqdist(Z, C):
    return np.maximum((Z * Z).sum(1)[:, None] - 2 * Z @ C.T + (C * C).sum(1)[None, :], 0.0)


def eknn_knowledge(Z, yk, T, seed=0, **cfg):
    """Evidential k-NN masses (singletons over T classes, Theta) from scPIKA's knowledge view."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scpika.core import SCPIKA, Config
    n = Z.shape[0]
    m = SCPIKA(Config(seed=seed, **cfg))
    m.views = [dict(mask=np.ones(n, bool))]
    m.n, m.K, m.V, m.y_known = n, T, 1, yk
    lab = np.where(yk >= 0)[0]
    S, Th, _ = m._eknn_view([torch.from_numpy(np.ascontiguousarray(Z, np.float32))], [0], lab,
                            torch.from_numpy(yk[lab]), torch.full((lab.size,), float(m.cfg.eps)))
    return S.numpy().astype(np.float64), Th.numpy().astype(np.float64)


class EvPFC:
    def __init__(self, K, T, beta=1.0, tau_scale=0.3, rho=1.0, lam_quantile=0.95, use_pair=True,
                 use_refusal=True, max_iter=60, tol=1e-4, seed=0, theta_init_q=0.8, const_w=False):
        self.K, self.T, self.beta, self.tau_scale, self.rho = K, T, beta, tau_scale, rho
        self.lam_quantile, self.use_pair, self.use_refusal = lam_quantile, use_pair, use_refusal
        self.max_iter, self.tol, self.seed, self.theta_init_q = max_iter, tol, seed, theta_init_q
        self.const_w = const_w   # ablation: the same prior strength for every cell (w_i = 1)

    # ------------------------------------------------------------------ prior
    def _prior(self, S, Th):
        K, T = self.K, self.T
        B = np.zeros((S.shape[0], K))
        B[:, :T] = S
        B += Th[:, None] / K
        w = np.ones_like(Th) if self.const_w else 1.0 - Th
        return np.clip(B, 1e-12, None), w

    # ------------------------------------------------------------------ init
    def _init(self, Z, yk, Th):
        rng = np.random.RandomState(self.seed)
        C = np.zeros((self.K, Z.shape[1]))
        missing = []
        for k in range(self.T):
            m = yk == k
            if m.any():
                C[k] = Z[m].mean(0)
            else:
                missing.append(k)
        n_free = self.K - self.T + len(missing)
        if n_free:
            cand = np.where(Th >= np.quantile(Th, self.theta_init_q))[0]
            pool = Z[cand] if cand.size >= 5 * n_free else Z
            extra = kmeans_plusplus(pool, n_free, random_state=rng)[0]
            for k, c in zip(missing + list(range(self.T, self.K)), extra):
                C[k] = c
        return C

    # ------------------------------------------------------------------ E-step (closed form)
    def _masses(self, Z, C, logB, w):
        n, K = Z.shape[0], self.K
        D = _sqdist(Z, C)
        bw = (self.beta * w)[:, None]
        logits = [-D / self.tau + bw * logB]
        pair = None
        if self.use_pair and K >= 2:
            near = np.argsort(D, 1)[:, :min(3, K)]
            cands = [(0, 1), (0, 2), (1, 2)] if K >= 3 else [(0, 1)]
            costs, priors = [], []
            for a, b in cands:
                ca, cb = C[near[:, a]], C[near[:, b]]
                costs.append((2.0 ** self.rho) * ((Z - 0.5 * (ca + cb)) ** 2).sum(1))
                pa = np.take_along_axis(logB, near[:, [a]], 1)[:, 0]
                pb = np.take_along_axis(logB, near[:, [b]], 1)[:, 0]
                priors.append(np.logaddexp(pa, pb) - np.log(2.0))
            costs, priors = np.stack(costs, 1), np.stack(priors, 1)
            best = costs.argmin(1)
            pc = costs[np.arange(n), best]; pp = priors[np.arange(n), best]
            ai = np.array([c[0] for c in cands])[best]; bi = np.array([c[1] for c in cands])[best]
            pair = (near[np.arange(n), ai], near[np.arange(n), bi])
            logits.append((-pc / self.tau + bw[:, 0] * pp)[:, None])
        if self.use_refusal:
            logits.append(np.full((n, 1), -self.lam / self.tau))
        L = np.concatenate(logits, 1)
        L -= L.max(1, keepdims=True)
        M = np.exp(L); M /= M.sum(1, keepdims=True)
        mu = M[:, :K]
        eta = M[:, K] if pair is not None else np.zeros(n)
        xi = M[:, -1] if self.use_refusal else np.zeros(n)
        return mu, eta, xi, pair, D

    # ------------------------------------------------------------------ M-step (closed form)
    def _centres(self, Z, mu, eta, pair, C_old):
        K = self.K
        H = np.diag(mu.sum(0))
        R = mu.T @ Z
        if pair is not None:
            A = np.zeros_like(mu)
            A[np.arange(len(Z)), pair[0]] = 1.0; A[np.arange(len(Z)), pair[1]] = 1.0
            Aw = A * eta[:, None]
            H = H + (2.0 ** (self.rho - 2)) * Aw.T @ A
            R = R + (2.0 ** (self.rho - 1)) * Aw.T @ Z
        H = H + 1e-8 * np.eye(K)
        C = np.linalg.solve(H, R)
        dead = mu.sum(0) < 1e-6
        C[dead] = C_old[dead]
        return C

    def fit(self, Z, yk, S, Th):
        Z = np.asarray(Z, np.float64)
        B, w = self._prior(S, Th)
        logB = np.log(B)
        C = self._init(Z, yk, Th)
        D = _sqdist(Z, C)
        self.tau = self.tau_scale * float(np.median(D.min(1)) + 1e-12)
        fe = -self.tau * np.log(np.exp(-(D - D.min(1, keepdims=True)) / self.tau).sum(1)) + D.min(1)
        self.lam = float(np.quantile(fe, self.lam_quantile))
        prev = None
        for it in range(self.max_iter):
            mu, eta, xi, pair, D = self._masses(Z, C, logB, w)
            C = self._centres(Z, mu, eta, pair, C)
            if prev is not None and np.abs(mu - prev).max() < self.tol:
                break
            prev = mu
        mu, eta, xi, pair, D = self._masses(Z, C, logB, w)
        self.C, self.mu, self.eta, self.xi, self.pair, self.n_iter = C, mu, eta, xi, pair, it + 1
        return self

    def betp(self):
        P = self.mu.copy()
        if self.pair is not None:
            idx = np.arange(len(P))
            P[idx, self.pair[0]] += self.eta / 2; P[idx, self.pair[1]] += self.eta / 2
        P += self.xi[:, None] / self.K
        return P

    def predict(self):
        return self.betp().argmax(1)


def select_and_fit(Z, yk, T, K, seed=0, betas=(0.5, 1, 2, 4), taus=(0.1, 0.3, 1.0), holdout=0.2, **kw):
    """Choose (beta, tau_scale) on held-out labelled cells (accuracy, ties broken by NLL), then refit on all labels."""
    rng = np.random.RandomState(seed + 101)
    lab = np.where(yk >= 0)[0]
    ho = lab[rng.rand(lab.size) < holdout]
    best = None
    if ho.size >= 5 and (lab.size - ho.size) >= T:
        yk_tr = yk.copy(); yk_tr[ho] = -1
        S, Th = eknn_knowledge(Z, yk_tr, T, seed)
        for b in betas:
            for t in taus:
                m = EvPFC(K, T, beta=b, tau_scale=t, seed=seed, **kw).fit(Z, yk_tr, S, Th)
                P = m.betp()[ho]
                acc = np.mean(P.argmax(1) == yk[ho])
                nll = -np.mean(np.log(P[np.arange(ho.size), yk[ho]] + 1e-12))
                score = (acc, -nll)
                if best is None or score > best[0]:
                    best = (score, b, t)
    b, t = (best[1], best[2]) if best else (1.0, 0.3)
    S, Th = eknn_knowledge(Z, yk, T, seed)
    m = EvPFC(K, T, beta=b, tau_scale=t, seed=seed, **kw).fit(Z, yk, S, Th)
    m.knowledge = (S, Th); m.selected = dict(beta=b, tau_scale=t)
    return m


def conformal_open_set_fit(Z, yk, T, K, seed=0, alpha=0.05, cal_frac=0.2, **kw):
    """EvPFC with split-conformal detection of unseen types.
    1. 20 % of the labelled cells form a calibration set; the evidential k-NN knowledge is built from the rest.
    2. q = conformal (1 - alpha) quantile of the knowledge ignorance m(Theta) on the calibration cells.
       Labelled cells are a random sample, so calibration cells are exchangeable with the unlabelled cells of
       known types: a known-type cell exceeds q with probability <= alpha.
    3. Unlabelled cells with m(Theta) > q are declared novel and partitioned into the K - T free clusters by
       picture fuzzy clustering without prior (beta = 0); all other cells are assigned by EvPFC over the T
       known clusters (knowledge from all labels).
    Returns pred (n,), novelty score (n,), dict of diagnostics."""
    rng = np.random.RandomState(seed + 303)
    lab = np.where(yk >= 0)[0]
    cal = lab[rng.rand(lab.size) < cal_frac]
    yk_tr = yk.copy(); yk_tr[cal] = -1
    _, th_tr = eknn_knowledge(Z, yk_tr, T, seed)
    s = np.sort(th_tr[cal])
    k = int(np.ceil((cal.size + 1) * (1 - alpha))) - 1
    q = s[min(k, cal.size - 1)] if cal.size else 1.0
    novel = (th_tr > q) & (yk < 0)
    m = select_and_fit(Z, yk, T, T, seed, **kw)            # known clusters only
    pred = m.predict()
    n_free = K - T
    info = dict(q=float(q), n_novel=int(novel.sum()), n_cal=int(cal.size))
    if n_free > 0 and novel.sum() >= 5 * n_free:
        Zn = Z[novel]
        f = EvPFC(n_free, 0, beta=0.0, tau_scale=m.selected['tau_scale'], seed=seed)
        f.fit(Zn, np.full(Zn.shape[0], -1), np.zeros((Zn.shape[0], 0)), np.ones(Zn.shape[0]))
        pred[novel] = T + f.predict()
    return pred, th_tr, info
