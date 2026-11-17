"""Semi-supervised fuzzy c-means with partial supervision (Pedrycz & Waletzky, IEEE TSMC-B 27(5), 1997).

Objective (fuzzifier m = 2):
    J = sum_ik u_ik^2 d_ik^2 + alpha * sum_ik (u_ik - f_ik b_i)^2 d_ik^2,
with b_i = 1 for labelled cells, f_ik the one-hot label (zero on the free clusters).
Closed-form updates (Pedrycz & Waletzky 1997, Eqs. for m = 2):
    u_ik = 1/(1+alpha) * [ (1 + alpha (1 - b_i sum_j f_ij)) / sum_j (d_ik^2 / d_ij^2) + alpha f_ik b_i ]
    v_k  = sum_i phi_ik z_i / sum_i phi_ik,   phi_ik = u_ik^2 + alpha (u_ik - f_ik b_i)^2
alpha is chosen on held-out labelled cells exactly as for EvPFC (20 % held out, best accuracy).
"""
import numpy as np


def _sqd(Z, C):
    return np.maximum((Z ** 2).sum(1)[:, None] - 2 * Z @ C.T + (C ** 2).sum(1)[None, :], 0.0)


def _init_centres(Z, yk, T, K, rng):
    C = np.zeros((K, Z.shape[1]))
    for c in range(T):
        idx = np.where(yk == c)[0]
        C[c] = Z[idx].mean(0) if len(idx) else Z[rng.integers(len(Z))]
    # free clusters: k-means++ seeding among the cells farthest from the known centres
    for k in range(T, K):
        d = _sqd(Z, C[:k]).min(1)
        p = d / d.sum()
        C[k] = Z[rng.choice(len(Z), p=p)]
    return C


def ssfcm(Z, yk, T, K, alpha, seed=0, max_iter=100, tol=1e-4):
    rng = np.random.default_rng(seed)
    n = len(Z)
    F = np.zeros((n, K)); b = (yk >= 0).astype(float)
    F[b > 0, yk[b > 0]] = 1.0
    C = _init_centres(Z, yk, T, K, rng)
    U = np.full((n, K), 1.0 / K)
    for it in range(max_iter):
        D = _sqd(Z, C) + 1e-10        # squared distances d_ik^2
        inv = 1.0 / D
        fcm = inv / inv.sum(1, keepdims=True)                               # 1 / sum_j (d_ik^2 / d_ij^2)
        fb = F * b[:, None]
        Un = (fcm * (1 + alpha * (1 - fb.sum(1, keepdims=True))) + alpha * fb) / (1 + alpha)
        phi = Un ** 2 + alpha * (Un - fb) ** 2
        C = (phi.T @ Z) / np.maximum(phi.sum(0)[:, None], 1e-12)
        delta = np.abs(Un - U).max(); U = Un
        if delta < tol:
            break
    return U, C, it + 1


def select_and_fit_ssfcm(Z, yk, T, K, seed, alphas=None, holdout=0.2):
    lab = np.where(yk >= 0)[0]
    n_lab = len(lab)
    if alphas is None:
        r = len(Z) / max(n_lab, 1)
        alphas = sorted({0.5, 1.0, 2.0, 5.0, round(r, 3)})
    rng = np.random.default_rng(1000 + seed)
    ho = rng.choice(lab, size=max(1, int(holdout * n_lab)), replace=False)
    ytr = yk.copy(); ytr[ho] = -1
    best, best_a = -1, alphas[0]
    for a in alphas:
        U, _, _ = ssfcm(Z, ytr, T, K, a, seed)
        acc = float((U[ho].argmax(1) == yk[ho]).mean())
        if acc > best:
            best, best_a = acc, a
    U, C, it = ssfcm(Z, yk, T, K, best_a, seed)
    return U, dict(alpha=best_a, n_iter=it)
