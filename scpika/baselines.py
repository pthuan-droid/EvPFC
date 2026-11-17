"""Reference methods that share scPIKA's preprocessing.

All operate on a joint embedding built from per-view PCA (30 components per view, each block
scaled to unit total variance). Missing views are filled with the per-view mean of observed
cells (zero after centring), which is what a method without a missing-view mechanism does.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelSpreading


def joint_embedding(views, n_comp=30, seed=0):
    blocks = []
    for v in views:
        m = v['mask']
        X = v['x']
        Xm = X[m]
        k = min(n_comp, Xm.shape[1] - 1, Xm.shape[0] - 1)
        if sp.issparse(Xm):
            if v['lik'] == 'bernoulli':  # TF-IDF + LSI for accessibility
                tf = sp.diags(1.0 / np.maximum(np.asarray(Xm.sum(1)).ravel(), 1)) @ Xm
                idf = np.log1p(Xm.shape[0] / np.maximum(np.asarray(Xm.sum(0)).ravel(), 1))
                Xm = (tf @ sp.diags(idf)).log1p() if hasattr(tf, 'log1p') else tf
                E = TruncatedSVD(k + 1, random_state=seed).fit_transform(Xm)[:, 1:]
            else:
                E = TruncatedSVD(k, random_state=seed).fit_transform(Xm)
        else:
            E = PCA(k, random_state=seed).fit_transform(np.asarray(Xm, np.float32))
        E = (E - E.mean(0)) / (np.sqrt((E.var(0)).sum()) + 1e-8)
        B = np.zeros((X.shape[0], E.shape[1]), np.float32)
        B[m] = E
        blocks.append(B)
    return np.concatenate(blocks, 1)


def kmeans(E, K, seed=0):
    return KMeans(K, n_init=10, random_state=seed).fit_predict(E)


def seeded_kmeans(E, K, y_known, seed=0):
    """Seeded k-means (Basu et al., 2002): centres of known classes initialised from labelled cells."""
    rng = np.random.RandomState(seed)
    lab = np.where(y_known >= 0)[0]
    T = int(y_known.max()) + 1 if lab.size else 0
    init = []
    for t in range(T):
        m = lab[y_known[lab] == t]
        init.append(E[m].mean(0) if m.size else E[rng.randint(E.shape[0])])
    while len(init) < K:
        init.append(E[rng.randint(E.shape[0])])
    return KMeans(K, init=np.stack(init), n_init=1, random_state=seed).fit_predict(E)


def fcm(E, K, m=2.0, iters=150, seed=0, tol=1e-5):
    """Fuzzy c-means (Bezdek). Returns hard labels and the membership matrix."""
    rng = np.random.RandomState(seed)
    U = rng.dirichlet(np.ones(K), size=E.shape[0])
    for _ in range(iters):
        W = U ** m
        C = (W.T @ E) / W.sum(0)[:, None]
        D = ((E[:, None, :] - C[None]) ** 2).sum(-1) + 1e-12
        Un = 1.0 / (D ** (1.0 / (m - 1)))
        Un /= Un.sum(1, keepdims=True)
        if np.abs(Un - U).max() < tol:
            U = Un
            break
        U = Un
    return U.argmax(1), U


def label_spreading(E, y_known, n_neighbors=15, alpha=0.2):
    """Label spreading on a kNN graph (Zhou et al., 2004). Predicts only known classes."""
    ls = LabelSpreading(kernel='knn', n_neighbors=n_neighbors, alpha=alpha, max_iter=100)
    ls.fit(E, y_known)
    return ls.transduction_, ls.label_distributions_
