import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score, roc_auc_score


def hungarian_map(y_true, y_pred):
    """Map predicted cluster ids to true labels maximising agreement (clusters left unmatched keep -1)."""
    t, p = np.unique(y_true), np.unique(y_pred)
    W = np.zeros((p.size, t.size))
    for i, a in enumerate(p):
        for j, b in enumerate(t):
            W[i, j] = np.sum((y_pred == a) & (y_true == b))
    r, c = linear_sum_assignment(-W)
    mp = {p[i]: t[j] for i, j in zip(r, c)}
    return np.array([mp.get(v, -1) for v in y_pred])


def clustering_scores(y_true, y_pred):
    ym = hungarian_map(y_true, y_pred)
    return dict(ARI=adjusted_rand_score(y_true, y_pred),
                NMI=normalized_mutual_info_score(y_true, y_pred),
                ACC=float(np.mean(ym == y_true)),
                F1=f1_score(y_true, ym, average='macro', labels=np.unique(y_true), zero_division=0))


def auroc(is_positive, score):
    is_positive = np.asarray(is_positive).astype(int)
    if is_positive.min() == is_positive.max():
        return float('nan')
    return float(roc_auc_score(is_positive, score))


def selective_scores(y_true, proba, conf=None, coverages=(0.5, 0.8, 1.0)):
    """Selective-prediction quality of class probabilities whose columns are the classes.
    Returns the expected calibration error (15 bins), the area under the risk-coverage curve (AURC,
    lower is better) and the accuracy on the most confident fraction of cells."""
    proba = np.asarray(proba, np.float64)
    pred = proba.argmax(1)
    conf = proba.max(1) if conf is None else np.asarray(conf, np.float64)
    correct = (pred == y_true).astype(float)
    pmax = proba.max(1)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pmax > lo) & (pmax <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - pmax[m].mean())
    order = np.argsort(-conf, kind='stable')
    cum = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    aurc = float(np.mean(1.0 - cum))
    out = dict(ECE=float(ece), AURC=aurc)
    for c in coverages:
        k = max(1, int(round(c * len(order))))
        out[f'acc@{int(c * 100)}'] = float(cum[k - 1])
    return out
