"""Belief-mass (picture-fuzzy) partitions and their combination, batched in PyTorch.

A batch of basic belief assignments (BBAs) over the frame Theta = {0..K-1} is stored with
three kinds of focal sets, which is all scPIKA ever needs:
    S  : (n, K)      mass on singletons {k}                     -> positive degree mu
    P  : (n, Q)      mass on Q two-element sets (pairs)         -> neutral degree eta
    Mk : (n, Q, K)   0/1 indicator of the members of each pair
    T  : (n,)        mass on Theta (total ignorance)            -> refusal degree xi
Masses on a row sum to one. Combining two BBAs only produces singletons, pairs, Theta and
the empty set, so the representation is closed under the conjunctive rule.
"""
from dataclasses import dataclass
import torch


@dataclass
class BBA:
    S: torch.Tensor
    P: torch.Tensor
    Mk: torch.Tensor
    T: torch.Tensor

    @property
    def K(self):
        return self.S.shape[1]

    def pair_mass_per_class(self):
        if self.P.shape[1] == 0:
            return torch.zeros_like(self.S)
        return (self.P.unsqueeze(-1) * self.Mk).sum(1)

    def betp(self):
        """Pignistic probability BetP(k) = sum_{A containing k} m(A)/|A|."""
        return self.S + 0.5 * self.pair_mass_per_class() + self.T.unsqueeze(1) / self.K

    def eta(self):
        return self.P.sum(1) if self.P.shape[1] else torch.zeros_like(self.T)

    def total(self):
        return self.S.sum(1) + self.eta() + self.T

    @staticmethod
    def vacuous(n, K, Q=0, device=None):
        return BBA(torch.zeros(n, K, device=device), torch.zeros(n, Q, device=device),
                   torch.zeros(n, Q, K, device=device), torch.ones(n, device=device))

    def where(self, keep, other):
        """Row-wise select: rows with keep=True from self, others from `other` (same shapes)."""
        k = keep.float()
        return BBA(self.S * k[:, None] + other.S * (1 - k[:, None]),
                   self.P * k[:, None] + other.P * (1 - k[:, None]),
                   self.Mk * k[:, None, None] + other.Mk * (1 - k[:, None, None]),
                   self.T * k + other.T * (1 - k))


def conjunctive(A: BBA, B: BBA):
    """Unnormalised conjunctive combination. Returns (BBA without the empty set, conflict)."""
    n, K = A.S.shape
    plA, plB = A.pair_mass_per_class(), B.pair_mass_per_class()
    S = A.S * B.S + A.S * plB + A.S * B.T[:, None] + B.S * plA + B.S * A.T[:, None]
    Ps, Ms = [], []
    Qa, Qb = A.P.shape[1], B.P.shape[1]
    if Qa and Qb:
        inter = A.Mk.unsqueeze(2) * B.Mk.unsqueeze(1)              # n,Qa,Qb,K
        size = inter.sum(-1)                                       # n,Qa,Qb
        w = A.P.unsqueeze(2) * B.P.unsqueeze(1)                    # n,Qa,Qb
        S = S + ((w * (size == 1).float()).unsqueeze(-1) * inter).sum((1, 2))
        same = (w * (size == 2).float()).reshape(n, Qa * Qb)
        Ps.append(same)
        Ms.append(inter.reshape(n, Qa * Qb, K))
    if Qa:
        Ps.append(A.P * B.T[:, None]); Ms.append(A.Mk)
    if Qb:
        Ps.append(B.P * A.T[:, None]); Ms.append(B.Mk)
    if Ps:
        P, Mk = torch.cat(Ps, 1), torch.cat(Ms, 1)
    else:
        P, Mk = torch.zeros(n, 0, device=S.device), torch.zeros(n, 0, K, device=S.device)
    T = A.T * B.T
    out = BBA(S, P, Mk, T)
    conflict = (A.total() * B.total() - out.total()).clamp(min=0.0)   # exact even if inputs drift from 1
    return out, conflict


def dempster(A: BBA, B: BBA, eps=1e-12):
    """Dempster's rule (conjunctive combination followed by normalisation of the conflict).
    The normaliser is the total non-conflicting mass actually present (equal to 1 - k in exact
    arithmetic); dividing by it keeps rows summing to one even when k is close to 1."""
    out, k = conjunctive(A, B)
    z = out.total().clamp(min=eps)
    return BBA(out.S / z[:, None], out.P / z[:, None], out.Mk, out.T / z), k


def conflict_to_refusal(A: BBA, B: BBA):
    """Yager-type rule: the conflicting mass is transferred to Theta (refusal)."""
    out, k = conjunctive(A, B)
    return BBA(out.S, out.P, out.Mk, out.T + k), k


def fuse_views(bbas, rule='dempster'):
    """Combine a list of BBAs with the given rule. Returns fused BBA and the accumulated conflict
    1 - prod_j (1 - k_j), i.e. the share of mass that was conflicting at some step."""
    fused = bbas[0]
    keep = torch.ones_like(fused.T)
    for b in bbas[1:]:
        if rule == 'dempster':
            fused, k = dempster(fused, b)
        else:
            fused, k = conflict_to_refusal(fused, b)
        keep = keep * (1.0 - k)
    return fused, 1.0 - keep
