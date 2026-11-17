"""View-specific autoencoders and count likelihoods."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, last_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers)


class ViewAE(nn.Module):
    """Encoder -> d-dim latent (batch-standardised, so the latent scale cannot collapse) -> decoder."""

    def __init__(self, n_in, lik, d=32, hidden=(256, 128)):
        super().__init__()
        self.lik = lik
        self.enc = mlp([n_in, *hidden], last_act=True)
        self.to_z = nn.Linear(hidden[-1], d)
        self.bn = nn.BatchNorm1d(d, affine=False)
        self.dec = mlp([d, *hidden[::-1]], last_act=True)
        h = hidden[0]
        if lik in ('nb', 'zinb'):
            self.mean = nn.Linear(h, n_in)
            self.log_theta = nn.Parameter(torch.zeros(n_in))
            if lik == 'zinb':
                self.pi = nn.Linear(h, n_in)
        else:
            self.out = nn.Linear(h, n_in)

    def encode(self, x):
        return self.bn(self.to_z(self.enc(x)))

    def recon_loss(self, z, target, sf):
        """Mean negative log-likelihood per feature and cell."""
        h = self.dec(z)
        if self.lik == 'gaussian':
            return F.mse_loss(self.out(h), target)
        if self.lik == 'bernoulli':
            return F.binary_cross_entropy_with_logits(self.out(h), target)
        mu = torch.softmax(self.mean(h), dim=1) * target.sum(1, keepdim=True).clamp(min=1.0)
        theta = torch.exp(self.log_theta).clamp(1e-4, 1e4)
        nb = _nb_logpmf(target, mu, theta)
        if self.lik == 'nb':
            return -nb.mean()
        logit_pi = self.pi(h)
        zero = target < 0.5
        log_nb_zero = theta * (torch.log(theta + 1e-8) - torch.log(theta + mu + 1e-8))
        ll_zero = torch.logaddexp(logit_pi, log_nb_zero) - F.softplus(logit_pi)
        ll_nz = nb - F.softplus(logit_pi)
        return -torch.where(zero, ll_zero, ll_nz).mean()


def _nb_logpmf(x, mu, theta):
    eps = 1e-8
    return (torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
            + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
            + x * (torch.log(mu + eps) - torch.log(theta + mu + eps)))
