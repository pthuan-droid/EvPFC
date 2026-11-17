"""Download the official code of scMDC and scEMC into round2/ext/ and apply the minimal patches needed to run it
with current Python / PyTorch / pandas on CPU or GPU. The patches change no computation:

  scEMC/utils.py   : `import mkl` made optional (it is only used to query the number of threads);
                     hard-coded `.cuda()` replaced by `.to(device)` (device = cuda if available, else cpu)
  scEMC/network.py : `.to('cuda:0')` in kmeans_loss replaced by `.to(z.device)`;
                     in kmeans_loss the per-row shift before exp() uses the row minimum instead of the row mean.
                     q is normalised per row, so q is mathematically unchanged, but exp() no longer overflows in
                     float32 when distances are large (the original line produced NaN and a single cluster).
  (scMDC needs no patch; r16_unsup.py passes size factors as numpy arrays instead of pandas Series)

Commits used in the paper: scMDC 0c8778e3112d5b791e0b42569e6c502f8309cfd6, scEMC 7baeffe6f13a1a91597db11b5c0f8fd04b75a939

    python setup_ext.py
    pip install munkres            # needed by scEMC/utils.py
"""
import os, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, 'ext')
REPOS = {'scMDC': ('https://github.com/xianglin226/scMDC.git', '0c8778e3112d5b791e0b42569e6c502f8309cfd6'),
         'scEMC': ('https://github.com/DayuHuu/scEMC.git', '7baeffe6f13a1a91597db11b5c0f8fd04b75a939')}


def sh(*a, cwd=None):
    print('>', ' '.join(a)); subprocess.run(a, check=True, cwd=cwd)


def main():
    os.makedirs(EXT, exist_ok=True)
    for name, (url, commit) in REPOS.items():
        d = os.path.join(EXT, name)
        if not os.path.exists(d):
            sh('git', 'clone', url, d)
        sh('git', 'checkout', '-q', commit, cwd=d)
    u = os.path.join(EXT, 'scEMC', 'scEMC', 'utils.py')
    s = open(u, encoding='utf-8').read()
    s = s.replace('\nimport mkl\n', '\ntry:\n    import mkl\nexcept ImportError:\n    mkl = None\n')
    s = re.sub(r'\nmkl\.get_max_threads\(\)\n', '\nmkl.get_max_threads() if mkl else None\n', s)
    s = s.replace('.cuda()', '.to(device)')
    open(u, 'w', encoding='utf-8').write(s)
    n = os.path.join(EXT, 'scEMC', 'scEMC', 'network.py')
    s = open(n, encoding='utf-8').read()
    s = s.replace("torch.tensor(self.tau).to('cuda:0')", 'torch.tensor(self.tau).to(z.device)')
    s = s.replace("torch.tensor(self.mu).to('cuda:0')", 'torch.tensor(self.mu).to(z.device)')
    old = 'temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])'
    if old in s:
        s = s.replace(old, 'temp_dist1 = dist1 - torch.reshape(torch.min(dist1, dim=1).values, [-1, 1])  '
                           '# was: row mean; row min avoids exp overflow, q is unchanged after row normalisation')
    open(n, 'w', encoding='utf-8').write(s)
    print('ok: code in', EXT)


if __name__ == '__main__':
    main()
