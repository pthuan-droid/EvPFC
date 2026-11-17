"""Download the raw-count data needed by round 2 (run on a machine with normal internet access).

    python download_round2.py            # everything
    python download_round2.py PBMC3K     # 10x Multiome PBMC 3k (two versions, ~0.5 GB in total)
    python download_round2.py BMMC       # NeurIPS 2021 BMMC CITE + Multiome processed .h5ad.gz (several GB)

Files go to <repo>/datasets/... or $EVPFC_DATA (same layout as the other datasets). Interrupted downloads resume
(HTTP Range), so the command can simply be run again.
"""
import os, sys, re, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.environ.get('EVPFC_DATA', os.path.join(HERE, '..', 'datasets')))

TENX = [  # both 10x PBMC 3k Multiome releases; r2_data.py keeps the one whose barcodes match the labels
    ('RNA_ATAC/PBMC3k_sorted_10x',
     'https://cf.10xgenomics.com/samples/cell-arc/1.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5'),
    ('RNA_ATAC/PBMC3k_unsorted_10x',
     'https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_3k/pbmc_unsorted_3k_filtered_feature_bc_matrix.h5'),
]
GEO_BMMC = ('RNA_ATAC_and_RNA_ADT/BMMC_NeurIPS2021', 'GSE194122', r'(cite|multiome)_BMMC_processed\.h5ad\.gz$')


def fetch(url, outdir, tries=5):
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, url.rstrip('/').split('/')[-1])
    part = out + '.part'
    try:
        total = int(urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60)
                    .headers.get('Content-Length', 0))
    except Exception:
        total = 0
    if os.path.exists(out) and (total == 0 or os.path.getsize(out) == total):
        print('  da co', out); return
    for t in range(tries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total and have >= total:
            break
        req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
        print(f'  downloading {os.path.basename(out)}: {have / 1e6:.0f}/{total / 1e6:.0f} MB', flush=True)
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, 'ab' if have else 'wb') as f:
                if have and r.status != 206:      # server ignored Range: start again
                    f.seek(0); f.truncate()
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
        except Exception as e:
            print('  error, retrying:', e); time.sleep(10)
    if total and os.path.getsize(part) != total:
        print('  DOWNLOAD INCOMPLETE, run the command again to resume:', out); return
    os.replace(part, out)
    print('  done', out)


def geo(outdir, acc, pat):
    base = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/suppl/'
    html = urllib.request.urlopen(base, timeout=60).read().decode()
    files = sorted(set(f for f in re.findall(r'href="([^"?/][^"]*)"', html) if re.search(pat, f)))
    print(acc, files)
    for f in files:
        fetch(base + f, os.path.join(ROOT, outdir))


if __name__ == '__main__':
    want = set(a.upper() for a in sys.argv[1:]) or {'PBMC3K', 'BMMC'}
    if 'PBMC3K' in want:
        for d, u in TENX:
            fetch(u, os.path.join(ROOT, d))
    if 'BMMC' in want:
        geo(*GEO_BMMC)
