"""R17: run the whole protocol of the paper on the NeurIPS 2021 BMMC data (GSE194122), CITE-seq and Multiome.

The two files must be in  datasets/RNA_ATAC_and_RNA_ADT/BMMC_NeurIPS2021/  (.h5ad.gz or .h5ad):
    GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad(.gz)
    GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad(.gz)
Only the cells of site 1 (donors s1d1, s1d2, s1d3) are used, as set in r2_data.BMMC_BATCHES.

Stages (run in this order; each stage can be stopped and restarted, finished runs are skipped):
  check     load both data sets, print cells / types / batches (a few minutes; decompresses the .gz once)
  base      label-free encoders (PCA, scVI, totalVI or MultiVI; seeds 3-7) + LS, kNN, eKNN, seeded k-means,
            scANVI and the end-to-end variant with 1 / 5 / 10 % labels            (r2_run.py --exp R1)
  noise     20 % wrong labels: LS, kNN, eKNN, end-to-end                          (r2_run.py --exp R2)
  hide      two hidden types: LS, kNN, eKNN, end-to-end                           (r2_run.py --exp R3)
  scanvi    scANVI with wrong labels and with hidden types                        (r14_fill.py)
  evpfc     EvPFC, EvPFC without prior, eKNN, LS, kNN, SSFCM on the main latent, all settings, seeds 3-7
  ablation  EvPFC variants (w_i = 1, no eta, no xi, neither), seeds 3-7
  confirm   EvPFC, eKNN, LS, kNN on the confirmation seeds 8-12
  cells     per-cell analysis (5 % labels)                                        (r11_cells.py)
  unsup     scMDC and scEMC, seeds 3-7                                            (r16_parallel.py)

    python r17_bmmc.py --stage check
    python r17_bmmc.py                      # all stages after 'check', in order
    python r17_bmmc.py --stage evpfc        # one stage only
Logs: results/R17_logs/<stage>.log
"""
import argparse, glob, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DS = ['BMMC_CITE', 'BMMC_Multiome']
SEEDS = ['3', '4', '5', '6', '7']
CONF = ['8', '9', '10', '11', '12']
PY = sys.executable


def stages(a):
    t = ['--threads', str(a.threads)]
    dr = ['--data-root', a.data_root] if a.data_root else []
    ds = a.datasets
    return {
        'check': None,
        'base': [[PY, 'r2_run.py', '--exp', 'R1', '--datasets', *ds, '--seeds', *SEEDS, *t, *dr]],
        'noise': [[PY, 'r2_run.py', '--exp', 'R2', '--datasets', *ds, '--seeds', *SEEDS, '--noises', '0.2', *t, *dr]],
        'hide': [[PY, 'r2_run.py', '--exp', 'R3', '--datasets', *ds, '--seeds', *SEEDS, *t, *dr]],
        'scanvi': [[PY, 'r14_fill.py', '--what', 'scanvi', '--scanvi-hide', '--scanvi-datasets', *ds, '--seeds', *SEEDS,
                    '--out', 'R17_scanvi.csv', *t, *dr]],
        'evpfc': [[PY, 'r7_evpfc.py', '--datasets', *ds, '--seeds', *SEEDS, '--out', 'R17_evpfc.csv',
                   '--heads', 'EvPFC', 'EvPFC-noprior', 'eKNN', 'LS', 'kNN', 'SSFCM', *t, *dr]],
        'ablation': [[PY, 'r7_evpfc.py', '--datasets', *ds, '--seeds', *SEEDS, '--out', 'R17_ablation.csv',
                      '--settings', 'frac1', 'frac5', 'noise20', 'hide2',
                      '--heads', 'EvPFC-constw', 'EvPFC-nopair', 'EvPFC-norefusal', 'EvPFC-plain', *t, *dr]],
        'confirm': [[PY, 'r7_evpfc.py', '--datasets', *ds, '--seeds', *CONF, '--out', 'R17_confirm.csv',
                     '--heads', 'EvPFC', 'eKNN', 'LS', 'kNN', *t, *dr]],
        'cells': [[PY, 'r11_cells.py', '--datasets', *ds, '--out', 'R17_cells_bins.csv']],
        'unsup': [[PY, 'r16_parallel.py', '--datasets', *ds, '--threads', str(max(2, a.threads // 2)), *dr]],
    }


PATTERN = 'GSE194122_openproblems_neurips2021_{}_BMMC_processed.h5ad'


def find_bmmc(user_dir=None):
    """Folder that holds both BMMC files (.h5ad or .h5ad.gz). Searched in --bmmc-dir, then in the folder of this
    script, its parent (scpika), the parent's parent, and datasets/RNA_ATAC_and_RNA_ADT/BMMC_NeurIPS2021 - up to
    four levels deep under each."""
    roots = [user_dir] if user_dir else [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for depth in range(5):
            pat = os.path.join(r, *(['*'] * depth), PATTERN.format('cite') + '*')
            for f in glob.glob(pat):
                if f.endswith('.part') or f.endswith('.tmp'):
                    continue
                folder = os.path.dirname(f)
                if glob.glob(os.path.join(folder, PATTERN.format('multiome') + '*')):
                    return folder
    return None


def check(a):
    sys.path.insert(0, HERE)
    import numpy as np
    import r2_data
    if a.data_root:
        r2_data.set_root(a.data_root)
    for ds in a.datasets:
        t0 = time.time()
        d = r2_data.load(ds)
        y = d['y']
        cnt = np.bincount(y)
        print(f'{ds}: {len(y)} cells, {len(cnt)} types (smallest {cnt.min()}, largest {cnt.max()}), '
              f'views {d["names"]} {[v.shape[1] for v in d["views"]]}, '
              f'batches {sorted(set(d["meta"]["batch"])) if d["meta"].get("batch") is not None else "-"}, '
              f'{time.time() - t0:.0f} s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', nargs='*', default=None, help='stages to run (default: all except check)')
    ap.add_argument('--datasets', nargs='*', default=DS)
    ap.add_argument('--threads', type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--bmmc-dir', default=None, help='folder with the two BMMC files (default: search around this script)')
    a = ap.parse_args()
    folder = find_bmmc(a.bmmc_dir)
    if folder is None:
        sys.exit('The two BMMC files (cite and multiome, .h5ad or .h5ad.gz) were not found. Use --bmmc-dir <folder>.')
    os.environ['BMMC_DIR'] = folder
    print('BMMC:', folder, flush=True)
    S = stages(a)
    todo = a.stage or [s for s in S if s != 'check']
    logdir = os.path.join(HERE, 'results', 'R17_logs')
    os.makedirs(logdir, exist_ok=True)
    for s in todo:
        if s not in S:
            sys.exit(f'unknown stage {s}; choose from {list(S)}')
        t0 = time.time()
        print(time.strftime('%H:%M:%S'), 'stage', s, '...', flush=True)
        if s == 'check':
            check(a); continue
        log = os.path.join(logdir, f'{s}.log')
        for cmd in S[s]:
            with open(log, 'a', encoding='utf-8') as fh:
                fh.write('\n$ ' + ' '.join(cmd) + '\n'); fh.flush()
                rc = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                print(f'   ERROR (code {rc}) - see {log}', flush=True)
                break
        print(time.strftime('%H:%M:%S'), 'stage', s, f'done ({(time.time() - t0) / 60:.0f} min)', flush=True)


if __name__ == '__main__':
    main()
