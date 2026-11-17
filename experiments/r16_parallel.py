"""Run the scMDC / scEMC comparison (R16) on a multi-core CPU, several jobs in parallel.

One job = one (method, data set, seed). Each job runs r16_unsup.py in its own process with a fixed number of
threads and writes its own small CSV in results/R16_parts/, so parallel jobs never write to the same file.
Finished jobs are skipped when the script is started again, so it can be stopped (Ctrl+C) and resumed at any time.
After every finished job all parts are merged into results/R16_unsup.csv.

Jobs are ordered seed by seed (all data sets with seed 3, then seed 4, ...), so that if the run is stopped early
every data set already has a few seeds.

    python r16_parallel.py                      # auto: threads per job = 4, jobs in parallel = cores // 4
    python r16_parallel.py --threads 4 --jobs 3
    python r16_parallel.py --only scMDC         # only scMDC jobs
    python r16_parallel.py --dry-run            # show the job list and exit
"""
import argparse, csv, glob, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
PARTS = os.path.join(RES, 'R16_parts')

# (method, data set) in order of priority
TASKS = [('scMDC', 'SLN111'), ('scMDC', 'SLN206'), ('scMDC', 'CellMix'),
         ('scEMC', 'Ma2020_1'), ('scEMC', 'PBMC10k_Multiome_feat'), ('scEMC', 'SMAGE3K'), ('scEMC', 'CellMix'),
         ('scEMC', 'BMNC'), ('scEMC', 'SLN111'), ('scEMC', 'SLN206')]
SEEDS = [3, 4, 5, 6, 7]


def part_file(m, ds, s):
    return os.path.join(PARTS, f'{m}__{ds}__s{s}.csv')


def done(m, ds, s):
    f = part_file(m, ds, s)
    if not os.path.exists(f):
        return False
    with open(f, newline='', encoding='utf-8') as fh:
        return any(r.get('ARI') not in (None, '') for r in csv.DictReader(fh))


def merge():
    rows, cols = [], None
    for f in sorted(glob.glob(os.path.join(PARTS, '*.csv'))):
        with open(f, newline='', encoding='utf-8') as fh:
            rd = csv.DictReader(fh)
            cols = cols or rd.fieldnames
            rows += list(rd)
    if rows:
        with open(os.path.join(RES, 'R16_unsup.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
            w.writeheader(); w.writerows(rows)
    return len(rows)


def run(job, a):
    m, ds, s = job
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
               OPENBLAS_NUM_THREADS=str(a.threads))
    log = os.path.join(PARTS, 'logs', f'{m}__{ds}__s{s}.log')
    cmd = [sys.executable, os.path.join(HERE, 'r16_unsup.py'), '--method', m, '--datasets', ds, '--seeds', str(s),
           '--threads', str(a.threads), '--device', 'cpu', '--out', os.path.relpath(part_file(m, ds, s), RES)]
    if a.data_root:
        cmd += ['--data-root', a.data_root]
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
    return job, rc, time.time() - t0, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=4, help='CPU threads per job')
    ap.add_argument('--jobs', type=int, default=None, help='jobs in parallel (default: cores // threads)')
    ap.add_argument('--only', choices=['scMDC', 'scEMC'], default=None)
    ap.add_argument('--seeds', nargs='*', type=int, default=SEEDS)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--datasets', nargs='*', default=None,
                    help='run scMDC and scEMC on these data sets instead of the default list')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--redo', choices=['scEMC', 'scMDC'], default=None,
                    help='delete all finished jobs of this method and run them again (e.g. after setup_ext.py patched it)')
    a = ap.parse_args()
    os.makedirs(os.path.join(PARTS, 'logs'), exist_ok=True)
    if a.redo and not a.dry_run:
        for f in glob.glob(os.path.join(PARTS, f'{a.redo}__*.csv')):
            os.remove(f)
        print('removed old results of', a.redo, '-> will be re-run')
    cores = os.cpu_count() or 4
    n_jobs = a.jobs or max(1, cores // a.threads)
    tasks = [(m, ds) for ds in a.datasets for m in ('scMDC', 'scEMC')] if a.datasets else TASKS
    jobs = [(m, ds, s) for s in a.seeds for (m, ds) in tasks if (a.only in (None, m)) and not done(m, ds, s)]
    print(f'{cores} logical cores; {n_jobs} jobs in parallel x {a.threads} threads; {len(jobs)} jobs to run')
    for j in jobs:
        print('  ', *j)
    if a.dry_run or not jobs:
        print('merged rows:', merge()); return
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futs = [ex.submit(run, j, a) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            (m, ds, s), rc, dt, log = f.result()
            ok = 'done' if rc == 0 and done(m, ds, s) else f'ERROR (see {log})'
            n = merge()
            print(time.strftime('%H:%M:%S'), f'[{k}/{len(jobs)}]', m, ds, f'seed {s}', ok, f'{dt / 60:.1f} min',
                  f'| tong {(time.time() - t_start) / 3600:.1f} gio | R16_unsup.csv: {n} dong', flush=True)
    print('Done. Results: results/R16_unsup.csv')


if __name__ == '__main__':
    main()
