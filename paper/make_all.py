"""Rebuild the tables and figures of the paper from the per-run results in results/.

    python paper/make_all.py              # tables + figures from results/per_run/results_tidy.csv
    python paper/make_all.py --rebuild    # first re-assemble results_tidy.csv from the raw files in results/raw
                                          # and check that it reproduces the tables of the paper

Output: paper/output/tables/*.md|csv and paper/output/fig/*.png|pdf.

    Table 3 (ARI, 5 % labels)        tables/tab_main.md          stats2.py
    Table 2 (summary, bootstrap)     tables/tab_summary.md       stats2.py
    Table 4 (NMI, 5 % labels)       tables/tab_main_nmi.md      stats2.py
    Table 7 (tuned baselines)        tables/tab_tuned.md         r18_summary.py
    Table 6 (fixed parameters)      tables/tab_sens.md          r18_summary.py
    Fig. 1 (method), Fig. 6 (toy)    fig/fig1_evpfc, fig2_toy    fig_method.py, fig_toy.py
    Fig. 2, 3, 4, 9                  fig/fig3_..., fig6_missing  fig_results.py
    Fig. 5, 7, 10, 8                 fig/fig7_..., fig10_runtime fig_extra.py
Table 5 (open-world criteria) is computed by experiments/r8_stats.py from results/raw/R8b_conformal.csv.
Fig. 7(a-c) needs experiments/results/R11_cells_Ma2020_1_s3.npz, written by experiments/r11_cells.py.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import OUT, TIDY, CELLS  # noqa: E402


def run(args, env=None):
    print('$', ' '.join(args), flush=True)
    e = dict(os.environ, PYTHONPATH=HERE + os.pathsep + os.environ.get('PYTHONPATH', ''), MPLBACKEND='Agg', **(env or {}))
    r = subprocess.run([sys.executable] + args, cwd=OUT, env=e)
    if r.returncode:
        print(f'   (exit code {r.returncode})', flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true', help='re-assemble results_tidy.csv from results/raw first')
    ap.add_argument('--skip-toy', action='store_true', help='skip Fig. 6 (needs PyTorch)')
    a = ap.parse_args()
    for d in ('tables', 'fig'):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    env = {}
    if a.rebuild:
        for s in ('collect.py', 'collect_add.py', 'collect_bmmc.py'):
            run([os.path.join(HERE, s)])
        env = {'EVPFC_TIDY': os.path.join(OUT, 'results_tidy_rebuilt.csv')}
    run([os.path.join(HERE, 'stats2.py')], env)
    run([os.path.join(HERE, 'fig_results.py'), '3', '4', '6', '7'], env)
    figs = ['7', '9', '10'] + (['8'] if os.path.exists(os.path.join(CELLS, 'R11_cells_Ma2020_1_s3.npz')) else [])
    if '8' not in figs:
        print('   Fig. 7 skipped: run experiments/r11_cells.py first (needs the cached latent spaces).')
    run([os.path.join(HERE, 'fig_extra.py')] + figs, env)
    run([os.path.join(HERE, 'r18_summary.py')], env)
    run([os.path.join(HERE, 'fig_method.py')])
    if not a.skip_toy:
        run([os.path.join(HERE, 'fig_toy.py')])
    print('done ->', OUT)


if __name__ == '__main__':
    main()
