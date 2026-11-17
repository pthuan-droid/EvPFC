"""Locations used by the scripts that rebuild the tables and figures of the paper."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, 'results', 'raw')                      # per-run CSV files written by experiments/
BMMC = os.path.join(RAW, 'bmmc_machine')                        # BMMC runs (R1-R3, R16, R17) made on a second workstation
R18 = os.path.join(RAW, 'review_R18')                           # robustness checks of Appendix B.9
TIDY = os.environ.get('EVPFC_TIDY', os.path.join(ROOT, 'results', 'per_run', 'results_tidy.csv'))
OUT = os.environ.get('EVPFC_OUT', os.path.join(HERE, 'output'))  # tables/ and fig/ are written here
CELLS = os.environ.get('EVPFC_CELLS', os.path.join(ROOT, 'experiments', 'results'))  # R11_cells_<ds>_s3.npz (r11_cells.py)
