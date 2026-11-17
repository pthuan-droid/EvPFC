# EvPFC – picture fuzzy clustering with an evidential prior

Code, label draws and per-run results for the article

> **Evidential priors for semi-supervised picture fuzzy clustering with noisy partial labels in single-cell multi-omics data.**
> Pham Huy Thong (Hanoi University of Industry) and Phung The Huan (Thai Nguyen University of Information and Communication Technology).
> Submitted to *Pattern Analysis and Applications*.

Table, figure and section numbers below refer to the submitted version of the article. The file names written by the
scripts in `paper/output/` (e.g. `fig3_labelfrac`) keep the numbering of an earlier draft; the tables below give the correspondence.

EvPFC clusters cells in a fixed latent space learned **without labels** (totalVI, MultiVI or an autoencoder).
Partial labels enter only through an evidential k-NN, whose pignistic probabilities form a prior in a picture
fuzzy energy. The strength of that prior for each cell is the confidence $w_i = 1 - m_i(\Theta)$ of the k-NN mass function. All
updates have a closed form. An optional split-conformal threshold on $m_i(\Theta)$ hands cells to free clusters
(open-world mode).

The repository lets a reader:

1. **Inspect** every number of the paper. `results/` holds the per-run results of all experiments, and `label_draws/` holds the exact labelled cells, corrupted labels and hidden types of every run.
2. **Rebuild** every table and figure from those results in a few minutes on a laptop (`python paper/make_all.py`).
3. **Re-run** the experiments from the public data. This takes hours to days on a CPU, mainly because of the encoders.

## Repository layout

```
EvPFC/
├── scpika/            shared package: data handling, metrics, evidential k-NN, simple baselines, and the
│                      end-to-end variant "E2E" of the paper (called scPIKA in the code)
├── experiments/       all experiment scripts; EvPFC itself is experiments/evpfc.py
│   ├── evpfc.py           EvPFC model, parameter selection, conformal open-world mode
│   ├── ssfcm.py           semi-supervised fuzzy c-means (Pedrycz & Waletzky 1997) baseline
│   ├── r2_core.py         protocol (label draws), encoders, baselines, CSV writer
│   ├── r2_data.py         data loading (incl. BMMC), r2_run.py: R1–R4
│   ├── r7_evpfc.py        EvPFC and the classifiers on cached latents (R7–R10, R14–R15)
│   ├── r8b_conformal.py   open-world mode, r8_stats.py: pre-registered criteria (Table 5)
│   ├── r11_cells.py, r12_converge.py, r13_runtime.py   per-cell analysis, convergence, run time
│   ├── r14_fill.py        E2E / scANVI under noise and hidden types
│   ├── setup_ext.py, r16_unsup.py, r16_parallel.py     scMDC and scEMC (authors' code, pinned commits)
│   ├── r17_bmmc.py        the whole protocol on the NeurIPS 2021 BMMC data
│   ├── r18_review.py      tuned baselines, iteration cap, free clusters, sensitivity (App. B.9)
│   └── export_label_draws.py
├── data/              download_github_datasets.py (all data sets except BMMC)
├── label_draws/       <data set>.npz: labelled cells, given labels, corrupted labels, hidden types per run
├── results/
│   ├── per_run/results_tidy.csv   one row per (data set, seed, setting, method): the table behind the paper
│   └── raw/                       the CSV files written by the experiment scripts (see results/README.md)
├── paper/             scripts that rebuild the tables and figures (make_all.py)
└── docs/              pre-registered criteria of the open-world mode (English translation + original)
```

## Installation

Python 3.10 or later. A GPU is not needed; all results in the paper were produced on CPUs.

```bash
git clone https://github.com/pthuan-droid/EvPFC.git
cd EvPFC
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                           # or: conda env create -f environment.yml
```

Rebuilding the tables and figures needs only `numpy scipy pandas scikit-learn matplotlib`. Fig. 6 also needs `torch`.

## 1. Rebuild the tables and figures (from the shipped results)

```bash
python paper/make_all.py               # writes paper/output/tables and paper/output/fig
python paper/make_all.py --rebuild     # first re-assembles results_tidy.csv from results/raw and checks it
```

| Paper | File in `paper/output/` | Script |
|---|---|---|
| Table 3 (ARI, 5 % labels), Table 4 (NMI) | `tables/tab_main.md`, `tables/tab_main_nmi.md` | `stats2.py` |
| Table 2 (summary, hierarchical bootstrap, Holm) | `tables/tab_summary.md`, `tables/summary_dev.csv` | `stats2.py` |
| Table 5 (open-world criteria) | – | `experiments/r8_stats.py` on `results/raw/R8b_conformal.csv` |
| Table 7 (tuned baselines), Table 6 (fixed parameters) | `tables/tab_tuned.md`, `tables/tab_sens.md` | `r18_summary.py` |
| Fig. 1, Fig. 6 | `fig/fig1_evpfc`, `fig/fig2_toy` | `fig_method.py`, `fig_toy.py` |
| Fig. 2, 3, 4, 9 | `fig/fig3_labelfrac`, `fig4_noise`, `fig5_openworld`, `fig6_missing` | `fig_results.py` |
| Fig. 5, 10, 8 | `fig/fig7_ablation`, `fig9_confirmation`, `fig10_runtime` | `fig_extra.py` |
| Fig. 7 | `fig/fig8_cells` | `fig_extra.py`; panels (a–c) need `experiments/r11_cells.py` to be run first |

`--rebuild` reproduces `results_tidy.csv` exactly from `results/raw`. The only exception is 44 rows of an early
ablation pilot that no table or figure uses. Tables 2, 3 and 4 are then identical to those in the paper.

## 2. Re-run the experiments

### Data

All ten data sets are public. They are not redistributed here, so download them into `datasets/` (or set
`EVPFC_DATA=/path/to/datasets`):

```bash
python data/download_github_datasets.py          # SLN111, SLN206, CellMix, ASAP-PBMC, BMNC, SMAGE-3K, Ma-2020, PBMC10k
python experiments/download_round2.py BMMC       # NeurIPS 2021 BMMC CITE-seq + Multiome (GEO GSE194122, several GB)
```

| Name in the code | Paper | Modalities | Source of files and labels |
|---|---|---|---|
| `SLN111`, `SLN206` | SLN111, SLN206 | RNA + protein (counts) | totalVI reproducibility repository (Gayoso et al. 2021) |
| `CellMix` | CellMix | RNA + ATAC (counts) | SNARE-seq cell-line mixture (Chen et al. 2019), via the scCLC / DCCA repositories |
| `ASAP_PBMC` | ASAP-PBMC | RNA/ATAC + protein, mosaic | scMoMaT repository (Mimitou et al. 2021 data) |
| `BMMC_CITE`, `BMMC_Multiome` | BMMC-CITE, BMMC-Multiome | RNA + protein / RNA + ATAC (counts) | GEO GSE194122 (Luecken et al. 2021), site-1 donors s1d1–s1d3 |
| `BMNC` | BMNC | RNA + protein (processed) | scEMC repository (Stuart et al. 2019 data) |
| `SMAGE3K` | SMAGE-3K | RNA + ATAC (processed) | scUNC repository |
| `Ma2020_1`, `PBMC10k_Multiome_feat` | Ma-2020, PBMC10k | RNA + ATAC (processed features) | scMDCL repository |

### Protocol

Every run is defined by a data set, a seed and a setting. All random choices of a run are drawn from
`numpy.random.RandomState(1000 + seed)` in `experiments/r2_core.make_setting`: removed views, hidden types,
labelled cells and corrupted labels. All methods therefore see the same labels. Seeds 3–7 are the development
draws and seeds 8–12 the confirmation draws; seeds 0–2 were used while writing the code and are not reported.
`label_draws/<data set>.npz` lists the resulting cells explicitly (see `label_draws/README.md`).

### Order of the runs

The scripts write CSV files to `experiments/results/` and cache the label-free latent spaces in
`experiments/cache/latents/`. Every script skips finished rows, so it can be stopped and restarted at any time.

| Code | Command (from `experiments/`) | Paper |
|---|---|---|
| R1 | `python r2_run.py --exp R1` (encoders; LS, kNN, eKNN, seeded k-means, scANVI, E2E; 1/5/10 %) | Tables 2–3, Fig. 2 |
| R2 | `python r2_run.py --exp R2 --noises 0.2` | Fig. 3 |
| R3 | `python r2_run.py --exp R3` (two hidden types) | Sec. 6.6 |
| R4 | `python r2_run.py --exp R4` (missing ATAC on CellMix) | Fig. 9 |
| R7 | `python r7_evpfc.py` (EvPFC, EvPFC w/o prior, eKNN, LS on totalVI/MultiVI) | Tables 2–3 |
| R8a | `python r7_evpfc.py --encoders ae --datasets BMNC Ma2020_1 PBMC10k_Multiome_feat SMAGE3K --out R8a_evpfc_ae.csv` | Tables 2–3, Fig. 5e–f |
| R8b | `python r8b_conformal.py` then `python r8_stats.py` | Table 5, Fig. 4 |
| R9 | `python r7_evpfc.py --seeds 8 9 10 11 12 --heads EvPFC EvPFC-noprior eKNN LS kNN --out R9_confirm.csv`; missing views: `--datasets CellMix --settings frac5 --miss 0.3` (0.5, 0.7) `--out R9_missing.csv` | App. B.8, Fig. 10, Fig. 9 |
| R10 | `python r7_evpfc.py --settings frac1 frac5 noise20 hide2 --heads EvPFC-constw EvPFC-nopair EvPFC-norefusal EvPFC-plain --out R10_ablation_count.csv` (and `--encoders ae` on the processed data, `R10_ablation_ae.csv`) | Fig. 5a–d |
| R11–R13 | `python r11_cells.py`, `python r12_converge.py`, `python r13_runtime.py` | Fig. 7, Fig. 8 |
| R14 | `python r14_fill.py --what e2e scanvi`; kNN: `python r7_evpfc.py --heads kNN --out R14_knn_count.csv` (and `--encoders ae`) | Tables 2–3, Fig. 3 |
| R15 | `python r7_evpfc.py --heads SSFCM --out R15_ssfcm.csv` (and `--encoders ae`) | Tables 2–3 |
| R16 | `python setup_ext.py` then `python r16_parallel.py` (scMDC, scEMC) | Table 3 |
| R17 | `python r17_bmmc.py` (all of the above on BMMC) | all tables |
| R18 | `python r18_review.py` (and `--datasets BMMC_CITE BMMC_Multiome`) | Table 7, App. B |

Then copy the CSV files into `results/raw/` (BMMC files into `results/raw/bmmc_machine/`, R18 files into
`results/raw/review_R18/`) and run `python paper/make_all.py --rebuild`.

**Notes.**

- scMDC and scEMC are taken from the authors' repositories at fixed commits (scMDC `0c8778e`, scEMC `7baeffe`). `setup_ext.py` applies two patches to scEMC: CPU execution, and a numerically stable shift (row minimum instead of row mean) in its clustering loss. Both are described in Appendix D.3 of the paper.
- The models trained with stochastic gradients (totalVI, MultiVI, scANVI, E2E, scMDC, scEMC) can differ slightly between machines and library versions. In our runs a repeated E2E fit differed by 0.008 ARI. EvPFC and the classifiers are deterministic given the latent space.
- The runs were made on three CPU machines: a 2-thread cloud server, a 16-thread workstation and a 24-thread workstation for BMMC. Run times in Fig. 8 refer to two CPU threads.

## License and citation

MIT License (see `LICENSE`). If you use the code or the results, please cite the article and this release
(`CITATION.cff`; the Zenodo DOI is given on the release page).
