# Results

## `per_run/results_tidy.csv`

One row per run: (data set, seed, setting, method). Every table and figure of the paper is computed from this file,
together with the raw files listed below for the ablation, per-cell, convergence and run-time figures.

| Column | Meaning |
|---|---|
| `ds`, `seed` | data set (name in the code) and seed; seeds 3–7 = development (`phase = dev`), 8–12 = confirmation |
| `setting` | `frac1`, `frac5`, `frac10` (labelled fraction), `noise20` (5 % labels, 20 % corrupted), `hide2` (5 % labels, two types hidden), `miss30/50/70` (CellMix, ATAC removed) |
| `method` | name used in the paper (EvPFC, EvPFC w/o prior, eKNN, LS, kNN, SSFCM, E2E, scANVI, Seeded k-means, scMDC, scEMC, EvPFC-constw/-nopair/-norefusal/-plain) |
| `encoder` | latent space: `totalvi`, `multivi`, `ae` (label-free autoencoder), `scvi` (scANVI), `pca`, `own` (E2E, scMDC, scEMC) |
| `ARI`, `NMI`, `ACC`, `F1` | scores on the cells that did not receive a label |
| `n_cells`, `K`, `T`, `n_labelled` | cells, clusters, known types, labelled cells |
| `hidden_to_free`, `known_to_free`, `auroc_novel`, `ACC_known_cells` | hidden-type runs: shares of cells sent to free clusters, AUROC of the novelty score, accuracy on known types |
| `beta`, `tau_scale`, `n_iter`, `mean_xi`, `mean_eta` | EvPFC: selected parameters, iterations, mean refusal and neutral masses |
| `src` | raw file the row comes from (R1 … R17) |
| `extra` | JSON with any further fields written by the script |

## `raw/`

CSV files as written by the scripts in `experiments/` (same columns as the first 24 of `results_tidy.csv`).

| File | Script | Content |
|---|---|---|
| `R1_main.csv` | `r2_run.py --exp R1` | encoders + LS, kNN, eKNN, seeded k-means, scANVI, E2E; 1/5/10 % labels |
| `R2_noise.csv` | `r2_run.py --exp R2` | 20 % corrupted labels |
| `R3_novel.csv` | `r2_run.py --exp R3` | two hidden types |
| `R4_missing.csv` | `r2_run.py --exp R4` | missing ATAC view (CellMix) |
| `R7_evpfc.csv` | `r7_evpfc.py` | EvPFC, EvPFC w/o prior, eKNN, LS on totalVI/MultiVI (and PCA) |
| `R8a_evpfc_ae.csv` | `r7_evpfc.py --encoders ae` | same on the label-free autoencoder latent |
| `R8b_conformal.csv` | `r8b_conformal.py` | open-world mode, confirmation seeds 8–12 |
| `R9_confirm.csv`, `R9_missing.csv` | `r7_evpfc.py` | confirmation seeds 8–12; missing views. Re-exported from `results_tidy.csv`, rows identical |
| `R10_ablation_count.csv`, `R10_ablation_ae.csv` | `r7_evpfc.py` | ablation of EvPFC |
| `R11_bins.csv`, `R11_bins_count.csv` | `r11_cells.py` | accuracy per quintile of the ignorance (Fig. 7) |
| `R12_converge.csv`, `R13_runtime.csv` | `r12_converge.py`, `r13_runtime.py` | iteration cap; run time (Fig. 8) |
| `R14_fill.csv`, `R14_knn_*.csv` | `r14_fill.py`, `r7_evpfc.py --heads kNN` | E2E / scANVI under noise and hidden types; kNN |
| `R15_ssfcm.csv` | `r7_evpfc.py --heads SSFCM` | semi-supervised fuzzy c-means |
| `R16_unsup.csv` | `r16_parallel.py` | scMDC and scEMC (`extra`: number of predicted clusters, size of the largest) |
| `bmmc_machine/*` | `r17_bmmc.py` | BMMC-CITE and BMMC-Multiome, all of the above (run on a second workstation) |
| `review_R18/*` | `r18_review.py` | tuned baselines, iteration cap, number of free clusters, fixed parameters |

The R1–R3 and R16 files in `bmmc_machine/` also contain rows for the other data sets, re-run on that machine; the
paper uses them only for the two BMMC data sets (`paper/collect_bmmc.py`).
