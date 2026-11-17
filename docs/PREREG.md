# Pre-registered criteria for steps 2a and 2b

English translation of `PREREG_original_vi.md`, which was written on 2026-10-15, **before** the runs it describes.
The original Vietnamese text is kept unchanged in this folder. In the code and the original, "scPIKA with its own
encoder" is the end-to-end variant called E2E in the paper.

## 2a. EvPFC on the latent space of the label-free autoencoder (encoder `ae`)

- Latent space: pre-training stage of the scPIKA autoencoder, 30 epochs, reconstruction loss only, no labels. Seeds 3–7.
- Data: BMNC, Ma2020_1, PBMC10k_Multiome_feat, SMAGE3K (processed), and SLN111, SLN206, CellMix, ASAP_PBMC (counts).
- Compared with: EvPFC on PCA (R7), scPIKA with its own encoder (R1, same seeds and labels), eKNN and LS on the same latent space.

| Code | Criterion | Threshold |
|---|---|---|
| A1 | Mean ARI over the 4 processed data sets, 5 % labels: EvPFC-ae vs scPIKA (own encoder) | EvPFC-ae ≥ scPIKA − 0.01 |
| A2 | As A1, vs EvPFC on PCA | EvPFC-ae > EvPFC-PCA |
| A3 | 20 % noisy labels: EvPFC-ae vs scPIKA (own encoder) on the data sets present in R2 | EvPFC-ae > scPIKA |

## 2b. Free clusters with a conformal threshold

**Mechanism:**
1. Hold out 20 % of the labelled cells as a calibration set.
2. Build the evidential k-NN knowledge from the remaining 80 % of the labels only.
3. The threshold q is the conformal 1 − α quantile (α = 0.05) of m(Θ) on the calibration set.
4. Unlabelled cells with m(Θ) > q are declared "novel" and partitioned into the K − T free clusters by picture fuzzy clustering without prior.
5. All other cells are assigned by EvPFC over the T known clusters, using all labels.

Because labels are drawn at random, calibration cells are exchangeable with the unlabelled cells of known types,
so the expected share of known-type cells sent to the free clusters is ≤ α.

**Evaluation:** two hidden types, 5 % labels, **seeds 8–12**. These are new label draws and hidden types, not used in
any earlier step. The latent spaces of seeds 3–7 are reused (they are label-free). Data: data sets with ≥ 6 cell types.

| Code | Criterion | Threshold |
|---|---|---|
| B1 | Share of hidden-type cells sent to free clusters (mean over data sets) | > 30 % |
| B2 | ARI relative to EvPFC without the conformal threshold | ≥ EvPFC − 0.01 |
| B3 | Share of known-type cells sent to free clusters (mean) | ≤ 0.07 (α + tolerance) |
| B4 | Mandatory report (not a criterion): ARI with the mechanism switched on but **no** novel type (5 % labels, K = T + 2), i.e. the cost of false alarms | — |

**Addendum (written before running 2b):**
- Latent spaces: totalVI for SLN111 and SLN206. For the processed data sets, run and report **both** PCA and `ae`.
- Criteria B1–B3 are computed on the **mean over all (data set, latent space) pairs**.

Outcome (paper, Table 5): B3 met; B1 (27.5 %) and B2 (0.688 vs 0.700) not met.
