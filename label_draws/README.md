# Label draws

`<data set>.npz` makes explicit the labelled cells of every run. The draws are produced by
`experiments/r2_core.make_setting(y, seed, frac, noise, hide, miss)` from `numpy.random.RandomState(1000 + seed)`,
and `experiments/export_label_draws.py` writes these files. Cell indices refer to the cell order produced by the
loaders in `scpika/loaders.py` and `experiments/r2_data.py`, after the filtering described in the paper.

Keys, for seed `s` (3–12) and setting `t` (`frac1`, `frac5`, `frac10`, `noise20`, `hide2`, and `miss30/50/70` on CellMix):

| Key | Content |
|---|---|
| `y` | annotation used (integer codes), one per cell |
| `s{s}_{t}_labelled` | indices of the labelled cells |
| `s{s}_{t}_given` | label given to each of them (code in `y`); differs from `y` for corrupted labels |
| `s{s}_{t}_corrupted` | indices of the labelled cells whose label was corrupted (`noise20`) |
| `s{s}_{t}_hidden` | codes of the two hidden types (`hide2`) |
| `s{s}_{t}_view_removed` | cells whose ATAC view was removed (CellMix, `miss*`) |

```python
import numpy as np
z = np.load('label_draws/SLN111.npz')
lab, given = z['s3_frac5_labelled'], z['s3_frac5_given']
```

The files for BMMC-CITE and BMMC-Multiome are written by
`python experiments/export_label_draws.py --datasets BMMC_CITE BMMC_Multiome` once the BMMC files have been downloaded.
