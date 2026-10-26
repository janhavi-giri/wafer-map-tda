# Harder synthetic data: where raw-pixel clustering breaks, and when topology helps

`SyntheticWaferMapGenerator(variation=...)` adds realistic variation to every defect type. The default (`None`) keeps the original idealized maps, so earlier results are unchanged.

| Knob | Effect |
|---|---|
| `noise_sigma` | additive Gaussian measurement noise |
| `speckle` | probability that a pixel becomes an isolated false defect |
| `size_jitter` | pattern scaled by exp(U(-j, j)) |
| `position_jitter` | pattern shifted by up to p pixels |
| `stretch` | anisotropic stretch along a random direction (axis ratio up to exp(k)) |
| `rotation` | random rotation, up to this many degrees |

Presets are `"clean"`, `"mild"`, `"moderate"` and `"hard"`. You can also pass a dict with any of the keys above.

![Each defect type at each variation level](variation_levels.png)

## Experiment

`scripts/variation_comparison.py` generates balanced datasets (60 maps per class, 540 per dataset), three seeds (42, 7, 123), and compares four feature sets. Values are means over the three seeds. Clustering uses k = 9, the known class count.

| Metric | Features | clean | mild | moderate | hard |
|---|---|---|---|---|---|
| k-means ARI | Raw pixels (PCA-20) | 0.46 | 0.30 | 0.19 | 0.17 |
| | Geometric (8) | 0.55 | 0.48 | 0.42 | 0.37 |
| | Topology (26) | 0.58 | 0.52 | 0.40 | 0.28 |
| | Topology + geometry (34) | 0.68 | 0.58 | 0.53 | 0.35 |
| RF CV accuracy | Raw pixels (PCA-20) | 0.93 | 0.90 | 0.82 | 0.75 |
| | Geometric (8) | 0.94 | 0.87 | 0.79 | 0.72 |
| | Topology (26) | 0.92 | 0.84 | 0.73 | 0.63 |
| | Topology + geometry (34) | 0.96 | 0.92 | 0.85 | 0.79 |
| Train on clean, test on level | Geometric (8) | 1.00 | 0.75 | 0.39 | 0.35 |
| | Topology + geometry (34) | 1.00 | 0.58 | 0.30 | 0.26 |

## What the results do and do not show

- Clustering on raw pixels fails as variation grows: ARI falls from 0.46 to 0.17.
- Topology plus geometry clusters best on clean, mild and moderate data, and degrades more slowly than raw pixels at mild and moderate levels.
- At the hard level the advantage is gone: topology plus geometry (0.35) is no better than geometry alone (0.37).
- Topology features alone are not more robust than the simple geometric ones, and degrade fastest in the supervised settings.
- Models trained on clean maps and tested on varied maps collapse for every feature set. Topology features do not protect against this shift.

The topology features used here are 26 summary statistics from a fixed pipeline (intensity threshold 0.1, filtration range 0 to 2 pixels). Speckle noise adds points above the threshold, which is a likely reason topology degrades at high noise. A noise-aware threshold or a different representation (persistence images, diagram distances) has not been tested here, and the pipeline was not tuned on these datasets.

Only three seeds were run, and these are synthetic maps from one generator. Across seeds the standard deviation is typically 0.02 to 0.03 and up to 0.06, so treat differences of a few hundredths as noise.

## Follow-up: a noise-aware point-cloud threshold

The speckle noise explanation above was tested, with the protocol fixed in advance:

1. **Dev seeds (42, 7).** On the mild, moderate and hard levels, try point-cloud thresholds 0.1 (the default), 0.2, 0.3, 0.4 and 0.5, and pick the one with the best topology-only k-means ARI. Mean ARI was 0.39, 0.42, **0.42**, 0.38 and 0.30 respectively, so 0.3 was chosen (0.2 was almost tied).
2. **Held-out seeds (2001, 2002, 2003), evaluated once.** Newly generated datasets, same size and balance. Nothing was changed after seeing these results.

`TDAFeatureExtractor(point_cloud_threshold=0.3)` sets the threshold; the default is still 0.1. A second representation, smoothed lifetime profiles of the H0 and H1 diagrams (18 features), was also tried.

Held-out means over three seeds:

| Metric | Features | clean | mild | moderate | hard |
|---|---|---|---|---|---|
| k-means ARI | Raw pixels | 0.43 | 0.31 | 0.21 | 0.17 |
| | Geometric (8) | 0.54 | 0.45 | 0.44 | 0.36 |
| | Topology, threshold 0.1 (26) | 0.58 | 0.50 | 0.45 | 0.27 |
| | Topology, threshold 0.3 (26) | 0.65 | 0.48 | 0.42 | 0.36 |
| | Topology 0.1 + geometry (34) | 0.68 | 0.56 | 0.51 | 0.34 |
| | Topology 0.3 + geometry (34) | 0.70 | 0.59 | 0.51 | 0.42 |
| | Lifetime profiles + geometry (26) | 0.51 | 0.45 | 0.40 | 0.30 |
| RF CV accuracy | Geometric (8) | 0.95 | 0.87 | 0.82 | 0.73 |
| | Topology 0.3 + geometry (34) | 0.95 | 0.90 | 0.85 | 0.80 |
| Train clean, test on level | Geometric (8) | 1.00 | 0.69 | 0.38 | 0.32 |
| | Raw pixels | 1.00 | 0.50 | 0.41 | 0.37 |
| | Topology 0.3 + geometry (34) | 1.00 | 0.69 | 0.50 | 0.43 |

Ward linkage ARI follows the same pattern (hard level: 0.39 for topology 0.3 + geometry, 0.38 for geometry alone, 0.14 for raw pixels).

### Conclusions

- **Supported:** with a noise-aware threshold, topology plus geometry is the best or tied-best feature set at every variation level for clustering, supervised cross-validation, and training on clean maps and testing on varied ones. At the hard level it beats geometry alone by about 0.06 ARI, 0.07 accuracy in cross-validation and 0.1 in the clean-to-varied test, and beats raw pixels by 0.25 ARI.
- **Not supported:** topology on its own is no better than the 8 geometric features once the threshold is tuned (ARI 0.48, 0.42, 0.36 against 0.45, 0.44, 0.36). The benefit comes from combining them.
- **Not supported:** the lifetime-profile representation did worse than summary statistics here.
- **Sensitivity:** the threshold matters. With the default 0.1, topology plus geometry is no better than geometry at the hard level, and at 0.5 topology degrades (dev ARI 0.30).
- **Limits:** all data comes from one synthetic generator, with three held-out seeds and seed standard deviations of about 0.02 to 0.06, so gaps of about 0.05 are suggestive and not conclusive. The threshold was tuned on the same generator it is evaluated on. Real WM811K maps have different noise, so the threshold would need retuning there.

Reproduce with `python scripts/topology_variants.py OUTPUT_DIR`.
