# Code and methodology review

Cell numbers below use zero-based indexing of the original source notebook (23 cells), which has since been removed from the repository.

## Priority findings

| Priority | Location | Finding and consequence | Action |
|---|---|---|---|
| High | 6, `train` | Scaler fitted to all training samples before five-fold CV; each validation fold influences preprocessing. Holdout scaling itself is correct. | Fixed: pass a scaler/model Pipeline to CV. |
| High | 6, `run_tda_classification_colab` | Chooses algorithm using test accuracy, then returns the final classifier rather than necessarily the best classifier. | Removed flawed example helper from working copy. Use CV on training only for future algorithm selection; reserve test set. |
| High | 6, persistence computation | Bare exception turns every Ripser failure into all-zero topology, silently making failures appear valid. | Fixed: raise contextual error. |
| High | 2, 7, 8 | Newly saved dataset path is ignored; later cells load hard-coded June datasets. Saved generation output refers to an August dataset. Not a coherent top-to-bottom run. | Fixed: train on generated arrays, generate second arrays explicitly with a different seed. |
| Medium | 0 | No seed; default description says uniform but code uses 60% None. Small random datasets can lack rare classes or fail five-fold CV. | Working notebook seeds generation; correct docstring. Small tests use fixed class counts. Validate counts for other experiments. |
| Medium | 6 | Empty image invents a center point; fewer than three points discard all H0 information. | Fixed: explicit empty point cloud and small-sample H0 intervals. |
| Medium | 6 | Docstring claims persistence landscapes, but only diagram and Betti summaries exist. | Corrected documentation. |
| Medium | 9 | `max(10, len(predictions))` prints all samples and indexes past short batches. | Fixed: `min`. |
| Medium | 13, 15–18 | Raw-feature UMAP/PCA/K-means are sensitive to mixed units; silhouette space differs from clustering space. | Working copy scales features and computes silhouette in clustering PCA space. |
| Medium | 21 | `projection=[0,1]` selects two features rather than PCA components. The second is H0 lifetime mean, not PC2. | Fixed: actual PCA object. |
| Medium | 21 | K-means requests nine clusters within every cover region, a brittle choice for small regions. | Changed to DBSCAN; tune eps/min_samples and inspect node/noise coverage. |
| Medium | 0, 6 | Shape/NaN/negative checks, distribution checks, and dataset validation are incomplete. | Added feature-input and basic generation checks. Dataset-level count/shape validation remains a next step. |
| Low | 0 | One-sample plot can fail because axes is scalar; zero samples divide by zero. | Fixed: squeeze=False and sample-count guard. |
| Low | 6 | Warnings suppressed globally; unused Gudhi and scikit-image imports add requirements without computation. | Removed global warning suppression and unused imports in working copy. |
| Low | 6 | `allow_pickle=True` unnecessary for the generator's string-label arrays. | Changed to False; legacy object arrays need deliberate conversion. |
| Low | 22 | Comments claim current directory despite explicit Drive output; absolute second operand discards first path in join. | Fixed: print resolved local HTML path. |

## Scientific limitations retained deliberately

- Raw pixel coordinates, fixed threshold 0.1, top-500 intensity sampling, and fixed Betti range 0–2 require sensitivity analysis. At 26 × 26, 0–2 pixels is a local range. Changing image resolution changes the meaning of these distances.
- Defect density uses the full rectangular image denominator, including pixels outside the wafer; compare on-wafer normalization.
- Uniform noise above the threshold can give None maps nontrivial point clouds. Stress-test threshold sensitivity rather than assuming None always has empty topology.
- Center/Donut/Edge-Ring intensity plus positive noise can exceed 1. Either document this scale or consistently clip and recalibrate thresholds. The reviewed copy preserves these values.
- Generator scratches use angles 0–pi, advance in positive step direction, and can clip at boundaries. Validate orientation/location diversity.
- Higher-dimensional Rips features do not imply a physical 3D structure. H0/H1 are a more economical starting experiment for these 2D patterns; the working copy changes the demonstration default, so its scores are not directly comparable to the original 47-feature run.
- Betti sum is exactly 20 times Betti mean; these are redundant features at a fixed 20-point grid. Assess correlations and feature ablations.
- No landscape features, learned filtration, graph ontology, root-cause inference, or causal inference is implemented.
- The same synthetic rules generate both datasets. Validate with varied simulator parameters and measured fab data, with lot/tool/time grouping to prevent correlated samples crossing splits.
- Report macro-F1, balanced accuracy, per-class recall, and an always-None baseline. Accuracy alone can hide failures in rare defects.
- Feature extraction is deterministic per sample and does not fit global statistics; extracting these fixed features before the split is not itself leakage. Preprocessing fitted across samples is the relevant CV issue here.
- The method combines topology and geometry. Compare geometry-only (last 8 features), topology-only (all preceding features), and combined models using identical splits. Tune within training data only.
- UMAP/t-SNE plots are exploratory. K=9 assumes category count; cluster IDs are arbitrary. ARI/NMI accept string labels directly; LabelEncoder is unnecessary for those metrics.
- Fit PCA/UMAP once on a reference dataset if comparing positions across datasets; separate fits do not share coordinates. t-SNE generally does not provide a standard transform for new samples.
- Dependency ranges are not locked. The original logs include mixed Python/runtime versions; archive a tested environment before benchmark publication.

## Patch boundaries

The original notebook has been removed from the repository. The reviewed notebook has its outputs cleared, and reusable classes are extracted into Python modules. It has targeted fixes rather than a claim of production readiness. Its original default 1,000-sample imbalanced experiment can still fail stratification/CV on unusual draws; use explicit balanced counts or check per-class minima. The full notebook, including its embeddings and Mapper graph, has since been run end to end; see VALIDATION.md.
