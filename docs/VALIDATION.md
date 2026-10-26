# Validation performed

## End-to-end run on a fresh install (2026-10-22)

- Created a new virtual environment and ran `pip install -e '.[notebook]'`: Python 3.13.16, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, ripser 0.6.15, persim 0.4.1, umap-learn 0.5.12, kmapper 2.1.0, numba 0.68.0, jupyterlab 4.6.4.
- All six unit tests passed (about 0.6 s).
- `notebooks/wafer_map_tda_reviewed.ipynb` ran top to bottom with no errors in about 47 seconds, including UMAP, t-SNE, PCA/k-means and the Mapper export.
  - 34 features; held-out split accuracy 0.9750; 5-fold CV accuracy 0.9650 ± 0.0140; second synthetic dataset accuracy 0.9780.
  - Silhouette 0.4671, ARI 0.3749, NMI 0.6443.
- `notebooks/wafer_tda_mapper_eps2_all_classes.ipynb` ran with no errors in about 13 seconds: 53 nodes, 95 edges, 944 of 1,000 maps covered.
- Only Python 3.13 was tested. The CI workflow is set up for Python 3.10 to 3.13 but had not been run when this note was written.

## Earlier checks

- The small balanced 45-map synthetic smoke test returned 100% holdout and CV accuracy. It checks execution only and is not a benchmark or evidence of real-fab performance.
- Original archive content was compared structurally with the source notebook.
- Google Colab and Google Drive behavior was not tested. Historical metrics from the original source notebook (no longer in the repository) were not reproduced; the reviewed notebook's results differ because of the fixes listed in `docs/REVIEW.md`.

These are a tested environment snapshot, not a dependency lock.
