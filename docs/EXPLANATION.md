# What the notebook does

The core question is: can a compact description of spatial structure classify wafer defect patterns?

## 1. Synthetic data

`SyntheticWaferMapGenerator` draws patterns on a 26 × 26 grid and masks pixels outside a disk. Center is a filled central spot; Donut is an interior annulus; Edge-Ring is a thin perimeter band; Edge-Loc and Loc are randomly placed spots; Scratch draws short thick lines; Random scatters pixels; Near-full smooths a high-coverage map; None adds weak measurement-like noise.

These are continuous synthetic intensities, not a direct implementation of a measured wafer die-state encoding. The default class distribution puts 60% probability on None and 2–9% on each defect category. Pattern parameters largely stay fixed across examples, making the classification problem easier than a varied manufacturing distribution. There is no empirical calibration to WM811K in this notebook.

## 2. From image to topology

Pixels above 0.1 become points. At most 500 are kept, ranked by intensity. Ripser receives only their coordinates, so intensity affects selection but is not a filtration value.

As a distance threshold grows, points connect and higher simplices fill in. H0 records components joining; H1 records loops appearing and filling; H2 records higher-dimensional cavities in the Rips complex. H2 here should not be interpreted as a physical 3D wafer cavity. Increasing the maximum dimension increases cost substantially.

For each dimension the code computes nine diagram statistics: interval count, lifetime mean/std/max/sum, birth mean/std, and death mean/std. Infinite intervals contribute to interval counts but not finite lifetime/death summaries. Four more features summarize the Betti curve at 20 thresholds from 0 to 2. Because coordinates are in pixel units, those thresholds probe local scales rather than an entire wafer diameter.

Eight additional features record image mean/std/max/min, thresholded pixel density, x/y centroid, and centroid distance from the center. This yields `13 × (max_dimension + 1) + 8` features: 47 at dimension 2 and 34 at dimension 1.

Topology based on distances cannot distinguish identical point patterns translated to different wafer locations. The geometric features supply location information, which matters for center-versus-edge defects. An ablation study is therefore necessary to attribute performance to topology.

## 3. Supervised learning

A stratified 80/20 train/test split keeps approximate class proportions. Labels are encoded, features are standardized, and a conventional classifier predicts the nine categories. The notebook supports random forest, SVM, gradient boosting, and logistic regression. Prediction probabilities are unavailable for its default SVM configuration.

In the original code, scaling is correct for the holdout split but occurs before five-fold cross-validation on the training partition. That makes CV less trustworthy. The reviewed copy uses a Pipeline so scaling is fitted independently inside every fold.

A second synthetic dataset is loaded for predictions and a confusion matrix. Independent generation from the same rule set tests repeatability within that simulator; it is not evidence of transfer to real manufacturing.

## 4. Exploratory analysis

UMAP and t-SNE display local relationships in two dimensions. PCA summarizes variance along linear axes. K-means partitions a two-component PCA representation into nine clusters, explicitly using the known category count as a prior. The clusters need not match defect labels.

Silhouette measures cohesion/separation in a chosen distance space; ARI and NMI compare cluster assignments to labels. The original saved values are 0.5267, 0.4844, and 0.6279. The silhouette was computed in the original feature space even though clustering used PCA space, so describe that distinction when quoting it.

Mapper forms overlapping regions in a lens space, clusters samples within each region, and links nodes sharing samples. Its output is an exploratory graph, not another classifier. The original comment calls the lens PCA, but the code selects feature columns 0 and 1. The reviewed version computes an actual PCA lens and uses DBSCAN within cover regions; epsilon remains a parameter requiring tuning.

## Assessment

Useful as a teaching and research prototype connecting synthetic wafer patterns to persistent homology. The main next experiment is to measure topology's added value over intensity and geometry under controlled, out-of-distribution evaluation. Neither a visually separated embedding nor high synthetic accuracy establishes physical root-cause discovery.
