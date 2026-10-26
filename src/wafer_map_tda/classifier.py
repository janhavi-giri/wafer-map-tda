import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import os
from typing import List, Tuple, Dict, Any


import ripser

class TDAFeatureExtractor:
    """
    Topological Data Analysis feature extractor for wafer maps.

    This class extracts topological features from wafer maps including:
    - Persistent homology features (H0, H1, H2)
    - Betti numbers
    - Geometric and intensity features (no persistence landscapes)
    - Statistical summaries of persistence diagrams
    """

    def __init__(self, max_dimension: int = 2, max_edge_length: float = np.inf,
                 point_cloud_threshold: float = 0.1):
        """
        Initialize TDA feature extractor.

        Args:
            max_dimension: Maximum homology dimension to compute
            max_edge_length: Maximum edge length for Rips complex
            point_cloud_threshold: Pixels with intensity above this value become
                points. The default 0.1 reproduces the original behavior; a
                higher value (0.3 worked better on noisy synthetic maps, see
                docs/VARIATION.md) rejects faint noise and speckle.
        """
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.point_cloud_threshold = point_cloud_threshold
        self.feature_names = []

    def _extract_point_cloud(self, wafer_map: np.ndarray,
                           threshold: float = 0.1,
                           max_points: int = 500) -> np.ndarray:
        """
        Extract point cloud from wafer map for TDA analysis.

        Args:
            wafer_map: 2D wafer map array
            threshold: Minimum intensity threshold for points
            max_points: Maximum number of points to extract

        Returns:
            Point cloud as Nx2 array
        """
        # Find points above threshold
        y_coords, x_coords = np.where(wafer_map > threshold)
        intensities = wafer_map[y_coords, x_coords]

        if len(x_coords) == 0:
            return np.empty((0, 2))

        # Sort by intensity and take top points
        sorted_indices = np.argsort(intensities)[::-1]
        n_points = min(max_points, len(sorted_indices))

        selected_indices = sorted_indices[:n_points]
        points = np.column_stack((x_coords[selected_indices], y_coords[selected_indices]))

        return points

    def _compute_persistence_diagram(self, points: np.ndarray) -> Dict:
        """
        Compute persistence diagram using Ripser.

        Args:
            points: Point cloud as Nx2 array

        Returns:
            Dictionary with persistence diagrams for each dimension
        """
        if len(points) < 3:
            # Explicit H0 intervals for zero, one, or two points.
            diagrams = {i: np.empty((0, 2)) for i in range(self.max_dimension + 1)}
            if len(points):
                h0 = [[0.0, np.inf]]
                if len(points) == 2:
                    distance = float(np.linalg.norm(points[0] - points[1]))
                    h0.insert(0, [0.0, distance if distance <= self.max_edge_length else np.inf])
                diagrams[0] = np.array(h0)
            return diagrams

        try:
            # Compute persistence diagram
            diagrams = ripser.ripser(points, maxdim=self.max_dimension,
                                   thresh=self.max_edge_length)['dgms']
            return {i: diagrams[i] for i in range(len(diagrams))}
        except Exception as exc:
            raise RuntimeError('Persistence calculation failed') from exc

    def _compute_betti_numbers(self, diagrams: Dict,
                              filtration_values: np.ndarray) -> np.ndarray:
        """
        Compute Betti numbers across filtration values.

        Args:
            diagrams: Persistence diagrams
            filtration_values: Values at which to compute Betti numbers

        Returns:
            Betti numbers array of shape (n_filtrations, n_dimensions)
        """
        betti_curves = []

        for dim in range(self.max_dimension + 1):
            if dim not in diagrams or len(diagrams[dim]) == 0:
                betti_curves.append(np.zeros(len(filtration_values)))
                continue

            betti_curve = np.zeros(len(filtration_values))
            for i, t in enumerate(filtration_values):
                # Count intervals that contain t
                diagram = diagrams[dim]
                if len(diagram) > 0:
                    births = diagram[:, 0]
                    deaths = diagram[:, 1]
                    # Handle infinite deaths
                    deaths = np.where(np.isinf(deaths), t + 1, deaths)
                    betti_curve[i] = np.sum((births <= t) & (deaths > t))

            betti_curves.append(betti_curve)

        return np.array(betti_curves).T

    def _persistence_statistics(self, diagram: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical features from persistence diagram.

        Args:
            diagram: Persistence diagram as Nx2 array

        Returns:
            Dictionary of statistical features
        """
        if len(diagram) == 0:
            return {
                'n_points': 0,
                'lifetime_mean': 0,
                'lifetime_std': 0,
                'lifetime_max': 0,
                'lifetime_sum': 0,
                'birth_mean': 0,
                'birth_std': 0,
                'death_mean': 0,
                'death_std': 0
            }

        # Filter out infinite deaths
        finite_mask = ~np.isinf(diagram[:, 1])
        if np.sum(finite_mask) == 0:
            diagram_finite = np.array([]).reshape(0, 2)
        else:
            diagram_finite = diagram[finite_mask]

        if len(diagram_finite) == 0:
            lifetimes = np.array([])
            births = diagram[:, 0]
            deaths = np.array([])
        else:
            lifetimes = diagram_finite[:, 1] - diagram_finite[:, 0]
            births = diagram_finite[:, 0]
            deaths = diagram_finite[:, 1]

        stats = {
            'n_points': len(diagram),
            'lifetime_mean': np.mean(lifetimes) if len(lifetimes) > 0 else 0,
            'lifetime_std': np.std(lifetimes) if len(lifetimes) > 0 else 0,
            'lifetime_max': np.max(lifetimes) if len(lifetimes) > 0 else 0,
            'lifetime_sum': np.sum(lifetimes) if len(lifetimes) > 0 else 0,
            'birth_mean': np.mean(births) if len(births) > 0 else 0,
            'birth_std': np.std(births) if len(births) > 0 else 0,
            'death_mean': np.mean(deaths) if len(deaths) > 0 else 0,
            'death_std': np.std(deaths) if len(deaths) > 0 else 0
        }

        return stats

    def extract_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """
        Extract TDA features from a single wafer map.

        Args:
            wafer_map: 2D wafer map array

        Returns:
            1D array of TDA features
        """
        wafer_map = np.asarray(wafer_map, dtype=float)
        if wafer_map.ndim != 2 or not wafer_map.size or not np.isfinite(wafer_map).all() or np.any(wafer_map < 0):
            raise ValueError('wafer_map must be a nonempty finite nonnegative 2D array')
        features = []

        # Extract point cloud
        points = self._extract_point_cloud(wafer_map, threshold=self.point_cloud_threshold)

        # Compute persistence diagrams
        diagrams = self._compute_persistence_diagram(points)

        # Extract statistical features from each dimension
        for dim in range(self.max_dimension + 1):
            diagram = diagrams.get(dim, np.array([]).reshape(0, 2))
            stats = self._persistence_statistics(diagram)

            for key, value in stats.items():
                features.append(value)

        # Compute Betti curves
        filtration_values = np.linspace(0, 2, 20)  # Adjust range as needed
        betti_numbers = self._compute_betti_numbers(diagrams, filtration_values)

        # Add Betti curve statistics
        for dim in range(self.max_dimension + 1):
            betti_curve = betti_numbers[:, dim]
            features.extend([
                np.mean(betti_curve),
                np.std(betti_curve),
                np.max(betti_curve),
                np.sum(betti_curve)
            ])

        # Add geometric features
        geometric_features = self._extract_geometric_features(wafer_map)
        features.extend(geometric_features)

        return np.array(features)

    def _extract_geometric_features(self, wafer_map: np.ndarray) -> List[float]:
        """
        Extract additional geometric features from wafer map.

        Args:
            wafer_map: 2D wafer map array

        Returns:
            List of geometric features
        """
        features = []

        # Basic statistics
        features.extend([
            np.mean(wafer_map),
            np.std(wafer_map),
            np.max(wafer_map),
            np.min(wafer_map),
            np.sum(wafer_map > 0.1) / wafer_map.size,  # Density of defects
        ])

        # Spatial moments
        y_coords, x_coords = np.meshgrid(range(wafer_map.shape[0]),
                                        range(wafer_map.shape[1]), indexing='ij')

        if np.sum(wafer_map) > 0:
            # Centroid
            total_intensity = np.sum(wafer_map)
            centroid_y = np.sum(y_coords * wafer_map) / total_intensity
            centroid_x = np.sum(x_coords * wafer_map) / total_intensity

            # Distance from center
            center_y, center_x = wafer_map.shape[0] // 2, wafer_map.shape[1] // 2
            centroid_distance = np.sqrt((centroid_y - center_y)**2 + (centroid_x - center_x)**2)

            features.extend([centroid_y, centroid_x, centroid_distance])
        else:
            features.extend([0, 0, 0])

        return features

    def fit_transform(self, wafer_maps: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Extract TDA features from multiple wafer maps.

        Args:
            wafer_maps: Array of wafer maps with shape (n_samples, height, width)

        Returns:
            Tuple of (features_array, feature_names)
        """
        print(f"Extracting TDA features from {len(wafer_maps)} wafer maps...")

        all_features = []
        for i, wafer_map in enumerate(wafer_maps):
            features = self.extract_features(wafer_map)
            all_features.append(features)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(wafer_maps)} maps...")

        # Generate feature names
        if not self.feature_names:
            self._generate_feature_names()

        return np.array(all_features), self.feature_names

    def _generate_feature_names(self):
        """Generate descriptive feature names."""
        self.feature_names = []

        # Persistence statistics for each dimension
        stat_names = ['n_points', 'lifetime_mean', 'lifetime_std', 'lifetime_max',
                     'lifetime_sum', 'birth_mean', 'birth_std', 'death_mean', 'death_std']

        for dim in range(self.max_dimension + 1):
            for stat in stat_names:
                self.feature_names.append(f'H{dim}_{stat}')

        # Betti curve statistics
        betti_stats = ['mean', 'std', 'max', 'sum']
        for dim in range(self.max_dimension + 1):
            for stat in betti_stats:
                self.feature_names.append(f'betti_{dim}_{stat}')

        # Geometric features
        geometric_names = ['intensity_mean', 'intensity_std', 'intensity_max',
                          'intensity_min', 'defect_density', 'centroid_y',
                          'centroid_x', 'centroid_distance']
        self.feature_names.extend(geometric_names)

class WaferDefectClassifier:
    """
    Supervised classifier for wafer defect detection using TDA features.
    """

    def __init__(self, tda_extractor: TDAFeatureExtractor = None):
        """
        Initialize the classifier.

        Args:
            tda_extractor: TDA feature extractor instance
        """
        self.tda_extractor = tda_extractor or TDAFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classifier = None
        self.feature_names = []
        self.is_fitted = False

    def load_dataset_from_colab(self, dataset_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load wafer map dataset from Google Colab directory structure.

        Args:
            dataset_path: Path to the dataset folder in Colab

        Returns:
            Tuple of (wafer_maps, labels)
        """
        print(f"Loading dataset from: {dataset_path}")

        # Try to load combined files first
        combined_maps_path = os.path.join(dataset_path, 'all_wafer_maps.npy')
        combined_labels_path = os.path.join(dataset_path, 'all_labels.npy')

        if os.path.exists(combined_maps_path) and os.path.exists(combined_labels_path):
            wafer_maps = np.load(combined_maps_path)
            labels = np.load(combined_labels_path, allow_pickle=False).tolist()
            print(f"Loaded {len(wafer_maps)} samples from combined files")
            return wafer_maps, labels

        # Load from category folders
        wafer_maps = []
        labels = []

        for category_name in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category_name)
            if not os.path.isdir(category_path):
                continue

            arrays_path = os.path.join(category_path, 'arrays')
            if os.path.exists(arrays_path):
                category_maps = []
                for filename in sorted(os.listdir(arrays_path)):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(arrays_path, filename)
                        wafer_map = np.load(file_path)
                        category_maps.append(wafer_map)

                wafer_maps.extend(category_maps)
                labels.extend([category_name] * len(category_maps))
                print(f"Loaded {len(category_maps)} samples from category: {category_name}")

        print(f"Total loaded: {len(wafer_maps)} samples")
        return np.array(wafer_maps), labels

    def train(self, wafer_maps: np.ndarray, labels: List[str],
              classifier_type: str = 'random_forest',
              test_size: float = 0.2,
              random_state: int = 42) -> Dict[str, Any]:
        """
        Train the classifier on wafer map data.

        Args:
            wafer_maps: Array of wafer maps
            labels: Corresponding labels
            classifier_type: Type of classifier ('random_forest', 'svm', 'gradient_boost', 'logistic')
            test_size: Fraction of data for testing
            random_state: Random state for reproducibility

        Returns:
            Dictionary with training results
        """
        print("Starting training process...")

        # Extract TDA features
        X, self.feature_names = self.tda_extractor.fit_transform(wafer_maps)
        print(f"Extracted {X.shape[1]} TDA features")

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scaling is fitted inside each cross-validation fold using a Pipeline.

        # Select classifier
        classifiers = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'svm': SVC(kernel='rbf', random_state=random_state),
            'gradient_boost': GradientBoostingClassifier(random_state=random_state),
            'logistic': LogisticRegression(random_state=random_state, max_iter=1000)
        }

        if classifier_type not in classifiers:
            raise ValueError(f'Unknown classifier: {classifier_type}')
        self.pipeline = Pipeline([('scaler', StandardScaler()), ('model', classifiers[classifier_type])])

        # Train classifier
        print(f"Training {classifier_type} classifier...")
        self.pipeline.fit(X_train, y_train)
        self.scaler = self.pipeline.named_steps['scaler']
        self.classifier = self.pipeline.named_steps['model']

        # Evaluate
        train_score = self.pipeline.score(X_train, y_train)
        test_score = self.pipeline.score(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5)

        # Predictions for detailed evaluation
        y_pred = self.pipeline.predict(X_test)

        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, labels=np.arange(len(class_names)), target_names=class_names, output_dict=True, zero_division=0)

        self.is_fitted = True

        results = {
            'classifier_type': classifier_type,
            'n_features': X.shape[1],
            'n_samples': len(wafer_maps),
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names))),
            'class_names': class_names,
            'feature_names': self.feature_names
        }

        print("Training completed!")
        print(f"Test Accuracy: {test_score:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return results

    def predict(self, wafer_maps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict defect types for new wafer maps.

        Args:
            wafer_maps: Array of wafer maps to classify

        Returns:
            Tuple of (predicted_labels, prediction_probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be trained before making predictions")

        # Extract features
        X, _ = self.tda_extractor.fit_transform(wafer_maps)
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.classifier.predict(X_scaled)
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        # Get probabilities if available
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(X_scaled)
        else:
            probabilities = None

        return predicted_labels, probabilities

    def visualize_results(self, results: Dict[str, Any], figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize training results.

        Args:
            results: Results dictionary from training
            figsize: Figure size for plots
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Confusion Matrix
        cm = results['confusion_matrix']
        class_names = results['class_names']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')

        # Classification Report Heatmap
        report_df = pd.DataFrame(results['classification_report']).iloc[:-1, :-1].T
        sns.heatmap(report_df, annot=True, cmap='Greens', ax=axes[0,1])
        axes[0,1].set_title('Classification Report')

        # Feature Importance (if available)
        if hasattr(self.classifier, 'feature_importances_'):
            importances = self.classifier.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features

            axes[1,0].bar(range(len(indices)), importances[indices])
            axes[1,0].set_title('Top 20 Feature Importances')
            axes[1,0].set_xlabel('Feature Index')
            axes[1,0].set_ylabel('Importance')
        else:
            axes[1,0].text(0.5, 0.5, 'Feature importance\nnot available for\nthis classifier',
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Feature Importance')

        # Accuracy Summary
        accuracies = {
            'Train': results['train_accuracy'],
            'Test': results['test_accuracy'],
            'CV Mean': results['cv_mean']
        }

        bars = axes[1,1].bar(accuracies.keys(), accuracies.values())
        axes[1,1].set_title('Accuracy Summary')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_ylim(0, 1)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

