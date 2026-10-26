import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, affine_transform
from typing import Tuple, List, Optional, Union
import os
import json
from datetime import datetime
import pickle

# Optional realism knobs. ``None`` (the default) reproduces the original, idealized maps.
#   noise_sigma     std of additive Gaussian measurement noise (clipped at 0)
#   speckle         probability that a pixel becomes an isolated false "defect"
#   size_jitter     defect pattern is scaled by exp(U(-j, j)) about the wafer center
#   position_jitter defect pattern is shifted by U(-p, p) pixels in x and y
#   stretch         anisotropic stretch: axis ratio exp(U(0, k)) along a random direction
#   rotation        pattern is rotated by U(0, rotation) degrees
VARIATION_PRESETS = {
    "clean":    dict(noise_sigma=0.0,  speckle=0.0,  size_jitter=0.0,  position_jitter=0.0, stretch=0.0, rotation=0.0),
    "mild":     dict(noise_sigma=0.03, speckle=0.01, size_jitter=0.15, position_jitter=1.0, stretch=0.2, rotation=360.0),
    "moderate": dict(noise_sigma=0.08, speckle=0.03, size_jitter=0.25, position_jitter=2.0, stretch=0.4, rotation=360.0),
    "hard":     dict(noise_sigma=0.12, speckle=0.06, size_jitter=0.35, position_jitter=3.0, stretch=0.6, rotation=360.0),
}


class SyntheticWaferMapGenerator:
    """
    Generator for synthetic wafer map data based on WM811K dataset patterns.

    The WM811K dataset contains 9 defect types:
    - Center: defects in the center area
    - Donut: ring-shaped defects
    - Edge-Loc: defects at the edge
    - Edge-Ring: defects forming a ring near the edge
    - Loc: localized defects
    - Random: randomly scattered defects
    - Scratch: linear scratch patterns
    - Near-full: nearly full wafer defects
    - None: no defects (good wafer)
    """

    def __init__(self, wafer_size: int = 26, variation: Optional[Union[str, dict]] = None):
        """
        Initialize the wafer map generator.

        Args:
            wafer_size: Size of the wafer map (default 26x26 like WM811K)
            variation: ``None`` for the original idealized maps, a preset name
                ("clean", "mild", "moderate", "hard") or a dict with the keys
                documented at ``VARIATION_PRESETS``. Variation is applied to every
                defect type, so class labels are unchanged.
        """
        if isinstance(variation, str):
            if variation not in VARIATION_PRESETS:
                raise ValueError(f"Unknown variation preset: {variation}")
            variation = VARIATION_PRESETS[variation]
        elif variation is not None:
            unknown = set(variation) - set(VARIATION_PRESETS["clean"])
            if unknown:
                raise ValueError(f"Unknown variation keys: {sorted(unknown)}")
            variation = {**VARIATION_PRESETS["clean"], **variation}
        self.variation = variation
        self.wafer_size = wafer_size
        self.center = wafer_size // 2
        self.radius = wafer_size // 2

        # Create circular mask for wafer boundary
        self.wafer_mask = self._create_wafer_mask()

    def _create_wafer_mask(self) -> np.ndarray:
        """Create a circular mask representing the wafer boundary."""
        y, x = np.ogrid[:self.wafer_size, :self.wafer_size]
        mask = ((x - self.center) ** 2 + (y - self.center) ** 2) <= (self.radius ** 2)
        return mask

    def _apply_variation(self, wafer_map: np.ndarray) -> np.ndarray:
        """Deform the defect pattern (rotation, stretch, size, shift) and add noise."""
        v = self.variation
        c = np.array([self.center, self.center], dtype=float)

        theta = np.deg2rad(np.random.uniform(0, v["rotation"])) if v["rotation"] else 0.0
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        phi = np.random.uniform(0, np.pi)
        axis = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        ratio = np.exp(np.random.uniform(0, v["stretch"])) if v["stretch"] else 1.0
        stretch = axis @ np.diag([np.sqrt(ratio), 1 / np.sqrt(ratio)]) @ axis.T

        scale = np.exp(np.random.uniform(-v["size_jitter"], v["size_jitter"])) if v["size_jitter"] else 1.0
        forward = scale * stretch @ rot                     # pattern -> output
        shift = np.random.uniform(-v["position_jitter"], v["position_jitter"], 2) if v["position_jitter"] else np.zeros(2)
        inverse = np.linalg.inv(forward)
        offset = c - inverse @ (c + shift)                  # output coords -> input coords
        out = affine_transform(wafer_map, inverse, offset=offset, order=1, mode="constant", cval=0.0)

        if v["noise_sigma"]:
            out = out + np.random.normal(0, v["noise_sigma"], out.shape)
        if v["speckle"]:
            speck = np.random.random(out.shape) < v["speckle"]
            out[speck] = np.maximum(out[speck], np.random.uniform(0.3, 0.9, speck.sum()))
        return np.clip(out, 0, None) * self.wafer_mask

    def generate_center_defect(self, intensity: float = 0.8, size_factor: float = 0.3) -> np.ndarray:
        """Generate center defect pattern."""
        wafer_map = np.zeros((self.wafer_size, self.wafer_size))

        # Create center region
        center_radius = int(self.radius * size_factor)
        y, x = np.ogrid[:self.wafer_size, :self.wafer_size]
        center_mask = ((x - self.center) ** 2 + (y - self.center) ** 2) <= (center_radius ** 2)

        # Add some randomness
        noise = np.random.random((self.wafer_size, self.wafer_size)) * 0.3
        wafer_map[center_mask] = intensity + noise[center_mask]

        return wafer_map * self.wafer_mask

    def generate_donut_defect(self, intensity: float = 0.8, inner_ratio: float = 0.3, outer_ratio: float = 0.7) -> np.ndarray:
        """Generate donut (ring) defect pattern."""
        wafer_map = np.zeros((self.wafer_size, self.wafer_size))

        y, x = np.ogrid[:self.wafer_size, :self.wafer_size]
        distances = np.sqrt((x - self.center) ** 2 + (y - self.center) ** 2)

        inner_radius = self.radius * inner_ratio
        outer_radius = self.radius * outer_ratio

        ring_mask = (distances >= inner_radius) & (distances <= outer_radius)

        # Add noise and variation
        noise = np.random.random((self.wafer_size, self.wafer_size)) * 0.2
        wafer_map[ring_mask] = intensity + noise[ring_mask]

        return wafer_map * self.wafer_mask

    def generate_edge_loc_defect(self, intensity: float = 0.8, num_spots: int = 3) -> np.ndarray:
        """Generate edge localized defect pattern."""
        wafer_map = np.zeros((self.wafer_size, self.wafer_size))

        # Generate random points near the edge
        for _ in range(num_spots):
            angle = np.random.uniform(0, 2 * np.pi)
            edge_radius = self.radius * np.random.uniform(0.7, 0.95)

            center_x = int(self.center + edge_radius * np.cos(angle))
            center_y = int(self.center + edge_radius * np.sin(angle))

            # Create localized defect
            spot_size = np.random.randint(2, 5)
            y, x = np.ogrid[:self.wafer_size, :self.wafer_size]
            spot_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= (spot_size ** 2)

            wafer_map[spot_mask] += intensity * np.random.uniform(0.7, 1.0)

        return np.clip(wafer_map, 0, 1) * self.wafer_mask

    def generate_edge_ring_defect(self, intensity: float = 0.8, ring_width: float = 0.1) -> np.ndarray:
        """Generate edge ring defect pattern."""
        wafer_map = np.zeros((self.wafer_size, self.wafer_size))

        y, x = np.ogrid[:self.wafer_size, :self.wafer_size]
        distances = np.sqrt((x - self.center) ** 2 + (y - self.center) ** 2)

        inner_radius = self.radius * (1 - ring_width)
        outer_radius = self.radius

        ring_mask = (distances >= inner_radius) & (distances <= outer_radius)

        # Add variation
        noise = np.random.random((self.wafer_size, self.wafer_size)) * 0.3
        wafer_map[ring_mask] = intensity + noise[ring_mask]

        return wafer_map * self.wafer_mask

    def generate_loc_defect(self, intensity: float = 0.8, num_spots: int = 5) -> np.ndarray:
        """Generate localized defect pattern."""
        wafer_map = np.zeros((self.wafer_size, self.wafer_size))

        # Generate random spots within the wafer
        for _ in range(num_spots):
            # Random position within wafer
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, self.radius * 0.8)

            center_x = int(self.center + radius * np.cos(angle))
            center_y = int(self.center + radius * np.sin(angle))

            # Create spot
            spot_size = np.random.randint(1, 4)
            y, x = np.ogrid[:self.wafer_size, :self.wafer_size]
            spot_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= (spot_size ** 2)

            wafer_map[spot_mask] += intensity * np.random.uniform(0.6, 1.0)

        return np.clip(wafer_map, 0, 1) * self.wafer_mask

    def generate_random_defect(self, intensity: float = 0.8, density: float = 0.1) -> np.ndarray:
        """Generate random scattered defect pattern."""
        wafer_map = np.zeros((self.wafer_size, self.wafer_size))

        # Random defects scattered across the wafer
        random_mask = np.random.random((self.wafer_size, self.wafer_size)) < density
        wafer_map[random_mask] = intensity * np.random.uniform(0.5, 1.0, size=np.sum(random_mask))

        return wafer_map * self.wafer_mask

    def generate_scratch_defect(self, intensity: float = 0.8, num_scratches: int = 2) -> np.ndarray:
        """Generate scratch defect pattern."""
        wafer_map = np.zeros((self.wafer_size, self.wafer_size))

        for _ in range(num_scratches):
            # Random scratch parameters
            angle = np.random.uniform(0, np.pi)
            length = np.random.uniform(self.wafer_size * 0.3, self.wafer_size * 0.8)
            width = np.random.randint(1, 3)

            # Start point
            start_x = np.random.randint(width, self.wafer_size - width)
            start_y = np.random.randint(width, self.wafer_size - width)

            # Draw scratch
            for i in range(int(length)):
                x = int(start_x + i * np.cos(angle))
                y = int(start_y + i * np.sin(angle))

                if 0 <= x < self.wafer_size and 0 <= y < self.wafer_size:
                    # Add width to scratch
                    for dx in range(-width, width + 1):
                        for dy in range(-width, width + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.wafer_size and 0 <= ny < self.wafer_size:
                                wafer_map[ny, nx] = intensity

        return wafer_map * self.wafer_mask

    def generate_near_full_defect(self, intensity: float = 0.8, coverage: float = 0.8) -> np.ndarray:
        """Generate near-full defect pattern."""
        wafer_map = np.zeros((self.wafer_size, self.wafer_size))

        # Fill most of the wafer with defects
        base_intensity = intensity * np.random.uniform(0.6, 1.0, (self.wafer_size, self.wafer_size))

        # Create some variation and holes
        noise_mask = np.random.random((self.wafer_size, self.wafer_size)) < coverage
        wafer_map[noise_mask] = base_intensity[noise_mask]

        # Smooth the pattern
        wafer_map = gaussian_filter(wafer_map, sigma=1)

        return wafer_map * self.wafer_mask

    def generate_none_defect(self) -> np.ndarray:
        """Generate good wafer (no defects)."""
        wafer_map = np.zeros((self.wafer_size, self.wafer_size))

        # Add very small amount of noise to simulate measurement noise
        noise = np.random.normal(0, 0.05, (self.wafer_size, self.wafer_size))
        wafer_map = np.clip(noise, 0, None)

        return wafer_map * self.wafer_mask

    def generate_wafer_map(self, defect_type: str, **kwargs) -> Tuple[np.ndarray, str]:
        """
        Generate a wafer map with specified defect type.

        Args:
            defect_type: Type of defect to generate
            **kwargs: Additional parameters for defect generation

        Returns:
            Tuple of (wafer_map, defect_type)
        """
        defect_generators = {
            'Center': self.generate_center_defect,
            'Donut': self.generate_donut_defect,
            'Edge-Loc': self.generate_edge_loc_defect,
            'Edge-Ring': self.generate_edge_ring_defect,
            'Loc': self.generate_loc_defect,
            'Random': self.generate_random_defect,
            'Scratch': self.generate_scratch_defect,
            'Near-full': self.generate_near_full_defect,
            'None': self.generate_none_defect
        }

        if defect_type not in defect_generators:
            raise ValueError(f"Unknown defect type: {defect_type}")

        wafer_map = defect_generators[defect_type](**kwargs)
        if self.variation is not None:
            wafer_map = self._apply_variation(wafer_map)
        return wafer_map, defect_type

    def generate_dataset(self, n_samples: int = 1000,
                        defect_distribution: Optional[dict] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Generate a dataset of synthetic wafer maps.

        Args:
            n_samples: Number of samples to generate
            defect_distribution: Distribution of defect types (if None, uses the illustrative imbalanced default)

        Returns:
            Tuple of (wafer_maps, labels)
        """
        if defect_distribution is None:
            # Illustrative distribution; not fitted to WM811K
            defect_distribution = {
                'Center': 0.04,
                'Donut': 0.03,
                'Edge-Loc': 0.05,
                'Edge-Ring': 0.04,
                'Loc': 0.09,
                'Random': 0.07,
                'Scratch': 0.06,
                'Near-full': 0.02,
                'None': 0.60
            }

        if n_samples < 1 or not defect_distribution or any(v < 0 for v in defect_distribution.values()) or sum(defect_distribution.values()) <= 0:
            raise ValueError('Positive sample count and nonnegative, positive-total distribution required')

        # Normalize distribution
        total = sum(defect_distribution.values())
        defect_distribution = {k: v/total for k, v in defect_distribution.items()}

        # Generate samples
        wafer_maps = []
        labels = []

        for _ in range(n_samples):
            # Choose defect type based on distribution
            defect_type = np.random.choice(
                list(defect_distribution.keys()),
                p=list(defect_distribution.values())
            )

            wafer_map, label = self.generate_wafer_map(defect_type)
            wafer_maps.append(wafer_map)
            labels.append(label)

        return np.array(wafer_maps), labels

    def visualize_samples(self, wafer_maps: np.ndarray, labels: List[str],
                         n_samples: int = 9, figsize: Tuple[int, int] = (12, 12)):
        """Visualize sample wafer maps."""
        n_samples = min(n_samples, len(wafer_maps))
        if n_samples < 1:
            raise ValueError('At least one sample is required')
        rows = int(np.sqrt(n_samples))
        cols = int(np.ceil(n_samples / rows))

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

        for i in range(n_samples):
            row = i // cols
            col = i % cols

            axes[row, col].imshow(wafer_maps[i], cmap='hot', interpolation='nearest')
            axes[row, col].set_title(f'Type: {labels[i]}')
            axes[row, col].axis('off')

        # Hide empty subplots
        for i in range(n_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    def save_dataset_to_folders(self, wafer_maps: np.ndarray, labels: List[str],
                               base_path: str = "./synthetic_wafer_dataset",
                               save_format: str = "npy",
                               save_images: bool = True,
                               google_drive_path: Optional[str] = None):
        """
        Save wafer maps organized by categories in folders.

        Args:
            wafer_maps: Array of wafer maps
            labels: List of corresponding labels
            base_path: Base directory path for saving
            save_format: Format to save arrays ('npy', 'pickle', or 'both')
            save_images: Whether to save visualization images
            google_drive_path: Path to Google Drive folder (if mounted)
        """
        # Use Google Drive path if provided
        if google_drive_path:
            base_path = os.path.join(google_drive_path, "synthetic_wafer_dataset")

        # Create timestamp for this generation session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = os.path.join(base_path, f"dataset_{timestamp}")

        # Create base directory
        os.makedirs(dataset_path, exist_ok=True)

        # Get unique categories
        unique_labels = list(set(labels))
        category_counts = {label: 0 for label in unique_labels}

        # Create category folders and save data
        for label in unique_labels:
            category_path = os.path.join(dataset_path, label)
            os.makedirs(category_path, exist_ok=True)

            # Create subdirectories
            if save_format in ['npy', 'both']:
                os.makedirs(os.path.join(category_path, 'arrays'), exist_ok=True)
            if save_format in ['pickle', 'both']:
                os.makedirs(os.path.join(category_path, 'pickle'), exist_ok=True)
            if save_images:
                os.makedirs(os.path.join(category_path, 'images'), exist_ok=True)

        # Save wafer maps by category
        print(f"Saving dataset to: {dataset_path}")

        for i, (wafer_map, label) in enumerate(zip(wafer_maps, labels)):
            category_counts[label] += 1
            sample_id = f"{label}_{category_counts[label]:04d}"

            category_path = os.path.join(dataset_path, label)

            # Save as numpy array
            if save_format in ['npy', 'both']:
                npy_path = os.path.join(category_path, 'arrays', f"{sample_id}.npy")
                np.save(npy_path, wafer_map)

            # Save as pickle
            if save_format in ['pickle', 'both']:
                pickle_path = os.path.join(category_path, 'pickle', f"{sample_id}.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(wafer_map, f)

            # Save visualization image
            if save_images:
                img_path = os.path.join(category_path, 'images', f"{sample_id}.png")
                self._save_wafer_image(wafer_map, img_path, title=f"Type: {label}")

            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(wafer_maps)} samples...")

        # Save dataset metadata
        metadata = {
            'generation_timestamp': timestamp,
            'total_samples': len(wafer_maps),
            'wafer_size': self.wafer_size,
            'categories': list(unique_labels),
            'category_distribution': {label: labels.count(label) for label in unique_labels},
            'save_format': save_format,
            'save_images': save_images
        }

        metadata_path = os.path.join(dataset_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save combined dataset files for convenience
        if save_format in ['npy', 'both']:
            np.save(os.path.join(dataset_path, 'all_wafer_maps.npy'), wafer_maps)
            np.save(os.path.join(dataset_path, 'all_labels.npy'), np.array(labels))

        # Create summary report
        self._create_summary_report(dataset_path, metadata)

        print("\nDataset saved successfully!")
        print(f"Location: {dataset_path}")
        print(f"Categories: {len(unique_labels)}")
        print(f"Total samples: {len(wafer_maps)}")

        return dataset_path

    def _save_wafer_image(self, wafer_map: np.ndarray, filepath: str, title: str = ""):
        """Save individual wafer map as image."""
        plt.figure(figsize=(6, 6))
        plt.imshow(wafer_map, cmap='hot', interpolation='nearest')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    def _create_summary_report(self, dataset_path: str, metadata: dict):
        """Create a summary report of the generated dataset."""
        report_path = os.path.join(dataset_path, 'dataset_summary.txt')

        with open(report_path, 'w') as f:
            f.write("SYNTHETIC WAFER MAP DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Generation Date: {metadata['generation_timestamp']}\n")
            f.write(f"Total Samples: {metadata['total_samples']}\n")
            f.write(f"Wafer Size: {metadata['wafer_size']}x{metadata['wafer_size']}\n")
            f.write(f"Save Format: {metadata['save_format']}\n")
            f.write(f"Images Saved: {metadata['save_images']}\n\n")

            f.write("CATEGORY DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            for category, count in metadata['category_distribution'].items():
                percentage = (count / metadata['total_samples']) * 100
                f.write(f"{category:12}: {count:5} samples ({percentage:5.1f}%)\n")

            f.write("\nFOLDER STRUCTURE:\n")
            f.write("-" * 30 + "\n")
            f.write("dataset_YYYYMMDD_HHMMSS/\n")
            f.write("├── metadata.json\n")
            f.write("├── dataset_summary.txt\n")
            f.write("├── all_wafer_maps.npy\n")
            f.write("├── all_labels.npy\n")
            for category in metadata['categories']:
                f.write(f"├── {category}/\n")
                f.write("│   ├── arrays/     (individual .npy files)\n")
                if metadata['save_format'] in ['pickle', 'both']:
                    f.write("│   ├── pickle/     (individual .pkl files)\n")
                if metadata['save_images']:
                    f.write("│   └── images/     (visualization .png files)\n")

    def setup_google_drive_colab(self):
        """
        Setup Google Drive in Google Colab environment.
        Returns the mounted drive path.
        """
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            drive_path = '/content/drive/MyDrive'
            print(f"Google Drive mounted at: {drive_path}")
            return drive_path
        except ImportError:
            print("Not running in Google Colab environment.")
            print("To use Google Drive, please:")
            print("1. Run this code in Google Colab")
            print("2. Or manually specify the Google Drive path")
            return None

    def load_dataset_from_folders(self, dataset_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load previously saved dataset from folders.

        Args:
            dataset_path: Path to the dataset folder

        Returns:
            Tuple of (wafer_maps, labels)
        """
        # Try to load combined files first
        combined_maps_path = os.path.join(dataset_path, 'all_wafer_maps.npy')
        combined_labels_path = os.path.join(dataset_path, 'all_labels.npy')

        if os.path.exists(combined_maps_path) and os.path.exists(combined_labels_path):
            wafer_maps = np.load(combined_maps_path)
            labels = np.load(combined_labels_path).tolist()
            print(f"Loaded dataset: {len(wafer_maps)} samples")
            return wafer_maps, labels

        # Load from individual category folders
        wafer_maps = []
        labels = []

        for category_name in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category_name)
            if not os.path.isdir(category_path):
                continue

            arrays_path = os.path.join(category_path, 'arrays')
            if os.path.exists(arrays_path):
                for filename in sorted(os.listdir(arrays_path)):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(arrays_path, filename)
                        wafer_map = np.load(file_path)
                        wafer_maps.append(wafer_map)
                        labels.append(category_name)

        print(f"Loaded dataset from folders: {len(wafer_maps)} samples")
        return np.array(wafer_maps), labels


