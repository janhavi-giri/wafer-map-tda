import unittest
from unittest.mock import patch
import numpy as np
from wafer_map_tda import SyntheticWaferMapGenerator, TDAFeatureExtractor, WaferDefectClassifier

class RegressionTests(unittest.TestCase):
    def test_feature_contract(self):
        for dim in (1, 2):
            extractor = TDAFeatureExtractor(max_dimension=dim)
            maps = np.zeros((2, 8, 8)); maps[1, 2:4, 2:4] = 1
            features, names = extractor.fit_transform(maps)
            self.assertEqual(features.shape, (2, 13*(dim+1)+8))
            self.assertEqual(features.shape[1], len(names))
            self.assertTrue(np.isfinite(features).all())
            self.assertFalse(np.array_equal(features[0], features[1]))

    def test_small_point_cloud_h0(self):
        e = TDAFeatureExtractor(max_dimension=1)
        self.assertEqual(len(e._compute_persistence_diagram(np.empty((0,2)))[0]), 0)
        d = e._compute_persistence_diagram(np.array([[0,0],[3,4]]))[0]
        np.testing.assert_array_equal(d[:,1], [5, np.inf])

    def test_persistence_errors_are_visible(self):
        with patch('wafer_map_tda.classifier.ripser.ripser', side_effect=ValueError('failure')):
            with self.assertRaises(RuntimeError):
                TDAFeatureExtractor()._compute_persistence_diagram(np.array([[0,0],[0,1],[1,0]]))

    def test_invalid_input(self):
        for x in (np.array([]), np.array([[np.nan]]), np.array([[-1]])):
            with self.assertRaises(ValueError): TDAFeatureExtractor().extract_features(x)
        with self.assertRaises(ValueError): SyntheticWaferMapGenerator().generate_dataset(0)

    def test_generator_reproducibility(self):
        np.random.seed(42)
        a, la = SyntheticWaferMapGenerator().generate_dataset(20)
        np.random.seed(42)
        b, lb = SyntheticWaferMapGenerator().generate_dataset(20)
        np.testing.assert_array_equal(a,b); self.assertEqual(la,lb)

    def test_training_and_prediction(self):
        # Balanced, fixed class counts keep five-fold CV meaningful in a small test.
        np.random.seed(42)
        g = SyntheticWaferMapGenerator(wafer_size=12)
        samples = [g.generate_wafer_map(label) for label in ('Center','Donut','None') for _ in range(15)]
        maps = np.array([x[0] for x in samples]); labels = [x[1] for x in samples]
        c = WaferDefectClassifier(TDAFeatureExtractor(max_dimension=1))
        result = c.train(maps, labels)
        pred, proba = c.predict(maps[:2])
        self.assertEqual(len(pred),2);self.assertEqual(proba.shape,(2,3))
        self.assertEqual(result['n_features'],34)
        self.assertIn('scaler', c.pipeline.named_steps)

    def test_variation_is_opt_in_and_label_preserving(self):
        np.random.seed(1); plain = SyntheticWaferMapGenerator().generate_wafer_map('Donut')[0]
        np.random.seed(1); same = SyntheticWaferMapGenerator(variation=None).generate_wafer_map('Donut')[0]
        np.testing.assert_array_equal(plain, same)
        for level in ('clean', 'mild', 'moderate', 'hard'):
            g = SyntheticWaferMapGenerator(variation=level)
            for kind in ('Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'None'):
                m, label = g.generate_wafer_map(kind)
                self.assertEqual(label, kind); self.assertEqual(m.shape, (26, 26))
                self.assertTrue(np.isfinite(m).all()); self.assertTrue((m >= 0).all())
        with self.assertRaises(ValueError): SyntheticWaferMapGenerator(variation='extreme')
        with self.assertRaises(ValueError): SyntheticWaferMapGenerator(variation={'blur': 1})

    def test_point_cloud_threshold(self):
        m = np.zeros((8, 8)); m[2:4, 2:4] = 0.2; m[5:7, 5:7] = 0.9
        low = TDAFeatureExtractor(max_dimension=1)
        high = TDAFeatureExtractor(max_dimension=1, point_cloud_threshold=0.5)
        self.assertEqual(low.point_cloud_threshold, 0.1)
        self.assertEqual(len(low._extract_point_cloud(m, threshold=low.point_cloud_threshold)), 8)
        self.assertEqual(len(high._extract_point_cloud(m, threshold=high.point_cloud_threshold)), 4)
        self.assertEqual(high.extract_features(m).shape, (34,))

if __name__ == '__main__': unittest.main()
