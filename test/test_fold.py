import unittest
from Fold import Fold
from load_data import load_files
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        text, self.labels, ad_id, self.phones = load_files(max_lines=500)
        self.folds = Fold(text, self.phones, self.labels, number_processes=4)

    def test_folds(self):
        k = 5
        label_folds = self.folds.get_kwik_labelkfolds(k)
        for train_idx, test_idx in label_folds:
            train_clusters = frozenset(self.folds._kwik_labels[train_idx])
            test_clusters = frozenset(self.folds._kwik_labels[test_idx])
            self.assertFalse(train_clusters.intersection(test_clusters))
        dedup_folds = self.folds.get_kwik_dedupkfolds(k)
        for train_idx, test_idx in dedup_folds:
            self.assertEqual(len(train_idx)+len(test_idx), len(self.folds._kwik_clusters))
            train_clusters = frozenset(self.folds._kwik_labels[train_idx])
            test_clusters = frozenset(self.folds._kwik_labels[test_idx])
            self.assertFalse(train_clusters.intersection(test_clusters))

