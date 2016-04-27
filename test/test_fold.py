import unittest
from Fold import Fold, clusters_to_labels
from load_data import load_files
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        text, self.labels, ad_id, self.phones = load_files(max_lines=500)
        self.folds = Fold(text, self.phones, self.labels, number_processes=4)

    def test_folds(self):
        k = 5

        ## MinHash ##
        minhash_labels = clusters_to_labels(self.folds._minhash_clusters)
        # MinHash Label Folds
        label_folds = self.folds.get_minhash_labelkfolds(k)
        for train_idx, test_idx in label_folds:
            train_clusters = frozenset(minhash_labels[train_idx])
            test_clusters = frozenset(minhash_labels[test_idx])
            self.assertFalse(train_clusters.intersection(test_clusters))
        # MinHash Dedup Folds
        dedup_folds = self.folds.get_minhash_dedupkfolds(k)
        for train_idx, test_idx in dedup_folds:
            self.assertEqual(len(train_idx) + len(test_idx), len(self.folds._minhash_clusters))
            train_clusters = frozenset(minhash_labels[train_idx])
            test_clusters = frozenset(minhash_labels[test_idx])
            self.assertFalse(train_clusters.intersection(test_clusters))

        ## Phones ##
        phone_labels = clusters_to_labels(self.folds._phone_clusters)
        # MinHash Label Folds
        label_folds = self.folds.get_phone_labelkfolds(k)
        for train_idx, test_idx in label_folds:
            train_clusters = frozenset(phone_labels[train_idx])
            test_clusters = frozenset(phone_labels[test_idx])
            self.assertFalse(train_clusters.intersection(test_clusters))
        # MinHash Dedup Folds
        dedup_folds = self.folds.get_phone_dedupkfolds(k)
        for train_idx, test_idx in dedup_folds:
            self.assertEqual(len(train_idx) + len(test_idx), len(self.folds._phone_clusters))
            train_clusters = frozenset(phone_labels[train_idx])
            test_clusters = frozenset(phone_labels[test_idx])
            self.assertFalse(train_clusters.intersection(test_clusters))


