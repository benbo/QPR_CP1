from KwikCluster.MinHash import MinHash, Banding
from KwikCluster.KwikCluster import kwik_cluster_minhash, kwik_cluster_dict
from sklearn import cross_validation
import random
import numpy as np


class Fold(object):
    """
    Folding using various deduplication strategies
    """
    def __init__(self, text, phone_numbers, labels, jaccard_threshold=0.9, number_hash_functions=128, number_processes=1):
        """
        :param text: List of record text
        :param phone_numbers: List of tuples, each tuple contains strings of each phone number in ad
        :param labels: List of class labels
        :param jaccard_threshold: Threshold to use in MinHash KwikCluster
        :param number_hash_functions: Number of hash functions to use in MinHash
        :param number_processes: Number of processes to use in MinHash
        """
        self._class_labels = labels
        minhash = MinHash(number_hash_functions)
        minhash.hash_corpus_list(text, number_threads=number_processes)
        bands = Banding(number_hash_functions, jaccard_threshold)
        bands.add_signatures(minhash.signatures, number_threads=number_processes)
        self._minhash_clusters = kwik_cluster_minhash(minhash, bands, jaccard_threshold)
        phone_dict = {doc_idx: set(phone_numbers) for (doc_idx, phone_numbers) in enumerate(phone_numbers)}
        self._phone_clusters = kwik_cluster_dict(phone_dict)

    def get_minhash_dedupkfolds(self, k):
        """
        Sample 1 record per kwik cluster, with standard k-fold
        :param k: Number of folds
        :return folds: List of folds
        """
        deduped_idx = np.array([random.sample(cluster, 1)[0] for cluster in self._minhash_clusters])
        number_clusters = len(self._minhash_clusters)
        original_folds = cross_validation.KFold(number_clusters, k)
        folds = [(deduped_idx[train_idx], deduped_idx[test_idx]) for train_idx, test_idx in original_folds]
        return folds

    def get_minhash_labelkfolds(self, k):
        """
        Label k-fold using KwikCluster labels. Split folds such that same kwik cluster does not appear in multiple folds
        :param k: Number of folds
        :return folds: sklearn fold iterator
        """
        minhash_labels = clusters_to_labels(self._minhash_clusters)
        folds = cross_validation.LabelKFold(minhash_labels, k)
        return folds

    def get_phone_dedupkfolds(self, k):
        """
        Sample 1 record per phone cluster, with standard k-fold
        :param k: Number of folds
        :return folds: List of folds
        """
        deduped_idx = np.array([random.sample(cluster, 1)[0] for cluster in self._phone_clusters])
        number_clusters = len(self._phone_clusters)
        original_folds = cross_validation.KFold(number_clusters, k)
        folds = [(deduped_idx[train_idx], deduped_idx[test_idx]) for train_idx, test_idx in original_folds]
        return folds

    def get_phone_labelkfolds(self, k):
        """
        Label k-fold using phone cluster labels. Split folds such that same kwik cluster does not appear in multiple folds
        :param k: Number of folds
        :return folds: sklearn fold iterator
        """
        phone_labels = clusters_to_labels(self._phone_clusters)
        folds = cross_validation.LabelKFold(phone_labels, k)
        return folds

    def get_naive_folds(self, k):
        """
        Naive stratified k-folds, does not consider duplicate information
        :param k: Number of folds
        :return folds: sklearn fold iterator
        """
        folds = cross_validation.StratifiedKFold(self._class_labels, k)
        return folds


def clusters_to_labels(clusters):
    """
    :param clusters: Frozen set of frozen sets, each frozen set is a set of record ids
    :return labels: Numpy array of cluster labels
    """
    label_dict = dict()
    for label, cluster in enumerate(clusters):
        for idx in cluster:
            label_dict[idx] = label
    labels = np.array([label_dict[idx] for idx in range(0, len(label_dict))])  # should be 0 indexed
    return labels
