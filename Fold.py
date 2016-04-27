from KwikCluster.MinHash import MinHash, Banding
from KwikCluster.KwikCluster import kwik_cluster
from sklearn import cross_validation
import random
import numpy as np

class Fold(object):
    """
    Folding using various deduplication strategies
    """
    def __init__(self, text, phone_numbers, jaccard_threshold=0.9, number_hash_functions=128, number_processes=1):
        """
        :param text: List of record text
        :param phone_numbers: List of phone numbers
        """
        minhash = MinHash(number_hash_functions)
        minhash.hash_corpus_list(text, number_threads=number_processes)
        bands = Banding(number_hash_functions, jaccard_threshold)
        bands.add_signatures(minhash.signatures, number_threads=number_processes)
        self._kwik_clusters = kwik_cluster(minhash, bands, jaccard_threshold)
        self._kwik_labels = clusters_to_labels(self._kwik_clusters)

    def get_kwik_dedupkfolds(self, k):
        """
        Sample 1 record per cluster, with standard k-fold
        :param k: Number of folds
        :return folds: List of folds
        """
        # need a list of labels. Then sklearn will return iterator of which samples belong in this fold. But then I
        # need to remap this iterator to the original ad ids
        deduped_idx = np.array([random.sample(cluster, 1)[0] for cluster in self._kwik_clusters])
        number_clusters = len(self._kwik_clusters)
        original_folds = cross_validation.KFold(number_clusters, k)
        folds = [(deduped_idx[train_idx], deduped_idx[test_idx] for train_idx, test_idx in original_folds)]
        return folds

    def get_kwik_labelkfolds(self, k):
        """
        Label k-fold using KwikCluster labels. Split folds such that same kwik cluster does not appear in multiple folds
        :param k: Number of folds
        :return folds: sklearn fold iterator
        """
        folds = cross_validation.LabelKFold(self._kwik_labels, k)
        return folds


def clusters_to_labels(clusters):
    """
    :param clusters: Frozen set of frozen sets, each frozen set is a set of record ids
    :return labels: Dict of [record id, cluster label]
    """
    labels = dict()
    for label, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = label
    return labels