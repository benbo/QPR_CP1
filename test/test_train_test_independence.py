import unittest
from KwikCluster.MinHash import MinHash, Banding
from load_data import load_files
import numpy as np
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        number_processes = 5
        number_hash_functions = 128
        self.jaccard_threshold = 0.7
        train_text, train_labels, _, train_phones = load_files(
            file_names=['data/ht_training_UPDATED.gz', 'data/ht_training_2.gz'])
        test_text, test_labels, test_ids, test_phones = load_files(
            file_names=['data/ht_evaluation_NOCLASS.gz'])
        minhash = MinHash(number_hash_functions)
        minhash.hash_corpus_list(train_text, number_threads=number_processes)
        minhash.hash_corpus_list(test_text, number_threads=number_processes)
        bands = Banding(number_hash_functions, self.jaccard_threshold)
        bands.add_signatures(minhash.signatures, number_threads=number_processes)
        self.minhash_violations = 0
        for i, (pivot_doc, pivot_bands) in enumerate(bands.doc_to_bands.iteritems()):
            if pivot_doc <= len(train_text):
                print 'Checking doc ' + str(i) + ' of ' + str(len(train_text))
                for band in pivot_bands:
                    docs = np.array(list(bands.band_to_docs[band]))
                    for doc in docs[docs > len(train_text)]:
                        if minhash.jaccard(pivot_doc, doc) > self.jaccard_threshold:
                            self.minhash_violations += 1
        self.phone_violations = len(set(train_phones).intersection(test_phones))

    def test_independence(self):
        print 'Number of phone numbers in train and test set: ' + str(self.phone_violations)
        print 'Number of pairs in train and test set with MinHash Jaccard score above ' + str(self.jaccard_threshold) + \
              ': ' + str(self.minhash_violations)
        self.assertEqual(self.phone_violations, 0)
        self.assertEqual(self.minhash_violations, 0)

        

