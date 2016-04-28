import unittest
from KwikCluster.MinHash import MinHash, Banding
from load_data import load_files
from itertools import combinations
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
        train_idx = set(range(0, len(train_text)))
        minhash.hash_corpus_list(test_text, number_threads=number_processes)
        test_idx = set(range(len(train_text), len(train_text) + len(test_text)))

        bands = Banding(number_hash_functions, self.jaccard_threshold)
        bands.add_signatures(minhash.signatures, number_threads=number_processes)
        self.minhash_violations = 0
        for i, (band, docs) in enumerate(bands.band_to_docs.iteritems()):
            print 'Checking band ' + str(i) + ' of ' + len(bands.band_to_docs)
            if docs.intersection(train_idx) and docs.intersection(test_idx):
                print '     Found some MinHash matches in band with ' + str(len(docs)) + ' docs'
                pairs = combinations(docs)
                for pair in pairs:
                    if minhash.jaccard(pair[0], pair[1]):
                        self.minhash_violations += 1

        self.phone_violations = len(set(train_phones).intersection(test_phones))

    def test_independence(self):
        print 'Number of phone numbers in train and test set: ' + str(self.phone_violations)
        print 'Number of pairs in train and test set with MinHash Jaccard score above ' + str(self.jaccard_threshold) + \
              ': ' + str(self.minhash_violations)
        self.assertEqual(self.phone_violations, 0)
        self.assertEqual(self.minhash_violations, 0)

        

