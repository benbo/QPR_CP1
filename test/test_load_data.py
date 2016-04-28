import unittest
from load_data import load_files
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.text, self.labels, self.indices, self.ad_id = load_files(max_lines=1000)
        self.test_text, self.test_labels, self.test_indices, self.test_ad_id = load_files(file_names=['data/ht_evaluation_NOCLASS.gz'], max_lines=500)

    def test_load_files(self):
        self.assertEqual(len(self.text), 1000)
        self.assertEqual(len(self.text), len(self.labels))
        self.assertEqual(len(self.text), len(self.indices))
        self.assertEqual(len(self.text), len(self.ad_id))
        self.assertEqual(len(self.test_text), 500)
        self.assertEqual(len(self.test_text), len(self.test_labels))
        self.assertEqual(len(self.test_text), len(self.test_indices))
        self.assertEqual(len(self.test_text), len(self.test_ad_id))
