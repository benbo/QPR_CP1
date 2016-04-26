import unittest
from load_data import load_files
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.text, self.labels, self.indices, self.ad_id = load_files()

    def test_load_files(self):
        self.assertEqual(len(self.text), len(self.labels))
        self.assertEqual(len(self.text), len(self.indices))
        self.assertEqual(len(self.text), len(self.ad_id))