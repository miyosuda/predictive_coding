# -*- coding: utf-8 -*-
import numpy as np
import unittest

from dataset import Dataset

class DatasetTest(unittest.TestCase):
    def test_init(self):
        dataset = Dataset()
        #self.assertEqual(dataset.patches.shape, (6080,16,26))
        self.assertEqual(dataset.patches.shape, (2375,16,26))


if __name__ == '__main__':
    unittest.main()
