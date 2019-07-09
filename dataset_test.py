# -*- coding: utf-8 -*-
import numpy as np
import unittest

from dataset import Dataset

class DatasetTest(unittest.TestCase):
    def test_init(self):
        dataset = Dataset(scale=10.0)
        self.assertEqual(dataset.patches.shape, (2375,16,26))

        image = dataset.get_image(0, 0)
        self.assertEqual(image.shape, (256,))


if __name__ == '__main__':
    unittest.main()
