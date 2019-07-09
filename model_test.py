# -*- coding: utf-8 -*-
import numpy as np
import unittest

from dataset import Dataset
from model import Model


class ModeTest(unittest.TestCase):
    def test_init(self):
        dataset = Dataset(scale=1.0)
        model = Model(iteration=1)
        model.train(dataset)

        rs, rh, error_tds = model.apply_image(dataset, 0, training=True, bar_type=None)
        self.assertEqual(rs.shape, (96,))
        self.assertEqual(rh.shape, (128,))
        self.assertEqual(error_tds.shape, (96,))
        
        rs, rh, error_tds = model.apply_image(dataset, 0, training=False, bar_type="short")
        self.assertEqual(rs.shape, (96,))
        self.assertEqual(rh.shape, (128,))
        self.assertEqual(error_tds.shape, (96,))

        rs, rh, error_tds = model.apply_image(dataset, 0, training=False, bar_type="long")
        self.assertEqual(rs.shape, (96,))
        self.assertEqual(rh.shape, (128,))
        self.assertEqual(error_tds.shape, (96,))

if __name__ == '__main__':
    unittest.main()
