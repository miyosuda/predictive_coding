# -*- coding: utf-8 -*-
import numpy as np
import unittest

from dataset import Dataset
from model import Model

DEBUG_TEST_SAVING = False


class ModeTest(unittest.TestCase):
    def test_init(self):
        dataset = Dataset(scale=1.0)
        model = Model(iteration=1)
        model.train(dataset)

        images = dataset.get_images(0)
        rs, r_tds, rh, error_tds = model.apply_images(images, training=True)
        self.assertEqual(rs.shape, (96,))
        self.assertEqual(r_tds.shape, (96,))
        self.assertEqual(rh.shape, (128,))
        self.assertEqual(error_tds.shape, (96,))

        patch_rec1 = model.reconstruct(rs, level=1)
        self.assertEqual(patch_rec1.shape, (16, 26))

        patch_rec2 = model.reconstruct(rh, level=2)
        self.assertEqual(patch_rec2.shape, (16, 26))

        bar_images_short = dataset.get_bar_images(is_short=True)
        rs, r_tds, rh, error_tds = model.apply_images(bar_images_short, training=False)
        self.assertEqual(rs.shape, (96,))
        self.assertEqual(r_tds.shape, (96,))
        self.assertEqual(rh.shape, (128,))
        self.assertEqual(error_tds.shape, (96,))

        bar_images_long = dataset.get_bar_images(is_short=False)
        rs, r_tds, rh, error_tds = model.apply_images(bar_images_long, training=False)
        self.assertEqual(rs.shape, (96,))
        self.assertEqual(r_tds.shape, (96,))
        self.assertEqual(rh.shape, (128,))
        self.assertEqual(error_tds.shape, (96,))

    def test_save(self):
        if DEBUG_TEST_SAVING:
            model = Model(iteration=1)
            model.save("tmp")
            model.load("tmp")

        

if __name__ == '__main__':
    unittest.main()
