# -*- coding: utf-8 -*-
import numpy as np
import unittest

from dataset import Dataset

class DatasetTest(unittest.TestCase):
    def test_init(self):
        dataset = Dataset(scale=10.0)
        self.assertEqual(dataset.patches.shape, (2375,16,26))

        for i in range(3):
            image = dataset.get_image(0, i)
            self.assertEqual(image.shape, (256,))

        for i in range(3):
            bar_image_short = dataset.get_bar_image(is_short=True, image_index=i)
            self.assertEqual(bar_image_short.shape, (256,))
            bar_image_long = dataset.get_bar_image(is_short=False, image_index=i)
            self.assertEqual(bar_image_long.shape, (256,))            

if __name__ == '__main__':
    unittest.main()
