# -*- coding: utf-8 -*-
import numpy as np
import unittest

from dataset import Dataset

class DatasetTest(unittest.TestCase):
    def test_init(self):
        dataset = Dataset(scale=10.0)
        #patch_size = 2375
        patch_size = 3040
        self.assertEqual(dataset.patches.shape, (patch_size,16,26))

        images = dataset.get_images(0)
        bar_images_short = dataset.get_bar_images(is_short=True)
        bar_images_long  = dataset.get_bar_images(is_short=False)
        
        for i in range(3):
            self.assertEqual(images[i].shape, (256,))
            self.assertEqual(bar_images_short[i].shape, (256,))
            self.assertEqual(bar_images_long[i].shape, (256,))

if __name__ == '__main__':
    unittest.main()
