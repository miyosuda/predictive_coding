# -*- coding: utf-8 -*-
import numpy as np
import unittest

from dataset import Dataset
from model import Model


class ModeTest(unittest.TestCase):
    def test_init(self):
        dataset = Dataset(scale=1.0)
        model = Model()
        model.train(dataset)

if __name__ == '__main__':
    unittest.main()
