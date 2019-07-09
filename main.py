# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import imageio

from dataset import Dataset
from model import Model

def main():
    dataset = Dataset(scale=1.0)
    model = Model()
    model.train(dataset)

    if not os.path.exists("result"):
        os.mkdir("result")

    for i in range(32):
        u1 = model.Us[1][:,i].reshape(16,16)
        u1 = cv2.resize(u1, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        imageio.imwrite("result/u1_{:0>2}.png".format(i), u1)
    
    for i in range(128):
        u2 = model.get_level2_rf(i)
        u2 = cv2.resize(u2, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        imageio.imwrite("result/u2_{:0>3}.png".format(i), u2)
        

if __name__ == '__main__':
    main()
