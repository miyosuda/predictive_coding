# -*- coding: utf-8 -*-
import numpy as np
from scipy import io
import cv2


class Dataset:
    def __init__(self):
        file_path = "./data/IMAGES_RAW.mat"
        matdata = io.loadmat(file_path)

        images = matdata['IMAGESr'].astype(np.float32)
        # Change image order
        images = np.array([images[:,:,i] for i in range(images.shape[2])])

        filtered_images = []
        for image in images:
            filtered_image = self.apply_DoG_filter(image)
            filtered_images.append(filtered_image)

        w = images.shape[2]
        h = images.shape[1]

        size_w = w // 26
        size_h = h // 16

        patches = np.empty((size_h * size_w * len(images), 16, 26), dtype=np.float32)
        for filtered_image in filtered_images:
            for j in range(size_h):
                y = 16 * j
                for i in range(size_w):
                    x = 26 * i
                    patch = image[y:y+16, x:x+26]
                    # (16, 26)
                    # print(patch.shape)
                    patches[j*size_w + i] = patch

        self.patches = patches

    def apply_DoG_filter(self, gray, ksize=(5,5), sigma1=1.3, sigma2=2.6):
        g1 = cv2.GaussianBlur(gray, ksize, sigma1)
        g2 = cv2.GaussianBlur(gray, ksize, sigma2)
        return g1 - g2
