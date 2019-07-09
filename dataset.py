# -*- coding: utf-8 -*-
import numpy as np
from scipy import io
import cv2


class Dataset:
    def __init__(self, scale=10.0):
        self.load_images(scale)
        self.mask = self.create_gauss_mask()

    def load_images(self, scale):
        images = np.empty([5,408,512])
        
        for i in range(len(images)):
            image = cv2.imread("data/images_rao/image{}.png".format(i))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            images[i] = image
        self.load_sub(images, scale)

    def create_gauss_mask(self, width=16, sigma=0.4): 
        rates = [0.0] * (width * width)
        hw = width // 2
        for i in range(width):
            x = (i - hw) / float(hw)
            for j in range(width):
                y = (j - hw) / float(hw)
                r = np.sqrt(x*x + y*y)
                rates[j*width + i] = self.gauss(r, sigma=sigma)
        rates = np.array(rates)
        # Normalize
        rates = rates / np.max(rates)
        return rates

    def gauss(self, x, sigma):
        sigma_sq = sigma * sigma
        return 1.0 / np.sqrt(2.0 * np.pi * sigma_sq) * np.exp(-x*x/(2 * sigma_sq))

    def load_sub(self, images, scale):
        self.images = images
        
        filtered_images = []
        for image in images:
            filtered_image = self.apply_DoG_filter(image)
            filtered_images.append(filtered_image)

        self.filtered_images = filtered_images

        w = images.shape[2]
        h = images.shape[1]

        size_w = w // 26
        size_h = h // 16

        patches = np.empty((size_h * size_w * len(images), 16, 26), dtype=np.float32)

        for image_index, filtered_image in enumerate(filtered_images):
            for j in range(size_h):
                y = 16 * j
                for i in range(size_w):
                    x = 26 * i
                    patch = filtered_image[y:y+16, x:x+26]
                    # (16, 26)
                    # print(patch.shape)
                    index = size_w*size_h*image_index + j*size_w + i
                    patches[index] = patch

        patches = patches * scale
        self.patches = patches

    def get_image(self, patch_index, image_index):
        """
        Arguments:
          patch_index: index of the patch
          image_index: 0,1,2

        Returns:
          nd-array: (256,)
        """
        patch = self.patches[patch_index]
        x = 5 * image_index
        return patch[:, x:x+16].reshape([-1]) * self.mask

    def load_matlab(self, scale):
        file_path = "./data/IMAGES_RAW.mat"
        matdata = io.loadmat(file_path)

        images = matdata['IMAGESr'].astype(np.float32)
        # Change image order
        images = np.array([images[:,:,i] for i in range(images.shape[2])])
        self.load_sub(images, scale)

    def apply_DoG_filter(self, gray, ksize=(5,5), sigma1=1.3, sigma2=2.6):
        g1 = cv2.GaussianBlur(gray, ksize, sigma1)
        g2 = cv2.GaussianBlur(gray, ksize, sigma2)
        return g1 - g2
