# -*- coding: utf-8 -*-
import numpy as np
import cv2


class Dataset:
    def __init__(self, scale=10.0):
        self.load_images(scale)
        self.mask = self.create_gauss_mask()

    def load_images(self, scale):
        images = []
        dir_name = "images_org"
        #dir_name = "images_brd"
        
        for i in range(5):
            image = cv2.imread("data/{}/image{}.png".format(dir_name, i))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            images.append(image)
        images = np.array(images)
        self.load_sub(images, scale)

    def create_gauss_mask(self, sigma=0.4):
        """ Create gaussian mask. """
        width = 16
        mask = [0.0] * (width * width)
        hw = width // 2
        for i in range(width):
            x = (i - hw) / float(hw)
            for j in range(width):
                y = (j - hw) / float(hw)
                r = np.sqrt(x*x + y*y)
                mask[j*width + i] = self.gauss(r, sigma=sigma)
        mask = np.array(mask)
        # Normalize
        mask = mask / np.max(mask)
        return mask

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

    def get_images_from_patch(self, patch):
        images = []
        for i in range(3):
            x = 5 * i
            # Apply gaussian mask
            image = patch[:, x:x+16].reshape([-1]) * self.mask
            images.append(image)
        return images

    def get_images(self, patch_index):
        patch = self.patches[patch_index]
        return self.get_images_from_patch(patch)

    def apply_DoG_filter(self, gray, ksize=(5,5), sigma1=1.3, sigma2=2.6):
        """
        Apply difference of gaussian (DoG) filter detect edge of the image.
        """
        g1 = cv2.GaussianBlur(gray, ksize, sigma1)
        g2 = cv2.GaussianBlur(gray, ksize, sigma2)
        return g1 - g2

    def get_bar_images(self, is_short):
        patch = self.get_bar_patch(is_short)
        return self.get_images_from_patch(patch)

    def get_bar_patch(self, is_short):
        """
        Get bar patch image for end stopping test.
        """
        bar_patch = np.zeros((16,26), dtype=np.float32)
    
        if is_short:
            bar_width = 6
        else:
            bar_width = 24
        bar_height = 2
    
        for x in range(bar_patch.shape[1]):
            for y in range(bar_patch.shape[0]):
                if x >= 26/2 - bar_width/2 and \
                x < 26/2 + bar_width/2 and \
                y >= 16/2 - bar_height/2 and \
                y < 16/2 + bar_height/2:
                    bar_patch[y,x] = 1.0

        # Sete scale with stddev of all patch images.
        scale = np.std(self.patches)
        # Original scaling value for bar
        bar_scale = 2.0
        return bar_patch * scale * bar_scale
