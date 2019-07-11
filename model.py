# -*- coding: utf-8 -*-
import numpy as np
import os


class Model:
    def __init__(self, iteration=30):
        self.iteration = iteration
        
        self.k1      = 0.0005 # Learning rate for r
        self.k2_init = 0.005  # Initial learning rate for U

        self.sigma_sq    = 1.0  # Variance of observation distribution of I
        self.sigma_sq_td = 10.0 # Variance of observation distribution of r
        self.alpha1      = 1.0  # Precision param of r prior    (var=1.0,  std=1.0)
        self.alpha2      = 0.05 # Precision param of r_td prior (var=20.0, std=4.5)
        self.lambd1      = 0.02 # Precision param of U prior    (var=50.0, std=7.1)
        self.lambd2      = 0.00001 # Precision param of Uh prior
        
        U_scale = 7.0
        self.Us = (np.random.rand(3,256,32) - 0.5) * U_scale
        self.Uh = (np.random.rand(96,128)   - 0.5) * U_scale

        self.k2 = self.k2_init

        # Scaling parameter for learning rate of level2
        self.level2_lr_scale = 10.0

    def apply_images(self, images, training):
        rs = np.zeros([96],  dtype=np.float32)
        rh = np.zeros([128], dtype=np.float32)
        
        error_tds = np.zeros([96], dtype=np.float32)
    
        for i in range(self.iteration):
            # Loop for iterations

            # Calculate r_td
            r_tds = self.Uh.dot(rh) # (96,)

            for j in range(3):
                I = images[j]
                r    = rs[   32*j:32*(j+1)]
                r_td = r_tds[32*j:32*(j+1)]
                    
                U  = self.Us[j]
                Ur = U.dot(r)
                
                error    = I - Ur
                error_td = r_td - r
                    
                dr = (self.k1/self.sigma_sq) * U.T.dot(error) + \
                     (self.k1/self.sigma_sq_td) * error_td - self.k1 * self.alpha1 * r
                if training:
                    dU = (self.k2/self.sigma_sq) * np.outer(error, r) \
                         - self.k2 * self.lambd1 * U
                    self.Us[j] += dU
                rs[32*j:32*(j+1)] += dr
                    
                error_tds[32*j:32*(j+1)] = error_td

            # Level2 update
            drh = (self.k1*self.level2_lr_scale / self.sigma_sq_td) * self.Uh.T.dot(-error_tds) \
                  - self.k1*self.level2_lr_scale * self.alpha2 * rh
            if training:
                dUh = (self.k2*self.level2_lr_scale / self.sigma_sq_td) * np.outer(-error_tds, rh) \
                      - self.k2*self.level2_lr_scale * self.lambd2 * self.Uh
                # (96,128)
                self.Uh += dUh
            rh += drh

        return rs, r_tds, rh, error_tds

    def train(self, dataset):
        self.k2 = self.k2_init
        
        patch_size = len(dataset.patches) # 2375

        for i in range(patch_size):
            # Loop for all patches
            images = dataset.get_images(i)
            rs, r_tds, rh, error_tds = self.apply_images(images, training=True)
            
            if i % 100 == 0:
                print("rs    std={:.2f}".format(np.std(rs)))
                print("r_tds std={:.2f}".format(np.std(r_tds)))
                print("U     std={:.2f}".format(np.std(self.Us)))
                print("Uh    std={:.2f}".format(np.std(self.Uh)))
    
            if i % 40 == 0:
                # Decay learning rate for U
                self.k2 = self.k2 / 1.015

        print("train finished")

    def reconstruct(self, r, level=1):
        if level==1:
            rs = r
        else:
            rh = r
            rs = self.Uh.dot(rh) # (96,)
            
        patch = np.zeros((16,26), dtype=np.float32)
        
        for i in range(3):
            r = rs[32*i:32*(i+1)]
            U = self.Us[i]
            Ur = U.dot(r).reshape(16,16)
            patch[:, 5*i:5*i+16] += Ur
        return patch

    def get_level2_rf(self, index):
        Uh0 = self.Uh[:,index][0:32]
        Uh1 = self.Uh[:,index][32:64]
        Uh2 = self.Uh[:,index][64:96]

        UU0 = self.Us[0].dot(Uh0).reshape((16,16))
        UU1 = self.Us[1].dot(Uh1).reshape((16,16))
        UU2 = self.Us[2].dot(Uh2).reshape((16,16))
        
        rf = np.zeros((16,26), dtype=np.float32)
        rf[:, 5*0:5*0+16] += UU0
        rf[:, 5*1:5*1+16] += UU1
        rf[:, 5*2:5*2+16] += UU2    
        return rf

    def save(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, "model") 

        np.savez_compressed(file_path,
                            Us=self.Us,
                            Uh=self.Uh)
        print("saved: {}".format(dir_name))

    def load(self, dir_name):
        file_path = os.path.join(dir_name, "model.npz")
        if not os.path.exists(file_path):
            print("saved file not found")
            return
        data = np.load(file_path)
        self.Us = data["Us"]
        self.Uh = data["Uh"]
        print("loaded: {}".format(dir_name))
