# -*- coding: utf-8 -*-
import numpy as np


class Model:
    def __init__(self, iteration=30):
        self.k1      = 0.0005 # Learning rate for r
        self.k2_init = 0.005  # Initial learning rate for U
        self.iteration = iteration

        self.sigma_sq    = 1.0  # Variance of observation distribution of I
        self.sigma_sq_td = 10.0 # Variance of observation distribution of r
        self.alpha1      = 1.0  # Precision param of r prior    (var=1.0,  std=1.0)
        self.alpha2      = 0.05 # Precision param of r_td prior (var=20.0, std=4.5)
        self.lambd       = 0.02 # Precision param of U prior    (var=50.0, std=7.1)
        
        U_scale = 7.0
        self.Us = (np.random.rand(3,256,32)-0.5) * U_scale
        self.Uh = (np.random.rand(96,128)  -0.5) * U_scale

        self.k2 = self.k2_init

    def apply_image(self, dataset, image_index, training, bar_type=None):
        rs = np.zeros([96],  dtype=np.float32)
        rh = np.zeros([128], dtype=np.float32)
        error_tds = np.zeros([96], dtype=np.float32)
    
        for i in range(self.iteration):
            # Loop for iterations

            # Calculate r_td
            r_tds = self.Uh.dot(rh)
        
            for j in range(3):
                # Level1 update
                # Loop for 3 receptive fields.
                if bar_type == None:
                    # Normal image input
                    I = dataset.get_image(image_index, j)
                elif bar_type == "long":
                    I = dataset.get_bar_image(is_short=False, image_index=j)
                elif bar_type == "short":
                    I = dataset.get_bar_image(is_short=True, image_index=j)
                    
                r    = rs[32*j:32*(j+1)]
                r_td = r_tds[32*j:32*(j+1)]
                    
                U  = self.Us[j]
                Ur = U.dot(r)
                
                error    = I - Ur
                error_td = r_td - r
                    
                dr = (self.k1 / self.sigma_sq) * U.T.dot(error) + \
                     (self.k1/self.sigma_sq_td) * error_td \
                     - self.k1 * self.alpha1 * r
                rs[32*j:32*(j+1)] += dr

                if training:
                    dU = (self.k2 / self.sigma_sq) * np.outer(error, r) \
                         - self.k2 * self.lambd * U
                    self.Us[j] += dU
                    
                error_tds[32*j:32*(j+1)] = error_td

            # Level2 update
            drh = (self.k1 / self.sigma_sq_td) * self.Uh.T.dot(-error_tds) \
                  - self.k1 * self.alpha2 * rh
            rh += drh

            if training:
                dUh = (self.k2 / self.sigma_sq_td) * np.outer(-error_tds, rh) \
                      - self.k2 * self.lambd * self.Uh
                self.Uh += dUh

        return rs, r_tds, rh, error_tds

    def train(self, dataset):
        self.k2 = self.k2_init
        
        patch_size = len(dataset.patches) # 2375

        for i in range(patch_size):
            # Loop for all patches
            rs, r_tds, rh, error_tds = self.apply_image(dataset, i, training=True)
            
            if i % 100 == 0:
                print("us mean={} std={}".format(np.mean(self.Us[0]), np.std(self.Us[0])))
                print("rs mean={} std={}".format(np.mean(rs), np.std(rs)))
    
            if i % 40 == 0:
                # Decay learning rate for U
                self.k2 = self.k2 / 1.015

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
