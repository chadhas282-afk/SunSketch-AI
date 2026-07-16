import numpy as np
import os

class Conv3x3:
    def __init__(self, num_filters, filter_size=3):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2.0 / (filter_size * filter_size))

    def forward(self, input_img):
        self.last_input = input_img
        H, W = input_img.shape
        out_h = H - self.filter_size + 1
        out_w = W - self.filter_size + 1

        shape = (out_h, out_w, self.filter_size, self.filter_size)
        strides = input_img.strides * 2
        patches = np.lib.stride_tricks.as_strided(input_img, shape=shape, strides=strides)

        self.patches_col = patches.reshape(-1, self.filter_size * self.filter_size)