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
        filters_col = self.filters.reshape(self.num_filters, -1)

        out_col = np.dot(self.patches_col, filters_col.T)
        output = out_col.reshape(out_h, out_w, self.num_filters)
        return output

    def backward(self, d_L_d_out, learn_rate):
        d_out_col = d_L_d_out.reshape(-1, self.num_filters)
        d_filters_col = np.dot(d_out_col.T, self.patches_col)
        d_filters = d_filters_col.reshape(self.filters.shape)
        self.filters -= learn_rate * d_filters
        return None

class MaxPool2:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, input_vol):
        self.last_input = input_vol
        H, W, num_filters = input_vol.shape