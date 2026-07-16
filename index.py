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
        out_h = H // self.pool_size
        out_w = W // self.pool_size

        output = np.zeros((out_h, out_w, num_filters))

        for i in range(out_h):
            for j in range(out_w):
                im_region = input_vol[(i * self.pool_size):(i * self.pool_size + self.pool_size), 
                                      (j * self.pool_size):(j * self.pool_size + self.pool_size)]
                output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backward(self, d_L_d_out):
        d_L_d_in = np.zeros(self.last_input.shape)
        H, W, num_filters = self.last_input.shape
        out_h = H // self.pool_size
        out_w = W // self.pool_size

        for i in range(out_h):
            for j in range(out_w):
                im_region = self.last_input[(i * self.pool_size):(i * self.pool_size + self.pool_size), 
                                            (j * self.pool_size):(j * self.pool_size + self.pool_size)]

                for f in range(num_filters):
                    flat_idx = np.argmax(im_region[:, :, f])
                    y, x = np.unravel_index(flat_idx, (self.pool_size, self.pool_size))
                    d_L_d_in[i * self.pool_size + y, j * self.pool_size + x, f] = d_L_d_out[i, j, f]

        return d_L_d_in

class ReLU:
    def forward(self, input_vol):
        self.last_input = input_vol
        return np.maximum(0, input_vol)

    def backward(self, d_L_d_out):
        d_L_d_in = d_L_d_out.copy()
        d_L_d_in[self.last_input <= 0] = 0
        return d_L_d_in
    
class Dense:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) * np.sqrt(2.0 / input_len)
        self.biases = np.zeros(nodes)

    def forward(self, input_vol):
        self.last_input_shape = input_vol.shape
        self.last_input = input_vol.flatten()
        return np.dot(self.last_input, self.weights) + self.biases

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_w = np.outer(self.last_input, d_L_d_out)
        d_L_d_b = d_L_d_out
        d_L_d_inputs = np.dot(self.weights, d_L_d_out)
        self.weights -= learn_rate * d_L_d_w
        self.biases -= learn_rate * d_L_d_b
        return d_L_d_inputs.reshape(self.last_input_shape)

class SoftmaxCrossEntropy:
    def forward(self, input_vec):
        self.last_input = input_vec
        shifted_logits = input_vec - np.max(input_vec)
        exps = np.exp(shifted_logits)
        self.probs = exps / np.sum(exps)
        return self.probs

    def backward(self, true_label_idx):
        gradient = self.probs.copy()
        gradient[true_label_idx] -= 1
        return gradient

    def calculate_loss(self, true_label_idx):
        return -np.log(self.probs[true_label_idx] + 1e-9)

class CNNModel:
    def __init__(self, num_classes=5):
        self.conv = Conv3x3(8, filter_size=3)
        self.relu1 = ReLU()
        self.pool = MaxPool2(pool_size=2)
        self.dense1 = Dense(13 * 13 * 8, 128)
        self.relu2 = ReLU()
        self.dense2 = Dense(128, num_classes)
        self.softmax = SoftmaxCrossEntropy()

        self.class_names = [
            "circle", "triangle", "square", "rectangle", "line"
        ]

    def forward(self, image):
        out = self.conv.forward(image)
        out = self.relu1.forward(out)
        out = self.pool.forward(out)
        out = self.dense1.forward(out)
        out = self.relu2.forward(out)
        out = self.dense2.forward(out)
        probs = self.softmax.forward(out)
        return probs

    def train_step(self, image, label_idx, learn_rate=0.01):