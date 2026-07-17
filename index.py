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
        probs = self.forward(image)
        loss = self.softmax.calculate_loss(label_idx)
        acc = 1 if np.argmax(probs) == label_idx else 0

        gradient = self.softmax.backward(label_idx)
        gradient = self.dense2.backward(gradient, learn_rate)
        gradient = self.relu2.backward(gradient)
        gradient = self.dense1.backward(gradient, learn_rate)
        gradient = self.pool.backward(gradient)
        gradient = self.relu1.backward(gradient)
        self.conv.backward(gradient, learn_rate)

        return loss, acc

    def predict(self, image):
        probs = self.forward(image)
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        return self.class_names[pred_idx], confidence
def save_weights(self, filepath="shape_weights.npz"):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(dir_path, filepath)
        np.savez(
            full_path, 
            conv_filters=self.conv.filters, 
            dense1_weights=self.dense1.weights, 
            dense1_biases=self.dense1.biases,
            dense2_weights=self.dense2.weights,
            dense2_biases=self.dense2.biases
        )
        print(f"Weights saved to {full_path}")

    def load_weights(self, filepath="shape_weights.npz"):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        full_path = os.path.join(dir_path, filepath)
        if os.path.exists(full_path):
            data = np.load(full_path)
            self.conv.filters = data['conv_filters']
            if 'dense1_weights' in data:
                
                self.dense1.weights = data['dense1_weights']
                self.dense1.biases = data['dense1_biases']
                self.dense2.weights = data['dense2_weights']
                self.dense2.biases = data['dense2_biases']
                print(f"Weights loaded from {full_path}")
            else:
                print("Warning: Old weights format detected. Cannot load. Using random.")
        else:
            print(f"Warning: {full_path} not found. Using random initialized weights.")

import numpy as np

def preprocess_image(image_matrix, target_size=(28, 28)):
    y_coords, x_coords = np.where(image_matrix > 0.1)
    
    if len(x_coords) > 0:
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        w = max_x - min_x
        h = max_y - min_y
        pad = max(int(max(w, h) * 0.15), 10)
        
        side = max(w, h) + 2 * pad
        cx = min_x + w // 2
        cy = min_y + h // 2
        
        crop_min_x = max(0, int(cx - side / 2))
        crop_max_x = min(image_matrix.shape[1], int(cx + side / 2))
        crop_min_y = max(0, int(cy - side / 2))
        crop_max_y = min(image_matrix.shape[0], int(cy + side / 2))
        
        image_matrix = image_matrix[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

    h, w = image_matrix.shape
    if h == 0 or w == 0:
        return np.zeros(target_size)
        
    from PIL import Image
    pil_img = Image.fromarray((image_matrix * 255).astype(np.uint8))
    resample_filter = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR
    resized_pil = pil_img.resize(target_size, resample_filter)
    
    resized = np.array(resized_pil).astype(float) / 255.0
    
    max_val = np.max(resized)
    if max_val > 0:
        resized = resized / max_val
        
    resized[resized > 0.1] = 1.0
    resized[resized <= 0.1] = 0.0
    
    return resized

def extract_centroid(pixel_matrix):
    y_coords, x_coords = np.where(pixel_matrix > 0)
    if len(y_coords) == 0:
        return 0, 0
    return np.mean(x_coords), np.mean(y_coords)

def process_circle(pixel_matrix):
    cx, cy = extract_centroid(pixel_matrix)
    y_coords, x_coords = np.where(pixel_matrix > 0)
    
    distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    r = np.max(distances) if len(distances) > 0 else 0
    
    area = np.pi * (r ** 2)
    circumference = 2 * np.pi * r
    
    svg_path = f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="var(--primary-color, #4f46e5)" fill="transparent" stroke-width="3" vector-effect="non-scaling-stroke"/>'
    return {
        "shape": "circle",
        "properties": {
            "radius": float(round(r, 2)),
            "area": float(round(area, 2)),
            "circumference": float(round(circumference, 2)),
            "centroid": (float(round(cx, 2)), float(round(cy, 2)))
        },
        "svg": svg_path
    }

def process_rectangle(pixel_matrix, shape_name="rectangle"):
    y_coords, x_coords = np.where(pixel_matrix > 0)
    
    if len(x_coords) == 0:
        return {"shape": shape_name, "properties": {}, "svg": ""}
        
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if shape_name == "square":
        side = (width + height) / 2
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        min_x, min_y = cx - side/2, cy - side/2
        width = height = side
    
    area = width * height
    perimeter = 2 * (width + height)
    
    svg_path = f'<rect x="{min_x}" y="{min_y}" width="{width}" height="{height}" stroke="var(--primary-color, #4f46e5)" fill="transparent" stroke-width="3" vector-effect="non-scaling-stroke"/>'
    
    return {
        "shape": shape_name,
        "properties": {
            "width": float(round(width, 2)),
            "height": float(round(height, 2)),
            "area": float(round(area, 2)),
            "perimeter": float(round(perimeter, 2))
        },
        "svg": svg_path
    }

def process_triangle(pixel_matrix):
    y_coords, x_coords = np.where(pixel_matrix > 0)
    
    if len(x_coords) == 0:
        return {"shape": "triangle", "properties": {}, "svg": ""}
        
    points = np.column_stack((x_coords, y_coords))
    
    top_idx = np.argmin(points[:, 1])
    top_point = points[top_idx]
    
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    max_y = np.max(points[:, 1])
    
    dist_bl = (points[:, 0] - min_x)**2 + (points[:, 1] - max_y)**2
    bl_point = points[np.argmin(dist_bl)]
    
    dist_br = (points[:, 0] - max_x)**2 + (points[:, 1] - max_y)**2
    br_point = points[np.argmin(dist_br)]
    
    vertices = [top_point, bl_point, br_point]
    
    def dist(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
    s1 = dist(vertices[0], vertices[1])
    s2 = dist(vertices[1], vertices[2])
    s3 = dist(vertices[2], vertices[0])
    
    perimeter = s1 + s2 + s3
    s = perimeter / 2
    area_squared = max(0, s * (s - s1) * (s - s2) * (s - s3))
    area = np.sqrt(area_squared)
    
    tol = 0.15 * max(s1, s2, s3)
    if abs(s1 - s2) < tol and abs(s2 - s3) < tol:
        triangle_type = "equilateral"
    elif abs(s1 - s2) < tol or abs(s2 - s3) < tol or abs(s1 - s3) < tol:
        triangle_type = "isosceles"
    else:
        triangle_type = "scalene"
        
    points_str = " ".join([f"{p[0]},{p[1]}" for p in vertices])
    svg_path = f'<polygon points="{points_str}" stroke="var(--primary-color, #4f46e5)" fill="transparent" stroke-width="3" vector-effect="non-scaling-stroke"/>'
    
     return {
        "shape": "triangle",
        "properties": {
            "type": triangle_type,
            "side_lengths": (float(round(s1, 2)), float(round(s2, 2)), float(round(s3, 2))),
            "area": float(round(area, 2)),
            "perimeter": float(round(perimeter, 2))
        },
        "svg": svg_path
    }

def process_line(pixel_matrix):
    y_coords, x_coords = np.where(pixel_matrix > 0)
    
    if len(x_coords) == 0:
        return {"shape": "line", "properties": {}, "svg": ""}
        
    points = np.column_stack((x_coords, y_coords))
    
    idx_min_x, idx_max_x = np.argmin(points[:, 0]), np.argmax(points[:, 0])
    idx_min_y, idx_max_y = np.argmin(points[:, 1]), np.argmax(points[:, 1])
    
    dist_x = np.linalg.norm(points[idx_min_x] - points[idx_max_x])
    dist_y = np.linalg.norm(points[idx_min_y] - points[idx_max_y])
    
    if dist_x > dist_y:
        p1, p2 = points[idx_min_x], points[idx_max_x]
    else:
        p1, p2 = points[idx_min_y], points[idx_max_y]
        
    length = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    svg_path = f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" stroke="var(--primary-color, #4f46e5)" stroke-width="3" vector-effect="non-scaling-stroke"/>'
    
    return {
        "shape": "line",
        "properties": {
            "length": float(round(length, 2)),
            "endpoints": ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
            },
        "svg": svg_path
    }

def extract_geometry(predicted_class, pixel_matrix):
    if predicted_class == "circle":
        return process_circle(pixel_matrix)
    elif predicted_class == "triangle":
        return process_triangle(pixel_matrix)
    elif predicted_class in ["square", "rectangle"]:
        return process_rectangle(pixel_matrix, shape_name=predicted_class)
    elif predicted_class == "line":
        return process_line(pixel_matrix)
    else:
        raise ValueError(f"Unknown shape class: {predicted_class}")

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask import send_from_directory
import base64
import numpy as np
from PIL import Image
import io
import os
import traceback


app = Flask(__name__)
CORS(app)

model = CNNModel(num_classes=5)
model.load_weights("shape_weights.npz")

@app.route('/')
def serve_index():
    public_dir = os.path.dirname(os.path.realpath(__file__))
    return send_from_directory(public_dir, 'index.html')

@app.route('/logo.png')
def serve_logo():
    public_dir = os.path.dirname(os.path.realpath(__file__))
    return send_from_directory(public_dir, 'logo.png')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        img_array = np.array(image)

        if np.max(img_array[:, :, 3]) > 0:
            pixel_matrix = img_array[:, :, 3] / 255.0
        else:
            gray = np.array(image.convert('L'))
            pixel_matrix = 1.0 - (gray / 255.0)

        processed_img = preprocess_image(pixel_matrix, target_size=(28, 28))

        predicted_shape, confidence = model.predict(processed_img)

        geometry_data = extract_geometry(predicted_shape, pixel_matrix)

        response = {
            "predicted_shape": predicted_shape,
            "confidence": round(float(confidence), 4),
            "geometry": geometry_data["properties"],
            "svg": geometry_data["svg"]
        }
