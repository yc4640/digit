# backend/utils.py
import numpy as np

def preprocess_pixels(pixels):
    arr = np.array(pixels, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0  # 0..1
    arr[arr < 0.1] = 0.0  # 简易去噪
    return arr.reshape(1, -1)  # (1,784)
