import cv2
import numpy as np

IMG_SIZE = 240


def preprocess(roi):
  
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=4
    )
    
    edges = cv2.Canny(adaptive, 20, 60)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    filtered = cv2.bitwise_not(edges)
    rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    reshaped = np.reshape(rgb / 255.0, (1, IMG_SIZE, IMG_SIZE, 3))
    
    return reshaped, filtered