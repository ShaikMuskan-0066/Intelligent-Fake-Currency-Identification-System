import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

def preprocess_image(img_path, target_size=(128, 128)):
    """Load and preprocess an image for prediction"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_array = img_to_array(img) / 255.0  # normalize
    return np.expand_dims(img_array, axis=0)

def preprocess_frame(frame, target_size=(128, 128)):
    """Preprocess a webcam frame"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_array = img_to_array(img) / 255.0  # normalize
    return np.expand_dims(img_array, axis=0)
