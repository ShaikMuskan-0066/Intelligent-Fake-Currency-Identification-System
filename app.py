import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------- Load Model ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.keras")
CLASS_FILE = os.path.join(BASE_DIR, "class_indices.npy")

if not os.path.exists(MODEL_FILE):
    st.error(f"❌ Model file not found at {MODEL_FILE}")
    st.stop()

# Load model
model = tf.keras.models.load_model(MODEL_FILE, compile=False)

# Load class names
if os.path.exists(CLASS_FILE):
    class_names = np.load(CLASS_FILE, allow_pickle=True).tolist()
    if isinstance(class_names, dict):
        class_names = list(class_names.keys())  # handle dict format
else:
    class_names = ["Genuine", "Fake"]

IMG_SIZE = (128, 128)

# ---------------------- Utils ----------------------
def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    image = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def predict_currency(image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed, verbose=0)[0]
    class_idx = np.argmax(preds)
    label = class_names[class_idx]
    confidence = float(preds[class_idx])
    return label, confidence

# ---------------------- Streamlit UI ----------------------
st.title("💵 Fake Currency Detection")

option = st.radio("Choose Input Method", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    file = st.file_uploader("Upload a currency image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        label, confidence = predict_currency(image)
        color = "green" if label == "Genuine" else "red"
        st.markdown(
            f"<h2 style='color:{color}; text-align:center;'>🔍 Result: {label} ({confidence:.2f})</h2>",
            unsafe_allow_html=True
        )

elif option == "Use Webcam":
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)

        label, confidence = predict_currency(image)
        color = "green" if label == "Genuine" else "red"
        st.markdown(
            f"<h2 style='color:{color}; text-align:center;'>🔍 Result: {label} ({confidence:.2f})</h2>",
            unsafe_allow_html=True
        )
