import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------- Load Model ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "..", "model.keras")
if not os.path.exists(MODEL_FILE):
    MODEL_FILE = os.path.join(BASE_DIR, "..", "model.h5")

if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file not found. Please place model.keras or model.h5 in project root.")
    st.stop()

model = tf.keras.models.load_model(MODEL_FILE)

IMG_SIZE = (128, 128)

# ---------------------- Utils ----------------------
def preprocess_for_model(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def scan_currency(img):
    """Detect note, crop and warp perspective like a scanner."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # fallback: return original

    # Find largest contour (likely the note)
    contour = max(contours, key=cv2.contourArea)

    # Approximate polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:  # likely a rectangle (note shape)
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # order points: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        scanned = cv2.warpPerspective(img, M, (width, height))
        return scanned

    return img  # fallback if contour not rectangle

def predict_currency(img):
    scanned = scan_currency(img)
    inp = preprocess_for_model(scanned)
    pred = model.predict(inp, verbose=0)[0][0]
    label = "Genuine ✅" if pred < 0.5 else "Fake ❌"
    return scanned, label

# ---------------------- Streamlit UI ----------------------
st.title("💵 Intelligent Fake Currency Identification System")

option = st.radio("Choose Input Method", ["Upload Image", "Scan from Webcam"])

if option == "Upload Image":
    file = st.file_uploader("Upload a currency image", type=["jpg", "jpeg", "png"])
    if file:
        image = np.array(Image.open(file).convert("RGB"))
        scanned, label = predict_currency(image)

        # Show scanned image
        st.image(scanned, use_column_width=True)

        # Highlighted Result (no confidence)
        color = "green" if "Genuine" in label else "red"
        st.markdown(
            f"<h2 style='color:{color}; text-align:center;'>🔍 Result: <b>{label}</b></h2>",
            unsafe_allow_html=True
        )

elif option == "Scan from Webcam":
    st.write("📷 Use your webcam to capture currency")
    picture = st.camera_input("Take a picture")
    if picture:
        image = np.array(Image.open(picture).convert("RGB"))
        scanned, label = predict_currency(image)

        # Show scanned image
        st.image(scanned, use_column_width=True)

        # Highlighted Result (no confidence)
        color = "green" if "Genuine" in label else "red"
        st.markdown(
            f"<h2 style='color:{color}; text-align:center;'>🔍 Result: <b>{label}</b></h2>",
            unsafe_allow_html=True
        )
