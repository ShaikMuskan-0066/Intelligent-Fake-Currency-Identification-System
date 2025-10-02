import sys
import cv2
import numpy as np
import tensorflow as tf
import os

# Config
IMG_SIZE = (128, 128)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.keras")
CLASS_FILE = os.path.join(BASE_DIR, "class_indices.npy")

# Load model
model = tf.keras.models.load_model(MODEL_FILE, compile=False)

# Load classes
if os.path.exists(CLASS_FILE):
    class_names = np.load(CLASS_FILE, allow_pickle=True)
else:
    class_names = ["Genuine", "Fake"]

def preprocess_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return
    img = cv2.imread(img_path)
    x = preprocess_frame(img)
    preds = model.predict(x, verbose=0)[0]
    label = class_names[np.argmax(preds)]
    print(f"✅ Prediction: {label} (conf: {np.max(preds):.4f})")

def predict_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not detected.")
        return
    print("🎥 Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x = preprocess_frame(frame)
        preds = model.predict(x, verbose=0)[0]
        label = class_names[np.argmax(preds)]
        cv2.putText(frame, f"{label} ({np.max(preds):.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0) if "Genuine" in label else (0,0,255), 2)
        cv2.imshow("Currency Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py [webcam|image path]")
    elif sys.argv[1] == "webcam":
        predict_webcam()
    elif sys.argv[1] == "image" and len(sys.argv) == 3:
        predict_image(sys.argv[2])
    else:
        print("❌ Invalid arguments")
