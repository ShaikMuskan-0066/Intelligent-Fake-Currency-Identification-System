from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Create FastAPI app
app = FastAPI()

# Define paths (go up one level from backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(BASE_DIR, "model.keras")
CLASS_FILE = os.path.join(BASE_DIR, "class_indices.npy")

# Ensure model file exists
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_FILE}")

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

# Preprocess function
def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    image = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# Root endpoint
@app.get("/")
def root():
    return {"message": "✅ Fake Currency Detection Backend Running"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    processed = preprocess_image(image)

    preds = model.predict(processed, verbose=0)[0]
    class_idx = np.argmax(preds)
    label = class_names[class_idx]
    confidence = float(preds[class_idx])

    return {"label": label, "confidence": confidence}
