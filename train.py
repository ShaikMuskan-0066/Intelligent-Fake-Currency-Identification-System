import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

layers = keras.layers
models = keras.models
AUTOTUNE = tf.data.AUTOTUNE

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.keras")
CLASS_FILE = os.path.join(BASE_DIR, "class_indices.npy")

# Constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

def train_model():
    # ✅ Corrected dataset paths (use dateset instead of dataset)
    train_dir = os.path.join(BASE_DIR, "dateset", "train", "dataset", "training")
    val_dir   = os.path.join(BASE_DIR, "dateset", "train", "dataset", "validation")

    # Load datasets
    train_ds_raw = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    val_ds_raw = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # ✅ Get and save class names BEFORE mapping
    class_names = train_ds_raw.class_names
    np.save(CLASS_FILE, class_names)
    print("✅ Classes:", class_names)

    # Normalize & optimize pipeline
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds_raw.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(buffer_size=AUTOTUNE)

    # ✅ Build model inside the function
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation="softmax")   # ✅ FIXED
    ])

    # ✅ Correct indentation
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Save model
    model.save(MODEL_FILE, include_optimizer=False)
    print(f"✅ Model saved as {MODEL_FILE}")

    # Plot training
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.savefig("training_plot.png")
    plt.show()

if __name__ == "__main__":
    train_model()
