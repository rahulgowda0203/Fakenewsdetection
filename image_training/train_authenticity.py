
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "datasets" / "image" / "authenticity" / "processed"
MODEL_DIR = ROOT / "models" / "image"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

CLASS_NAMES = ["0_real", "1_fake"]

print("TensorFlow version:", tf.__version__)
print("Dataset:", DATA_DIR)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("Class mapping:", train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# --------------------------------------------------
# 3. BUILD MOBILENETV2 MODEL
# --------------------------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.05)(x)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    2,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# 4. SAVE CLASS LABELS
# --------------------------------------------------

with open(MODEL_DIR / "authenticity_classes.json", "w") as f:
    json.dump(CLASS_NAMES, f)

# --------------------------------------------------
# 5. TRAINING CALLBACKS
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "image_authenticity.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

print("\nStarting image authenticity training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# --------------------------------------------------
# 6B. FINE-TUNE MOBILENETV2
# --------------------------------------------------

print("\nStarting MobileNetV2 fine-tuning...")

base_model.trainable = True

# Freeze all but the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# Combine histories for graphs
for key in history.history:
    history.history[key].extend(fine_history.history[key])


# --------------------------------------------------
# 7. VALIDATION EVALUATION
# --------------------------------------------------

print("\nEvaluating on validation data...")

val_loss, val_accuracy = model.evaluate(val_ds)

print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# --------------------------------------------------
# 8. VALIDATION PREDICTIONS
# --------------------------------------------------

y_true = np.concatenate([
    labels.numpy() for _, labels in val_ds
])

probabilities = model.predict(val_ds, verbose=1)

y_pred = np.argmax(probabilities, axis=1)

print("\nValidation Classification Report:")

print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Real", "Fake"],
    zero_division=0
))

# --------------------------------------------------
# 9. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real", "Fake"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Image Authenticity Validation Confusion Matrix")
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "authenticity_confusion_matrix.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 10. ACCURACY GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Image Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "authenticity_accuracy.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 11. LOSS GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Image Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "authenticity_loss.png",
    dpi=300
)

plt.close()

# --------------------------------------------------

# 12. SAVE RESULTS
# --------------------------------------------------

results = {
    "validation_loss": float(val_loss),
    "validation_accuracy": float(val_accuracy),
    "class_names": CLASS_NAMES,
    "confusion_matrix": cm.tolist()
}

with open(MODEL_DIR / "authenticity_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nTRAINING COMPLETED SUCCESSFULLY!")

print("Model:", MODEL_DIR / "image_authenticity.keras")
print("Confusion Matrix:", MODEL_DIR / "authenticity_confusion_matrix.png")
print("Accuracy Graph:", MODEL_DIR / "authenticity_accuracy.png")
print("Loss Graph:", MODEL_DIR / "authenticity_loss.png")
