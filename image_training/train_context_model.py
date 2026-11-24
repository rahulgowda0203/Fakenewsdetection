
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
import joblib


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "datasets" / "image" / "verite_source" / "VERITE"
OUTPUT_DIR = ROOT / "models" / "image" / "context"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "VERITE.csv"
IMAGE_EMBEDDINGS_PATH = DATA_DIR / "VERITE_clip_image_embeddings_ViTL14.npy"
TEXT_EMBEDDINGS_PATH = DATA_DIR / "VERITE_clip_text_embeddings_ViTL14.npy"

CLASS_NAMES = ["true", "miscaptioned", "out-of-context"]
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

EPOCHS = 40
BATCH_SIZE = 16
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading VERITE dataset...")

df = pd.read_csv(CSV_PATH)

image_embeddings = np.load(IMAGE_EMBEDDINGS_PATH)
text_embeddings = np.load(TEXT_EMBEDDINGS_PATH)

if len(df) != len(image_embeddings) or len(df) != len(text_embeddings):
    raise ValueError(
        "CSV and embedding row counts do not match. "
        "Check the dataset files before training."
    )

if not set(df["label"].unique()).issubset(LABEL_TO_ID):
    raise ValueError("Unexpected labels found in VERITE.csv")

if df["image_path"].isna().any() or df["label"].isna().any():
    raise ValueError("Missing image paths or labels found.")

print("CSV shape:", df.shape)
print("Image embeddings:", image_embeddings.shape)
print("Text embeddings:", text_embeddings.shape)


# --------------------------------------------------
# 3. BUILD MULTIMODAL FEATURES
# --------------------------------------------------

print("\nCombining image and text embeddings...")

image_norm = image_embeddings / np.maximum(
    np.linalg.norm(image_embeddings, axis=1, keepdims=True),
    1e-8
)

text_norm = text_embeddings / np.maximum(
    np.linalg.norm(text_embeddings, axis=1, keepdims=True),
    1e-8
)

difference = np.abs(image_norm - text_norm)
product = image_norm * text_norm

cosine_similarity = np.sum(
    image_norm * text_norm, axis=1, keepdims=True
)

X = np.concatenate(
    [
        image_norm,
        text_norm,
        difference,
        product,
        cosine_similarity
    ],
    axis=1
).astype(np.float32)

y = df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int32)

groups = df["image_path"].astype(str).to_numpy()

print("Combined feature shape:", X.shape)
print("Number of unique images:", len(np.unique(groups)))


# --------------------------------------------------
# 4. GROUPED TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

print("\nSplitting data by image_path to avoid image leakage...")

first_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_val_indices, test_indices = next(
    first_split.split(X, y, groups=groups)
)

second_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_relative, val_relative = next(
    second_split.split(
        X[train_val_indices],
        y[train_val_indices],
        groups=groups[train_val_indices]
    )
)

train_indices = train_val_indices[train_relative]
val_indices = train_val_indices[val_relative]

X_train, X_val, X_test = (
    X[train_indices],
    X[val_indices],
    X[test_indices]
)

y_train, y_val, y_test = (
    y[train_indices],
    y[val_indices],
    y[test_indices]
)

train_groups = set(groups[train_indices])
val_groups = set(groups[val_indices])
test_groups = set(groups[test_indices])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

print("Training samples:", len(train_indices))
print("Validation samples:", len(val_indices))
print("Testing samples:", len(test_indices))

for name, labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Testing", y_test)
]:
    print(f"\n{name} label distribution:")
    print(pd.Series(
        [CLASS_NAMES[i] for i in labels]
    ).value_counts())


# --------------------------------------------------
# 5. SCALE FEATURES
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

joblib.dump(scaler, OUTPUT_DIR / "context_scaler.pkl")


# --------------------------------------------------
# 6. BUILD TENSORFLOW MODEL
# --------------------------------------------------

print("\nBuilding context consistency model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.40),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------------
# 7. HANDLE CLASS IMBALANCE
# --------------------------------------------------

class_ids = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(class_ids, weights)
}


# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------

checkpoint_path = OUTPUT_DIR / "image_context_model.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

print("\nStarting training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)


# --------------------------------------------------
# 9. LOAD BEST MODEL AND TEST
# --------------------------------------------------

model = tf.keras.models.load_model(checkpoint_path)

test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

probabilities = model.predict(X_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

print("\n========== TEST RESULTS ==========")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    zero_division=0,
    output_dict=True
)

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    zero_division=0
))


# --------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(cmap="Blues", values_format="d")
plt.title("VERITE Context Consistency - Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "context_confusion_matrix.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# 11. ACCURACY GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Context Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "context_accuracy.png", dpi=300)
plt.close()


# --------------------------------------------------
# 12. LOSS GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Context Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "context_loss.png", dpi=300)
plt.close()


# --------------------------------------------------
# 13. SAVE RESULTS
# --------------------------------------------------

results = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "class_names": CLASS_NAMES,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "training_samples": int(len(train_indices)),
    "validation_samples": int(len(val_indices)),
    "testing_samples": int(len(test_indices)),
    "unique_images": int(len(np.unique(groups))),
    "feature_dimension": int(X.shape[1])
}

with open(OUTPUT_DIR / "context_results.json", "w") as file:
    json.dump(results, file, indent=4)

with open(OUTPUT_DIR / "context_classes.json", "w") as file:
    json.dump(CLASS_NAMES, file, indent=4)

print("\n========== TRAINING COMPLETED ==========")
print("Model:", checkpoint_path)
print("Scaler:", OUTPUT_DIR / "context_scaler.pkl")
print("Confusion matrix:", OUTPUT_DIR / "context_confusion_matrix.png")
print("Accuracy graph:", OUTPUT_DIR / "context_accuracy.png")
print("Loss graph:", OUTPUT_DIR / "context_loss.png")
print("Results:", OUTPUT_DIR / "context_results.json")
