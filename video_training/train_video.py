
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE = Path(__file__).resolve().parents[1]

MANIFEST = BASE / "video_training/video_manifest_split.csv"
MODEL_DIR = BASE / "models/video"
GRAPH_DIR = BASE / "video_training/results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# ==========================================================
# 2. GPU CONFIGURATION
# ==========================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:", gpus)

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print("GPU configuration warning:", error)
else:
    print("No GPU detected. Training will use CPU.")

# ==========================================================
# 3. LOAD DATASET MANIFEST
# ==========================================================

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)

required_columns = {
    "frame_path", "source_id", "generator", "split"
}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Manifest must contain columns: {required_columns}"
    )


df["source_id"] = df["source_id"].astype(str).str.zfill(3)

df["frame_path"] = df["frame_path"].apply(
    lambda path: str(BASE / path)
)

# Verify image files exist.
missing_files = [
    path for path in df["frame_path"]
    if not Path(path).exists()
]

if missing_files:
    raise FileNotFoundError(
        f"{len(missing_files)} image files are missing. "
        f"First missing file: {missing_files[0]}"
    )

class_names = sorted(df["generator"].unique())

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

print("\nClasses:")
for name, index in class_to_index.items():
    print(index, ":", name)

# ==========================================================
# 4. CREATE TRAIN / VALIDATION / TEST SPLITS
# ==========================================================

validation_ids = ["027", "525"]

train_df = df[
    (df["split"] == "train") &
    (~df["source_id"].isin(validation_ids))
].copy()

val_df = df[
    df["source_id"].isin(validation_ids)
].copy()

test_df = df[
    df["split"] == "test"
].copy()

if train_df.empty or val_df.empty or test_df.empty:
    raise ValueError("One of the dataset splits is empty.")

# Verify no source ID overlaps.
train_ids = set(train_df["source_id"])
val_ids = set(val_df["source_id"])
test_ids = set(test_df["source_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print("\nDataset distribution:")
print("Training frames:", len(train_df))
print("Validation frames:", len(val_df))
print("Testing frames:", len(test_df))

print("\nTraining source IDs:", sorted(train_ids))
print("Validation source IDs:", sorted(val_ids))
print("Testing source IDs:", sorted(test_ids))

# Every split should contain all seven generator classes.
for split_name, split_df in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    if set(split_df["generator"]) != set(class_names):
        raise ValueError(
            f"{split_name} split does not contain all classes."
        )

# ==========================================================
# 5. IMAGE LOADING PIPELINE
# ==========================================================

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


def make_dataset(dataframe, training=False):

    paths = dataframe["frame_path"].values

    labels = dataframe["generator"].map(
        class_to_index
    ).values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# ==========================================================
# 6. DATA AUGMENTATION
# ==========================================================

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
], name="data_augmentation")

# ==========================================================
# 7. BUILD MOBILENETV2 MODEL
# ==========================================================

print("\nLoading MobileNetV2 pretrained weights...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Fine-tune the final 30 MobileNetV2 layers.
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen for stability.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax",
    name="generator_predictions"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

# ==========================================================
# 8. TRAINING CALLBACKS
# ==========================================================

best_model_path = MODEL_DIR / "best_video_model.keras"

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]

# ==========================================================
# 9. TRAIN MODEL
# ==========================================================

print("\nStarting MobileNetV2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ==========================================================
# 10. LOAD BEST MODEL
# ==========================================================

print("\nLoading best checkpoint...")

model = tf.keras.models.load_model(
    best_model_path
)

# ==========================================================
# 11. FINAL TEST EVALUATION
# ==========================================================

print("\nEvaluating on independent test dataset...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(
    y_true,
    y_pred
)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(matrix)

# ==========================================================
# 12. SAVE MODEL AND CLASS LABELS
# ==========================================================



# ==========================================================
# 11A. VIDEO-LEVEL EVALUATION
# ==========================================================

print("\nVideo-Level Evaluation")

video_predictions = []

# Predict all test frames in the same order as test_df.
frame_probabilities = model.predict(
    test_ds,
    verbose=1
)

# Attach predictions to their original video names.
prediction_df = test_df[
    ["video_name", "generator"]
].copy()

for index, class_name in enumerate(class_names):
    prediction_df[f"prob_{index}"] = frame_probabilities[:, index]

# Average frame probabilities for each video.
probability_columns = [
    f"prob_{index}" for index in range(len(class_names))
]

for video_name, group in prediction_df.groupby("video_name", sort=False):

    true_class = group["generator"].iloc[0]

    average_probabilities = group[probability_columns].mean().values

    predicted_index = int(np.argmax(average_probabilities))
    predicted_class = class_names[predicted_index]

    video_predictions.append({
        "video_name": video_name,
        "true_generator": true_class,
        "predicted_generator": predicted_class,
        "correct": true_class == predicted_class
    })

video_results_df = pd.DataFrame(video_predictions)

video_accuracy = accuracy_score(
    video_results_df["true_generator"],
    video_results_df["predicted_generator"]
)

print("\nVideo-Level Accuracy:", round(video_accuracy * 100, 2), "%")

print("\nVideo-Level Predictions:")
print(video_results_df.to_string(index=False))

print("\nVideo-Level Classification Report:")

print(
    classification_report(
        video_results_df["true_generator"],
        video_results_df["predicted_generator"],
        labels=class_names,
        target_names=class_names,
        zero_division=0
    )
)

print("\nVideo-Level Confusion Matrix:")

video_matrix = confusion_matrix(
    video_results_df["true_generator"],
    video_results_df["predicted_generator"],
    labels=class_names
)

print(video_matrix)

# Save video-level predictions.
video_results_df.to_csv(
    RESULTS_DIR / "video_level_predictions.csv",
    index=False
)

final_model_path = MODEL_DIR / "video_generator_model.keras"

model.save(final_model_path)

with open(
    MODEL_DIR / "class_names.json",
    "w"
) as file:
    json.dump(class_names, file, indent=4)

# ==========================================================
# 13. SAVE EVALUATION RESULTS
# ==========================================================

results = {

    "experiment": "Video Generator Classification",

    "model": "MobileNetV2",

    "test_accuracy": float(accuracy),

    "test_loss": float(test_loss),

    "classes": class_names,

    "training_frames": int(len(train_df)),

    "validation_frames": int(len(val_df)),

    "testing_frames": int(len(test_df)),

    "training_source_ids": sorted(train_ids),

    "validation_source_ids": sorted(val_ids),

    "testing_source_ids": sorted(test_ids),

    "classification_report": report,

    "confusion_matrix": matrix.tolist(),

    "dataset_note": (
        "All samples are generated-video frames. "
        "This experiment does not establish real-versus-fake "
        "detection performance."
    )
}

with open(
    MODEL_DIR / "training_results.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

# ==========================================================
# 14. CONFUSION MATRIX GRAPH
# ==========================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(11, 9))

display.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    colorbar=True
)

plt.title("Video Generator Classification - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "video_confusion_matrix.png",
    dpi=300
)

plt.close()

# ==========================================================
# 15. TRAINING ACCURACY GRAPH
# ==========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Video Model Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "video_training_accuracy.png",
    dpi=300
)

plt.close()

# ==========================================================
# 16. TRAINING LOSS GRAPH
# ==========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Video Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "video_training_loss.png",
    dpi=300
)

plt.close()

# ==========================================================
# 17. FINISHED
# ==========================================================

print("\n" + "=" * 55)
print("VIDEO GENERATOR TRAINING COMPLETED")
print("=" * 55)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nSaved files:")
print("Best model:", best_model_path)
print("Final model:", final_model_path)
print("Class names:", MODEL_DIR / "class_names.json")
print("Results:", MODEL_DIR / "training_results.json")

print("\nGraphs:")
print(GRAPH_DIR / "video_confusion_matrix.png")
print(GRAPH_DIR / "video_training_accuracy.png")
print(GRAPH_DIR / "video_training_loss.png")

print("\nExperiment limitation:")
print("The model classifies seven video generators.")
print("It is not a validated real-vs-fake detector.")
