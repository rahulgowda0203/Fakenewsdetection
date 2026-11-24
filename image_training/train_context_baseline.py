
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets/image/verite_source/VERITE"
OUTPUT_DIR = ROOT / "models/image/context_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["true", "miscaptioned", "out-of-context"]
LABEL_MAP = {name: i for i, name in enumerate(CLASSES)}

df = pd.read_csv(DATA_DIR / "VERITE.csv")
image = np.load(DATA_DIR / "VERITE_clip_image_embeddings_ViTL14.npy")
text = np.load(DATA_DIR / "VERITE_clip_text_embeddings_ViTL14.npy")

assert len(df) == len(image) == len(text)

# Normalize embeddings
image = image / np.maximum(
    np.linalg.norm(image, axis=1, keepdims=True), 1e-8
)
text = text / np.maximum(
    np.linalg.norm(text, axis=1, keepdims=True), 1e-8
)

# Construct a smaller feature set
cosine = np.sum(image * text, axis=1, keepdims=True)
difference = np.abs(image - text)

X = np.concatenate([cosine, difference], axis=1)
y = df["label"].map(LABEL_MAP).to_numpy()
groups = df["image_path"].astype(str).to_numpy()

# Same grouped split strategy and random seed
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_val, test = next(split1.split(X, y, groups))

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_rel, val_rel = next(
    split2.split(
        X[train_val],
        y[train_val],
        groups[train_val]
    )
)

train = train_val[train_rel]
val = train_val[val_rel]

X_train, X_val, X_test = X[train], X[val], X[test]
y_train, y_val, y_test = y[train], y[val], y[test]

assert set(groups[train]).isdisjoint(set(groups[val]))
assert set(groups[train]).isdisjoint(set(groups[test]))
assert set(groups[val]).isdisjoint(set(groups[test]))

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=3000
        ))
    ]),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=0.1,
            kernel="linear",
            class_weight="balanced"
        ))
    ])
}

results = {}
best_name = None
best_validation_accuracy = -1

for name, model in models.items():
    print(f"\n========== {name.upper()} ==========")

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)

    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

    results[name] = {
        "validation_accuracy": float(val_accuracy)
    }

    if val_accuracy > best_validation_accuracy:
        best_validation_accuracy = val_accuracy
        best_name = name

# Select model using validation data only
best_model = models[best_name]

# Evaluate selected model once on the held-out test set
test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

print("\n========== SELECTED MODEL ==========")
print("Model:", best_name)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(
    y_test,
    test_pred,
    labels=[0, 1, 2],
    target_names=CLASSES,
    zero_division=0
))

cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2])
print("\nConfusion matrix:")
print(cm)

joblib.dump(best_model, OUTPUT_DIR / "context_baseline.pkl")

results["selected_model"] = best_name
results["test_accuracy"] = float(test_accuracy)
results["confusion_matrix"] = cm.tolist()
results["classes"] = CLASSES

with open(OUTPUT_DIR / "baseline_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved model:", OUTPUT_DIR / "context_baseline.pkl")
print("Saved results:", OUTPUT_DIR / "baseline_results.json")
