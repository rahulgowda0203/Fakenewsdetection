from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

# Run this script from your FakeNewsDetection project root.
BASE_DIR = Path(__file__).resolve().parent

TEST_CSV = BASE_DIR / "dataset" / "test.csv"
MODEL_PATH = BASE_DIR / "model" / "fake_news_bilstm.keras"
TOKENIZER_PATH = BASE_DIR / "model" / "tokenizer.pkl"
OUTPUT_IMAGE = BASE_DIR / "bilstm_confusion_matrix.png"

# Load test data
df = pd.read_csv(TEST_CSV)
if "title" not in df.columns or "label" not in df.columns:
    raise ValueError("test.csv must contain 'title' and 'label' columns.")

texts = df["title"].fillna("").astype(str).tolist()
y_true = df["label"].astype(int).to_numpy()

# Load the exact tokenizer used during training
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load trained BiLSTM
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Use the model's expected sequence length
input_shape = model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]
max_len = input_shape[1]
if max_len is None:
    max_len = 300

sequences = tokenizer.texts_to_sequences(texts)
X_test = pad_sequences(
    sequences,
    maxlen=int(max_len),
    padding="post",
    truncating="post",
)

# Predict Fake (0) / Real (1)
probabilities = model.predict(X_test, batch_size=64, verbose=1).reshape(-1)
y_pred = (probabilities >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
accuracy = accuracy_score(y_true, y_pred)
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
)

print("\nBiLSTM Confusion Matrix (rows=actual, columns=predicted)")
print("                 Predicted Fake   Predicted Real")
print(f"Actual Fake          {cm[0,0]:>6}           {cm[0,1]:>6}")
print(f"Actual Real          {cm[1,0]:>6}           {cm[1,1]:>6}")
print(f"\nAccuracy:        {accuracy * 100:.2f}%")
print(f"Macro precision: {macro_precision * 100:.2f}%")
print(f"Macro recall:    {macro_recall * 100:.2f}%")
print(f"Macro F1-score:  {macro_f1 * 100:.2f}%")
print("\nClass mapping: 0 = Fake, 1 = Real")
print(classification_report(
    y_true, y_pred, labels=[0, 1], target_names=["Fake", "Real"], zero_division=0
))

fig, ax = plt.subplots(figsize=(8, 6))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake", "Real"],
)
display.plot(ax=ax, cmap="Greens", values_format=",d", colorbar=True)
ax.set_title("BiLSTM Confusion Matrix", fontsize=16, weight="bold", pad=14)
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
fig.tight_layout()
fig.savefig(OUTPUT_IMAGE, dpi=220, bbox_inches="tight")
print(f"\nSaved confusion matrix image: {OUTPUT_IMAGE}")
