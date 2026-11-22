
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------
# 1. Load test dataset
# ------------------------------------------------

test_df = pd.read_csv("dataset/test.csv")

X_test = test_df["title"].fillna("").astype(str)
y_test = test_df["label"].astype(int).values

print("Test records:", len(test_df))

# ------------------------------------------------
# 2. Load CNN model
# ------------------------------------------------

print("\nLoading CNN model...")

model = load_model("model/fake_news_cnn.keras")

# ------------------------------------------------
# 3. Load tokenizer and configuration
# ------------------------------------------------

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/config.pkl", "rb") as f:
    config = pickle.load(f)

MAX_LENGTH = config["max_length"]

# ------------------------------------------------
# 4. Apply original CNN preprocessing
# ------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

X_clean = X_test.apply(clean_text)

# ------------------------------------------------
# 5. Tokenize and pad
# ------------------------------------------------

sequences = tokenizer.texts_to_sequences(X_clean)

X_padded = pad_sequences(
    sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("Input shape:", X_padded.shape)

# ------------------------------------------------
# 6. Predict
# ------------------------------------------------

probabilities = model.predict(X_padded, verbose=1)

predictions = (probabilities >= 0.5).astype(int).flatten()

# ------------------------------------------------
# 7. Evaluate
# ------------------------------------------------

print("\n===================================")
print("CNN TEST RESULTS")
print("===================================")

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

print(
    "Macro F1:",
    round(f1_score(y_test, predictions, average="macro"), 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# ------------------------------------------------
# 8. Save predictions for comparison
# ------------------------------------------------

results = test_df.copy()

results["cnn_probability_real"] = probabilities.flatten()
results["cnn_prediction"] = predictions

results.to_csv(
    "cnn_test_predictions.csv",
    index=False
)

print("\nPredictions saved to cnn_test_predictions.csv")
