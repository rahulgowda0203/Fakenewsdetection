
import pickle
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("Loading test dataset...")
df = pd.read_csv("dataset/test.csv")
df = df.dropna(subset=["title", "label"])

texts = df["title"].apply(clean_text).tolist()
y_true = df["label"].astype(int).to_numpy()

print("Test records:", len(texts))

print("\nLoading weighted CNN model...")
model = tf.keras.models.load_model(
    "model/fake_news_cnn_weighted.keras"
)

with open("model/tokenizer_weighted.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/config_weighted.pkl", "rb") as f:
    config = pickle.load(f)

sequences = tokenizer.texts_to_sequences(texts)

X_test = pad_sequences(
    sequences,
    maxlen=config["max_length"],
    padding="post",
    truncating="post"
)

print("\nMaking predictions...")
probabilities = model.predict(X_test, verbose=1).ravel()

y_pred = (
    probabilities >= config["threshold"]
).astype(int)

print("\n===== CLASS-WEIGHTED CNN RESULTS =====")

print(
    "Accuracy:",
    round(accuracy_score(y_true, y_pred), 4)
)

print(
    "Macro F1:",
    round(f1_score(y_true, y_pred, average="macro"), 4)
)

print("\nClassification Report:")

print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Fake News", "Real News"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

df["predicted_label"] = y_pred
df["fake_probability"] = 1 - probabilities
df["real_probability"] = probabilities

df.to_csv(
    "cnn_weighted_test_predictions.csv",
    index=False
)

print("\nSaved: cnn_weighted_test_predictions.csv")
