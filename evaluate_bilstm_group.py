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

print("Loading untouched test dataset...")

df = pd.read_csv("dataset/test.csv")
df = df.dropna(subset=["title", "label"])

texts = df["title"].apply(clean_text).tolist()
y_test = df["label"].astype(int).to_numpy()

with open("model/tokenizer_bilstm_group.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/config_bilstm_group.pkl", "rb") as f:
    config = pickle.load(f)

model = tf.keras.models.load_model(
    "model/fake_news_bilstm_group.keras"
)

threshold = config["threshold"]
print("Test records:", len(df))
print("Selected fake probability threshold:", threshold)

X_test = pad_sequences(
    tokenizer.texts_to_sequences(texts),
    maxlen=config["max_length"],
    padding="post",
    truncating="post"
)

print("\nRunning final predictions...")

real_probabilities = model.predict(
    X_test,
    batch_size=32,
    verbose=1
).ravel()

fake_probabilities = 1 - real_probabilities

# Label 0 = Fake News; label 1 = Real News
y_pred = (fake_probabilities < threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n========== GROUP-AWARE BiLSTM TEST RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1-score: {macro_f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Fake News", "Real News"],
    digits=4,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

results = df[["title", "label"]].copy()
results["predicted_label"] = y_pred
results["fake_probability"] = fake_probabilities
results["real_probability"] = real_probabilities

results.to_csv("bilstm_group_test_predictions.csv", index=False)

print("\nPredictions saved to bilstm_group_test_predictions.csv")
print("Final evaluation completed.")
