
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

# 1. Clean headline text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 2. Load test dataset
df = pd.read_csv("dataset/test.csv")
df = df.dropna(subset=["title", "label"])

texts = df["title"].apply(clean_text).tolist()
y_test = df["label"].astype(int).to_numpy()

print("Test records:", len(df))

# 3. Load trained model
model = tf.keras.models.load_model(
    "model/fake_news_bilstm.keras"
)

# 4. Load tokenizer
with open("model/tokenizer_bilstm.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 5. Load configuration
with open("model/config_bilstm.pkl", "rb") as f:
    config = pickle.load(f)

max_length = config["max_length"]
threshold = config.get("threshold", 0.5)

print("Maximum sequence length:", max_length)
print("Classification threshold:", threshold)

# 6. Convert headlines into padded sequences
sequences = tokenizer.texts_to_sequences(texts)

X_test = pad_sequences(
    sequences,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

# 7. Predict
print("\nRunning BiLSTM predictions...")

probabilities = model.predict(
    X_test,
    batch_size=32,
    verbose=1
).ravel()

y_pred = (probabilities >= threshold).astype(int)

# 8. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n========== BiLSTM TEST RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1-score: {macro_f1 * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# 9. Save predictions
results = df[["title", "label"]].copy()
results["predicted_label"] = y_pred
results["fake_probability"] = 1 - probabilities
results["real_probability"] = probabilities

results.to_csv("bilstm_test_predictions.csv", index=False)

print("\nPredictions saved to bilstm_test_predictions.csv")
print("\nBiLSTM evaluation completed successfully!")
