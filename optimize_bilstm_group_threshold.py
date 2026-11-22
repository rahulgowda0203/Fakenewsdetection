import pickle
import re
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Loading group validation data...")

df = pd.read_csv("dataset/group_validation.csv")
df = df.dropna(subset=["title", "label"])

texts = df["title"].apply(clean_text).tolist()
y_true = df["label"].astype(int).to_numpy()

with open("model/tokenizer_bilstm_group.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/config_bilstm_group.pkl", "rb") as f:
    config = pickle.load(f)

model = tf.keras.models.load_model(
    "model/fake_news_bilstm_group.keras"
)

X = pad_sequences(
    tokenizer.texts_to_sequences(texts),
    maxlen=config["max_length"],
    padding="post",
    truncating="post"
)

print("Predicting validation probabilities...")

real_probs = model.predict(X, batch_size=32, verbose=1).ravel()
fake_probs = 1 - real_probs

results = []

for threshold in np.arange(0.20, 0.81, 0.025):
    y_pred = (fake_probs >= threshold).astype(int)
    y_pred = 1 - y_pred

    results.append({
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "fake_precision": precision_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "fake_recall": recall_score(
            y_true, y_pred, pos_label=0, zero_division=0
        ),
        "fake_f1": f1_score(
            y_true, y_pred, pos_label=0, zero_division=0
        )
    })

results_df = pd.DataFrame(results)
results_df.to_csv(
    "bilstm_group_threshold_results.csv",
    index=False
)

best = results_df.loc[results_df["macro_f1"].idxmax()]

print("\nTHRESHOLD OPTIMIZATION RESULTS")
print("=" * 65)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nBEST THRESHOLD (by validation macro F1)")
print(best.to_string())

config["threshold"] = float(best["threshold"])

with open("model/config_bilstm_group.pkl", "wb") as f:
    pickle.dump(config, f)

print("\nSelected threshold:", config["threshold"])
print("Updated model/config_bilstm_group.pkl")
print("Results saved: bilstm_group_threshold_results.csv")
