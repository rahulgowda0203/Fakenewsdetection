
import pickle
import re
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from tensorflow.keras.preprocessing.sequence import pad_sequences


SEED = 42


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("Loading training data...")
df = pd.read_csv("dataset/train.csv")
df = df.dropna(subset=["title", "label"])

texts = df["title"].apply(clean_text).tolist()
labels = df["label"].astype(int).to_numpy()

# Recreate exactly the same validation split.
train_texts, val_texts, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)

print("Validation records:", len(val_texts))

print("\nLoading weighted CNN...")
model = tf.keras.models.load_model(
    "model/fake_news_cnn_weighted.keras"
)

with open("model/tokenizer_weighted.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/config_weighted.pkl", "rb") as f:
    config = pickle.load(f)

X_val = pad_sequences(
    tokenizer.texts_to_sequences(val_texts),
    maxlen=config["max_length"],
    padding="post",
    truncating="post"
)

print("\nPredicting validation probabilities...")
probabilities = model.predict(X_val, verbose=1).ravel()

# Label 0 = Fake News; label 1 = Real News.
# A lower threshold predicts more records as fake.
thresholds = np.arange(0.20, 0.801, 0.01)

results = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    fake_f1 = f1_score(
        y_val, predictions, pos_label=0, zero_division=0
    )

    real_f1 = f1_score(
        y_val, predictions, pos_label=1, zero_division=0
    )

    macro_f1 = f1_score(
        y_val, predictions, average="macro", zero_division=0
    )

    results.append({
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy_score(y_val, predictions),
        "macro_f1": macro_f1,
        "fake_precision": precision_score(
            y_val, predictions, pos_label=0, zero_division=0
        ),
        "fake_recall": recall_score(
            y_val, predictions, pos_label=0, zero_division=0
        ),
        "fake_f1": fake_f1,
        "real_f1": real_f1
    })

results_df = pd.DataFrame(results)

# Select threshold with the highest validation macro F1.
best = results_df.loc[results_df["macro_f1"].idxmax()]

print("\n===== BEST VALIDATION THRESHOLD =====")
print(best.to_string())

print("\n===== TOP 10 THRESHOLDS BY MACRO F1 =====")
print(
    results_df.sort_values("macro_f1", ascending=False)
    .head(10)
    .to_string(index=False)
)

# Save the threshold separately for reproducibility.
config["threshold"] = float(best["threshold"])

with open("model/config_weighted.pkl", "wb") as f:
    pickle.dump(config, f)

results_df.to_csv(
    "weighted_threshold_validation_results.csv",
    index=False
)

print("\nUpdated threshold:", config["threshold"])
print("Saved: weighted_threshold_validation_results.csv")
