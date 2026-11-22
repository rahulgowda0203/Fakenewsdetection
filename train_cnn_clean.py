
import os
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    Input
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
DATA_PATH = "dataset/train.csv"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# Configuration
MAX_WORDS = 10000
MAX_LENGTH = 300
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Loading leakage-safe training dataset...")
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["title", "label"])
texts = df["title"].apply(clean_text).tolist()
labels = df["label"].astype(int).to_numpy()

print("Training records:", len(texts))
print("Label distribution:")
print(pd.Series(labels).value_counts().sort_index())

# Create validation split from training data only.
# The final test.csv is not accessed by this script.
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)

print("Training subset:", len(train_texts))
print("Validation subset:", len(val_texts))

# Fit tokenizer only on training subset.
tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)

X_train = pad_sequences(
    train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_val = pad_sequences(
    val_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

# CNN architecture
model = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(MAX_WORDS, EMBEDDING_DIM),
    Conv1D(128, 5, activation="relu"),
    MaxPooling1D(2),
    Conv1D(64, 5, activation="relu"),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fake_news_cnn_leakage_safe.keras"
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )
]

print("\nStarting CNN training...")

history = model.fit(
    X_train,
    train_labels,
    validation_data=(X_val, val_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save tokenizer and configuration separately.
with open(
    os.path.join(MODEL_DIR, "tokenizer_leakage_safe.pkl"),
    "wb"
) as f:
    pickle.dump(tokenizer, f)

config = {
    "max_length": MAX_LENGTH,
    "max_words": MAX_WORDS,
    "embedding_dim": EMBEDDING_DIM,
    "threshold": 0.5,
    "labels": {
        0: "Fake News",
        1: "Real News"
    }
}

with open(
    os.path.join(MODEL_DIR, "config_leakage_safe.pkl"),
    "wb"
) as f:
    pickle.dump(config, f)

print("\nTraining completed.")
print("Model saved:", MODEL_PATH)
print("Tokenizer and configuration saved.")
print("Final test dataset has not been used.")
