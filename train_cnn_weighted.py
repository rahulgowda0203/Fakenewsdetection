
import os
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, Dense, Dropout, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("model", exist_ok=True)

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

print("Loading training dataset...")
df = pd.read_csv("dataset/train.csv")
df = df.dropna(subset=["title", "label"])

texts = df["title"].apply(clean_text).tolist()
labels = df["label"].astype(int).to_numpy()

train_texts, val_texts, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)

print("Training records:", len(train_texts))
print("Validation records:", len(val_texts))

# Calculate class weights from training subset only.
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

print("Class weights:", class_weights)

# Tokenizer fitted only on training text.
tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)
tokenizer.fit_on_texts(train_texts)

X_train = pad_sequences(
    tokenizer.texts_to_sequences(train_texts),
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_val = pad_sequences(
    tokenizer.texts_to_sequences(val_texts),
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

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

MODEL_PATH = "model/fake_news_cnn_weighted.keras"

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

print("\nStarting weighted CNN training...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

with open("model/tokenizer_weighted.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

config = {
    "max_length": MAX_LENGTH,
    "max_words": MAX_WORDS,
    "embedding_dim": EMBEDDING_DIM,
    "threshold": 0.5,
    "labels": {0: "Fake News", 1: "Real News"}
}

with open("model/config_weighted.pkl", "wb") as f:
    pickle.dump(config, f)

print("\nWeighted CNN training completed.")
print("Model saved:", MODEL_PATH)
