import os
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional,
    Dense, Dropout, SpatialDropout1D, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("model", exist_ok=True)

MAX_WORDS = 10000
MAX_LENGTH = 300
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10

MODEL_PATH = "model/fake_news_bilstm_group.keras"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Loading group-aware datasets...")

train_df = pd.read_csv("dataset/group_train.csv")
val_df = pd.read_csv("dataset/group_validation.csv")

train_df = train_df.dropna(subset=["title", "label"])
val_df = val_df.dropna(subset=["title", "label"])

train_texts = train_df["title"].apply(clean_text).tolist()
val_texts = val_df["title"].apply(clean_text).tolist()

y_train = train_df["label"].astype(int).to_numpy()
y_val = val_df["label"].astype(int).to_numpy()

print("Training records:", len(train_texts))
print("Validation records:", len(val_texts))

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

print("Class weights:", class_weights)

print("\nPreparing tokenizer...")

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

print("Training input shape:", X_train.shape)
print("Validation input shape:", X_val.shape)

print("\nBuilding BiLSTM model...")

model = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(MAX_WORDS, EMBEDDING_DIM),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
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

print("\nStarting group-aware BiLSTM training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Save tokenizer separately
with open("model/tokenizer_bilstm_group.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save model configuration
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

with open("model/config_bilstm_group.pkl", "wb") as f:
    pickle.dump(config, f)

print("\nGroup-aware BiLSTM training completed.")
print("Model saved:", MODEL_PATH)
print("Tokenizer saved: model/tokenizer_bilstm_group.pkl")
print("Configuration saved: model/config_bilstm_group.pkl")
