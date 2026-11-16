import os
import re
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    Dropout,
    Dense
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_FOLDER = "dataset"
MODEL_FOLDER = "model"

MAX_WORDS = 10000
MAX_LENGTH = 300

EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10


# ============================================================
# 2. CREATE MODEL FOLDER
# ============================================================

os.makedirs(MODEL_FOLDER, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

files = {
    "politifact_fake.csv": 0,
    "politifact_real.csv": 1,
    "gossipcop_fake.csv": 0,
    "gossipcop_real.csv": 1
}

all_data = []

print("\nLoading FakeNewsNet dataset...\n")

for filename, label in files.items():

    filepath = os.path.join(DATASET_FOLDER, filename)

    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found")
        continue

    print(f"Loading: {filename}")

    df = pd.read_csv(filepath)

    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))

    # --------------------------------------------------------
    # Find text columns
    # --------------------------------------------------------

    title_column = None
    text_column = None

    for column in df.columns:

        column_lower = column.lower()

        if column_lower == "title":
            title_column = column

        if column_lower in ["text", "article", "content", "body"]:
            text_column = column

    # --------------------------------------------------------
    # Create text
    # --------------------------------------------------------

    if title_column and text_column:

        df["news_text"] = (
            df[title_column].fillna("").astype(str)
            + " "
            + df[text_column].fillna("").astype(str)
        )

    elif title_column:

        df["news_text"] = df[title_column].fillna("").astype(str)

    elif text_column:

        df["news_text"] = df[text_column].fillna("").astype(str)

    else:

        # Try first string column
        string_columns = df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        if len(string_columns) > 0:

            df["news_text"] = df[string_columns[0]].fillna("").astype(str)

        else:

            print(f"Could not find text column in {filename}")
            continue

    # --------------------------------------------------------
    # Add label
    # --------------------------------------------------------

    df["label"] = label

    all_data.append(
        df[["news_text", "label"]]
    )


# ============================================================
# 4. COMBINE DATA
# ============================================================

if len(all_data) == 0:
    raise Exception(
        "\nNo dataset files were found.\n"
        "Make sure the CSV files are inside the dataset folder."
    )

data = pd.concat(
    all_data,
    ignore_index=True
)

print("\n======================================")
print("Dataset loaded successfully")
print("======================================")

print("Total samples:", len(data))

print("\nClass distribution:")
print(data["label"].value_counts())

print("\n0 = FAKE")
print("1 = REAL")


# ============================================================
# 5. CLEAN TEXT
# ============================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


print("\nCleaning text...")

data["news_text"] = data["news_text"].apply(clean_text)

# Remove empty rows
data = data[data["news_text"].str.len() > 0]

# Remove duplicate articles
data = data.drop_duplicates(
    subset=["news_text"]
)

data = data.reset_index(drop=True)

print("Samples after cleaning:", len(data))


# ============================================================
# 6. X AND Y
# ============================================================

X = data["news_text"].values
y = data["label"].values


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. TOKENIZATION
# ============================================================

print("\nCreating tokenizer...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(
    X_train
)

X_test_sequences = tokenizer.texts_to_sequences(
    X_test
)


# ============================================================
# 9. PADDING
# ============================================================

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("\nTraining input shape:", X_train_padded.shape)
print("Testing input shape:", X_test_padded.shape)


# ============================================================
# 10. CNN MODEL
# ============================================================

print("\nCreating CNN model...")

model = Sequential([

    Input(shape=(MAX_LENGTH,)),

    Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM
    ),

    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    ),

    MaxPooling1D(
        pool_size=2
    ),

    Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu"
    ),

    GlobalMaxPooling1D(),

    Dropout(0.5),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 12. MODEL SUMMARY
# ============================================================

print("\n======================================")
print("CNN MODEL SUMMARY")
print("======================================")

model.summary()


# ============================================================
# 13. CALLBACKS
# ============================================================

model_path = os.path.join(
    MODEL_FOLDER,
    "fake_news_cnn.keras"
)

checkpoint = ModelCheckpoint(
    model_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print("\n======================================")
print("STARTING TRAINING")
print("======================================\n")

history = model.fit(
    X_train_padded,
    y_train,
    validation_split=0.20,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        checkpoint,
        early_stopping
    ],
    verbose=1
)


# ============================================================
# 15. EVALUATE MODEL
# ============================================================

print("\n======================================")
print("EVALUATING MODEL")
print("======================================")

test_loss, test_accuracy = model.evaluate(
    X_test_padded,
    y_test,
    verbose=0
)

print("\nTest Loss:", test_loss)

print(
    "Test Accuracy:",
    test_accuracy * 100,
    "%"
)


# ============================================================
# 16. PREDICTIONS
# ============================================================

y_probability = model.predict(
    X_test_padded,
    verbose=0
)

y_prediction = (
    y_probability >= 0.5
).astype(int).flatten()


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        y_prediction,
        target_names=[
            "Fake News",
            "Real News"
        ]
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print(
    confusion_matrix(
        y_test,
        y_prediction
    )
)


# ============================================================
# 19. SAVE TOKENIZER
# ============================================================

tokenizer_path = os.path.join(
    MODEL_FOLDER,
    "tokenizer.pkl"
)

with open(
    tokenizer_path,
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )

print("\nTokenizer saved to:")
print(tokenizer_path)


# ============================================================
# 20. SAVE CONFIGURATION
# ============================================================

config = {
    "max_words": MAX_WORDS,
    "max_length": MAX_LENGTH,
    "embedding_dim": EMBEDDING_DIM
}

config_path = os.path.join(
    MODEL_FOLDER,
    "config.pkl"
)

with open(
    config_path,
    "wb"
) as file:

    pickle.dump(
        config,
        file
    )


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n======================================")
print("TRAINING COMPLETED")
print("======================================")

print("\nSaved files:")

print(
    "Model:",
    model_path
)

print(
    "Tokenizer:",
    tokenizer_path
)

print(
    "Config:",
    config_path
)