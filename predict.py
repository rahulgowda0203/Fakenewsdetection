import os
import pickle
import re
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# 1. PATHS
# ============================================================

MODEL_PATH = "model/fake_news_cnn.keras"
TOKENIZER_PATH = "model/tokenizer.pkl"
CONFIG_PATH = "model/config.pkl"


# ============================================================
# 2. CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run train.py first."
    )

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        "Tokenizer not found. Run train.py first."
    )

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        "Config not found. Run train.py first."
    )


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\nLoading CNN model...")

model = load_model(
    MODEL_PATH
)


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

with open(
    TOKENIZER_PATH,
    "rb"
) as file:

    tokenizer = pickle.load(
        file
    )


# ============================================================
# 5. LOAD CONFIG
# ============================================================

with open(
    CONFIG_PATH,
    "rb"
) as file:

    config = pickle.load(
        file
    )


MAX_LENGTH = config["max_length"]


# ============================================================
# 6. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================

def predict_news(news_text):

    # Clean text
    cleaned_text = clean_text(
        news_text
    )

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Prediction
    probability = model.predict(
        padded,
        verbose=0
    )[0][0]

    # Classification
    if probability >= 0.5:

        result = "REAL NEWS"
        confidence = probability * 100

    else:

        result = "FAKE NEWS"
        confidence = (1 - probability) * 100

    return result, confidence


# ============================================================
# 8. USER INPUT
# ============================================================

print("\n======================================")
print("FAKE NEWS DETECTION - CNN")
print("======================================")

print("\nEnter a news article.")
print("Type 'exit' to stop.\n")


while True:

    news = input(
        "Enter news: "
    )

    if news.lower() == "exit":
        print("\nProgram ended.")
        break

    if not news.strip():

        print(
            "Please enter some news text."
        )

        continue

    result, confidence = predict_news(
        news
    )

    print("\n--------------------------------------")
    print("Prediction:", result)
    print(
        f"Confidence: {confidence:.2f}%"
    )
    print("--------------------------------------\n")