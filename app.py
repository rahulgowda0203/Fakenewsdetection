
from pathlib import Path
import pickle
import re
import sqlite3
import uuid
from datetime import datetime

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing.sequence import pad_sequences
from werkzeug.utils import secure_filename


# =====================================================
# APPLICATION CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATABASE = BASE_DIR / "prediction_history.db"

app = Flask(__name__)

UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_UPLOAD_DIR = UPLOAD_DIR / "audio"
IMAGE_UPLOAD_DIR = UPLOAD_DIR / "images"

AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

ALLOWED_AUDIO = {"wav", "flac"}
ALLOWED_IMAGES = {"jpg", "jpeg", "png", "webp", "bmp"}


# =====================================================
# DATABASE INITIALIZATION
# =====================================================

def init_db():
    conn = sqlite3.connect(DATABASE)

    try:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                headline TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL,
                fake_probability REAL,
                real_probability REAL,
                model_name TEXT,
                created_at TEXT
            )
        """)

        conn.commit()

    finally:
        conn.close()


def save_prediction(headline, prediction):
    conn = sqlite3.connect(DATABASE)

    try:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions (
                headline,
                result,
                confidence,
                fake_probability,
                real_probability,
                model_name,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            headline,
            prediction["result"],
            prediction["confidence"],
            prediction["fake_probability"],
            prediction["real_probability"],
            prediction["model_name"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()

    finally:
        conn.close()


init_db()


# =====================================================
# LOAD LOGISTIC REGRESSION MODEL
# =====================================================

with open(MODEL_DIR / "logistic_regression_group.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open(MODEL_DIR / "tfidf_vectorizer_group.pkl", "rb") as f:
    lr_vectorizer = pickle.load(f)

LR_THRESHOLD = 0.325


# =====================================================
# LOAD CNN MODEL
# =====================================================

CNN_MODEL_PATH = MODEL_DIR / "fake_news_cnn.keras"
CNN_TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
CNN_CONFIG_PATH = MODEL_DIR / "config.pkl"

cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

with open(CNN_TOKENIZER_PATH, "rb") as f:
    cnn_tokenizer = pickle.load(f)

with open(CNN_CONFIG_PATH, "rb") as f:
    cnn_config = pickle.load(f)

CNN_THRESHOLD = 0.5

print("CNN model loaded successfully!")


# =====================================================
# LOAD BiLSTM MODEL
# =====================================================

bilstm_model = tf.keras.models.load_model(
    MODEL_DIR / "fake_news_bilstm_group.keras"
)

with open(MODEL_DIR / "tokenizer_bilstm_group.pkl", "rb") as f:
    bilstm_tokenizer = pickle.load(f)

with open(MODEL_DIR / "config_bilstm_group.pkl", "rb") as f:
    bilstm_config = pickle.load(f)

BILSTM_THRESHOLD = float(bilstm_config["threshold"])

print("BiLSTM model loaded successfully!")


# =====================================================
# TEXT PREPROCESSING
# =====================================================

def clean_text(text):

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# CNN PREDICTION
# =====================================================

def predict_cnn(headline):

    cleaned = clean_text(headline)

    sequence = cnn_tokenizer.texts_to_sequences(
        [cleaned]
    )

    padded = pad_sequences(
        sequence,
        maxlen=int(cnn_config["max_length"]),
        padding="post",
        truncating="post"
    )

    output = cnn_model.predict(
        padded,
        verbose=0
    )

    class_1_probability = float(output[0][0])

    # Assumes training labels:
    # 0 = Fake News
    # 1 = Real News
    real_probability = class_1_probability
    fake_probability = 1.0 - real_probability

    return fake_probability, real_probability


# =====================================================
# NEWS PREDICTION FUNCTION
# =====================================================

def predict_news(headline, selected_model):

    headline = str(headline).strip()

    if not headline:
        raise ValueError("Please enter a news headline.")

    if selected_model == "cnn":

        fake_probability, real_probability = predict_cnn(
            headline
        )

        threshold = CNN_THRESHOLD
        model_name = "CNN Deep Learning"

    elif selected_model == "bilstm":

        cleaned = clean_text(headline)

        sequence = bilstm_tokenizer.texts_to_sequences(
            [cleaned]
        )

        padded = pad_sequences(
            sequence,
            maxlen=int(bilstm_config["max_length"]),
            padding="post",
            truncating="post"
        )

        real_probability = float(
            bilstm_model.predict(
                padded,
                verbose=0
            )[0][0]
        )

        fake_probability = 1.0 - real_probability

        threshold = BILSTM_THRESHOLD
        model_name = "BiLSTM Deep Learning"

    elif selected_model == "logistic":

        features = lr_vectorizer.transform([headline])

        probabilities = lr_model.predict_proba(features)[0]

        class_indices = {
            label: index
            for index, label in enumerate(lr_model.classes_)
        }

        fake_probability = float(
            probabilities[class_indices[0]]
        )

        real_probability = float(
            probabilities[class_indices[1]]
        )

        threshold = LR_THRESHOLD
        model_name = "Logistic Regression"

    else:
        raise ValueError("Please select a valid model.")

    if fake_probability >= threshold:
        result = "Fake News"
    else:
        result = "Real News"

    confidence = (
        fake_probability if result == "Fake News"
        else real_probability
    ) * 100

    return {
        "result": result,
        "confidence": round(confidence, 2),
        "fake_probability": round(fake_probability * 100, 2),
        "real_probability": round(real_probability * 100, 2),
        "model_name": model_name,
        "threshold": threshold * 100
    }


# =====================================================
# HOME PAGE - CNN NEWS DETECTION
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    headline = ""
    selected_model = "cnn"
    error = None

    if request.method == "POST":

        headline = request.form.get(
            "headline", ""
        ).strip()

        # CNN is always selected automatically.
        selected_model = "cnn"

        try:

            prediction = predict_news(
                headline,
                selected_model
            )

            save_prediction(
                headline,
                prediction
            )

        except Exception as exc:
            app.logger.exception("News prediction failed")
            error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        headline=headline,
        selected_model=selected_model,
        error=error
    )


# =====================================================
# AUDIO DEEPFAKE DETECTION
# =====================================================

@app.route("/audio", methods=["GET", "POST"])
def audio_detection():

    prediction = None
    error = None

    if request.method == "POST":

        uploaded_file = request.files.get("audio")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select an audio file."

        else:

            original_name = secure_filename(
                uploaded_file.filename
            )

            extension = original_name.rsplit(".", 1)[-1].lower()

            if extension not in ALLOWED_AUDIO:
                error = "Please upload a WAV or FLAC audio file."

            else:

                filename = f"{uuid.uuid4().hex}.{extension}"
                file_path = AUDIO_UPLOAD_DIR / filename

                try:

                    uploaded_file.save(file_path)

                    from audio_detector import detect_audio

                    prediction = detect_audio(str(file_path))

                except Exception as exc:

                    app.logger.exception("Audio analysis failed")
                    error = f"Audio analysis failed: {exc}"

                finally:

                    if file_path.exists():
                        file_path.unlink()

    return render_template(
        "audio.html",
        prediction=prediction,
        error=error
    )


# =====================================================
# IMAGE AUTHENTICITY AND CONTEXT DETECTION
# =====================================================

@app.route("/image", methods=["GET", "POST"])
def image_detection():

    authenticity_result = None
    context_result = None
    error = None

    caption = ""
    selected_mode = "authenticity"

    if request.method == "POST":

        uploaded_file = request.files.get("image")

        selected_mode = request.form.get(
            "detection_mode", "authenticity"
        )

        caption = request.form.get(
            "caption", ""
        ).strip()

        if not uploaded_file or not uploaded_file.filename:

            error = "Please select an image."

        else:

            original_name = secure_filename(
                uploaded_file.filename
            )

            if "." not in original_name:

                error = "Invalid image filename."

            else:

                extension = original_name.rsplit(".", 1)[-1].lower()

                if extension not in ALLOWED_IMAGES:

                    error = "Upload JPG, JPEG, PNG, WEBP or BMP images."

                elif selected_mode not in {"authenticity", "context"}:

                    error = "Please select a valid detection mode."

                elif selected_mode == "context" and not caption:

                    error = "Please enter the news caption for context analysis."

                else:

                    filename = f"{uuid.uuid4().hex}.{extension}"
                    file_path = IMAGE_UPLOAD_DIR / filename

                    try:

                        uploaded_file.save(file_path)

                        if selected_mode == "authenticity":

                            from image_detector import predict_image

                            authenticity_result = predict_image(
                                str(file_path)
                            )

                        else:

                            from image_context_detector import predict

                            context_result = predict(
                                str(file_path),
                                caption
                            )

                    except Exception as exc:

                        app.logger.exception("Image analysis failed")
                        error = f"Image analysis failed: {exc}"

                    finally:

                        if file_path.exists():
                            file_path.unlink()

    return render_template(
        "image.html",
        authenticity_result=authenticity_result,
        context_result=context_result,
        selected_mode=selected_mode,
        caption=caption,
        error=error
    )


# =====================================================
# MODEL COMPARISON DASHBOARD
# =====================================================

@app.route("/dashboard")
def dashboard():
    # Existing LR/BiLSTM values are group-specific dashboard metrics.
    # CNN values are from the separately reported held-out evaluation.
    metrics = {
        "logistic": {
            "accuracy": 83.43, "macro_f1": 77.83,
            "fake_precision": 65.03, "fake_recall": 68.44, "fake_f1": 66.70
        },
        "cnn": {
            "accuracy": 89.63, "macro_f1": 85.00,
            "fake_precision": 84.36, "fake_recall": 70.25, "fake_f1": 76.67
        },
        "bilstm": {
            "accuracy": 81.10, "macro_f1": 71.83,
            "fake_precision": 64.56, "fake_recall": 48.92, "fake_f1": 55.66
        }
    }
    return render_template("dashboard.html", metrics=metrics)

# =====================================================
# PREDICTION HISTORY
# =====================================================

@app.route("/history")
def history():

    search_query = request.args.get(
        "search", ""
    ).strip()

    selected_model = request.args.get(
        "model", "all"
    )

    conn = sqlite3.connect(DATABASE)

    try:

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM predictions WHERE 1=1"
        parameters = []

        if search_query:

            query += " AND headline LIKE ?"
            parameters.append(f"%{search_query}%")

        if selected_model == "logistic":

            query += " AND model_name = ?"
            parameters.append("Logistic Regression")

        elif selected_model == "bilstm":

            query += " AND model_name = ?"
            parameters.append("BiLSTM Deep Learning")

        elif selected_model == "cnn":

            query += " AND model_name = ?"
            parameters.append("CNN Deep Learning")

        query += " ORDER BY id DESC LIMIT 100"

        cursor.execute(query, parameters)

        predictions = cursor.fetchall()

    finally:

        conn.close()

    return render_template(
        "history.html",
        predictions=predictions,
        search_query=search_query,
        selected_model=selected_model
    )


# =====================================================
# DELETE PREDICTION HISTORY
# =====================================================

@app.route(
    "/history/delete/<int:prediction_id>",
    methods=["POST"]
)
def delete_prediction(prediction_id):

    conn = sqlite3.connect(DATABASE)

    try:

        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM predictions WHERE id = ?",
            (prediction_id,)
        )

        conn.commit()

    finally:

        conn.close()

    return redirect(url_for("history"))


# =====================================================
# APPLICATION START
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )
