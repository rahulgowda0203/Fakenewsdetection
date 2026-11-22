
from pathlib import Path
import pickle
import re
import sqlite3
from datetime import datetime

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing.sequence import pad_sequences
from werkzeug.utils import secure_filename
import uuid

# =====================================================
# APPLICATION CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATABASE = BASE_DIR / "prediction_history.db"

app = Flask(__name__)

# =====================================================
# AUDIO AND VIDEO UPLOAD CONFIGURATION
# =====================================================

UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_UPLOAD_DIR = UPLOAD_DIR / "audio"
VIDEO_UPLOAD_DIR = UPLOAD_DIR / "video"

AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

ALLOWED_AUDIO = {"wav", "flac"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv"}
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


# Create database when application starts
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
# LOAD BiLSTM DEEP LEARNING MODEL
# =====================================================

bilstm_model = tf.keras.models.load_model(
    MODEL_DIR / "fake_news_bilstm_group.keras"
)

with open(MODEL_DIR / "tokenizer_bilstm_group.pkl", "rb") as f:
    bilstm_tokenizer = pickle.load(f)

with open(MODEL_DIR / "config_bilstm_group.pkl", "rb") as f:
    bilstm_config = pickle.load(f)

BILSTM_THRESHOLD = float(bilstm_config["threshold"])


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
# NEWS PREDICTION FUNCTION
# =====================================================

def predict_news(headline, selected_model):

    headline = str(headline).strip()

    if not headline:
        raise ValueError("Please enter a news headline.")

    if selected_model == "logistic":

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

    elif selected_model == "bilstm":

        cleaned = clean_text(headline)

        sequence = bilstm_tokenizer.texts_to_sequences(
            [cleaned]
        )

        padded = pad_sequences(
            sequence,
            maxlen=bilstm_config["max_length"],
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
        "fake_probability": round(
            fake_probability * 100, 2
        ),
        "real_probability": round(
            real_probability * 100, 2
        ),
        "model_name": model_name,
        "threshold": threshold * 100
    }


# =====================================================
# HOME PAGE - NEWS DETECTION
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    headline = ""
    selected_model = "logistic"
    error = None

    if request.method == "POST":

        headline = request.form.get(
            "headline", ""
        ).strip()

        selected_model = request.form.get(
            "model_type", "logistic"
        )

        try:

            prediction = predict_news(
                headline,
                selected_model
            )

            # Save successful prediction to database
            save_prediction(
                headline,
                prediction
            )

        except ValueError as exc:

            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        headline=headline,
        selected_model=selected_model,
        error=error
    )


# =====================================================
# MODEL COMPARISON DASHBOARD
# =====================================================

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
            extension = uploaded_file.filename.rsplit(".", 1)[-1].lower()

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
# VIDEO DEEPFAKE DETECTION
# =====================================================

@app.route("/video", methods=["GET", "POST"])
def video_detection():

    prediction = None
    error = None

    if request.method == "POST":

        uploaded_file = request.files.get("video")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select a video file."

        else:
            extension = uploaded_file.filename.rsplit(".", 1)[-1].lower()

            if extension not in ALLOWED_VIDEO:
                error = "Please upload MP4, AVI, MOV or MKV video."

            else:
                filename = f"{uuid.uuid4().hex}.{extension}"
                file_path = VIDEO_UPLOAD_DIR / filename

                try:
                    uploaded_file.save(file_path)

                    from video_detector import detect_video

                    prediction = detect_video(str(file_path))

                except Exception as exc:
                    error = f"Video analysis failed: {exc}"

                finally:
                    if file_path.exists():
                        file_path.unlink()

    return render_template(
        "video.html",
        prediction=prediction,
        error=error
    )

@app.route("/dashboard")
def dashboard():

    metrics = {
        "logistic": {
            "accuracy": 83.43,
            "macro_f1": 77.83,
            "fake_precision": 65.03,
            "fake_recall": 68.44,
            "fake_f1": 66.70
        },

        "bilstm": {
            "accuracy": 81.10,
            "macro_f1": 71.83,
            "fake_precision": 64.56,
            "fake_recall": 48.92,
            "fake_f1": 55.66
        }
    }

    return render_template(
        "dashboard.html",
        metrics=metrics
    )


# =====================================================
# PREDICTION HISTORY
# =====================================================


@app.route("/history")
def history():

    search_query = request.args.get("search", "").strip()
    selected_model = request.args.get("model", "all")

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

@app.route("/history/delete/<int:prediction_id>", methods=["POST"])
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
