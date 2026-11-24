#!/usr/bin/env python3
"""
Evaluate the three existing TruthLens text models:
  1. Logistic Regression + TF-IDF
  2. CNN
  3. BiLSTM

Run from the project root:
    python evaluate_text_models.py

Expected default files:
  dataset/test.csv
  model/logistic_regression_final.pkl
  model/tfidf_vectorizer_final.pkl
  model/fake_news_cnn.keras
  model/tokenizer.pkl
  model/config.pkl
  model/fake_news_bilstm_group.keras
  model/tokenizer_bilstm_group.pkl
  model/config_bilstm_group.pkl

The script does not train or overwrite any model. Results are written to
evaluation_results/text/.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "test.csv"
OUT_DIR = ROOT / "evaluation_results" / "text"

# Existing artifacts in this project. The first existing path is selected.
ARTIFACTS = {
    "logistic_regression": {
        "model": [
            "model/logistic_regression_final.pkl",
            "model/logistic_regression_group.pkl",
        ],
        "vectorizer": [
            "model/tfidf_vectorizer_final.pkl",
            "model/tfidf_vectorizer_group.pkl",
        ],
    },
    "cnn": {
        "model": [
            "model/fake_news_cnn.keras",
            "model/fake_news_cnn_leakage_safe.keras",
            "model/fake_news_cnn_weighted.keras",
        ],
        "tokenizer": [
            "model/tokenizer.pkl",
        ],
        "config": [
            "model/config.pkl",
        ],
    },
    "bilstm": {
        "model": [
            "model/fake_news_bilstm_group.keras",
            "model/fake_news_bilstm.keras",
        ],
        "tokenizer": [
            "model/tokenizer_bilstm_group.pkl",
            "model/tokenizer.pkl",
        ],
        "config": [
            "model/config_bilstm_group.pkl",
            "model/config.pkl",
        ],
    },
}

# Dataset convention in this project: 0 = FAKE, 1 = REAL.
LABEL_NAMES = ["FAKE", "REAL"]

# For one-output sigmoid models, this is the class represented by the sigmoid
# probability. Change only if your model training code used the opposite mapping.
SIGMOID_POSITIVE_LABEL = 1

# If the test CSV contains several text columns, these are tried in order.
TEXT_COLUMN_CANDIDATES = [
    "text", "content", "news", "article", "title", "statement", "headline"
]
LABEL_COLUMN_CANDIDATES = ["label", "labels", "target", "class", "fake", "is_fake"]


def find_existing(relative_candidates, description):
    for rel in relative_candidates:
        path = ROOT / rel
        if path.is_file():
            return path
    tried = "\n".join(f"  - {ROOT / p}" for p in relative_candidates)
    raise FileNotFoundError(
        f"Could not find {description}. Checked:\n{tried}"
    )


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df, candidates, kind):
    normalized = {normalize_column_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Could not identify the {kind} column in {list(df.columns)}. "
        f"Expected one of: {candidates}"
    )


def load_test_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Test dataset not found: {DATA_PATH}\n"
            "Place your labelled test CSV at dataset/test.csv."
        )

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError(f"The test dataset is empty: {DATA_PATH}")

    text_col = find_column(df, TEXT_COLUMN_CANDIDATES, "text")
    label_col = find_column(df, LABEL_COLUMN_CANDIDATES, "label")

    # Convert Arrow-backed / extension arrays to ordinary Python/NumPy values.
    texts = df[text_col].fillna("").astype(str).to_numpy(dtype=object)
    raw_labels = pd.to_numeric(df[label_col], errors="coerce")
    valid = raw_labels.notna().to_numpy() & (np.char.str_len(texts.astype(str)) > 0)

    texts = texts[valid]
    y = raw_labels[valid].astype("int64").to_numpy()

    unique = set(np.unique(y).tolist())
    if not unique.issubset({0, 1}) or len(unique) != 2:
        raise ValueError(
            f"Expected binary labels 0 and 1, found {sorted(unique)}. "
            "This script assumes 0=FAKE and 1=REAL."
        )

    print(f"Test file : {DATA_PATH}")
    print(f"Text col  : {text_col}")
    print(f"Label col : {label_col}")
    print(f"Samples   : {len(y)}")
    print(f"Labels    : {dict(zip(*np.unique(y, return_counts=True)))}")
    return texts, y


def clean_text(value):
    """Lightweight text normalization used before sequence-model tokenization."""
    value = str(value).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def get_config_value(config, keys, default):
    if isinstance(config, dict):
        for key in keys:
            if key in config and config[key] is not None:
                return config[key]
    return default


def load_keras_model(path):
    try:
        from tensorflow.keras.models import load_model
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not be imported. Activate the project virtual "
            "environment and install the project's TensorFlow dependency."
        ) from exc
    return load_model(path, compile=False)


def positive_class_scores(model, X, model_name):
    """Return probability for label 1, supporting sigmoid and softmax outputs."""
    raw = model.predict(X, verbose=0)
    pred = np.asarray(raw)

    if pred.ndim == 2 and pred.shape[1] == 1:
        p = pred[:, 0].astype(float)
        if SIGMOID_POSITIVE_LABEL == 1:
            return np.clip(p, 0.0, 1.0)
        return np.clip(1.0 - p, 0.0, 1.0)

    if pred.ndim == 2 and pred.shape[1] == 2:
        # Standard project convention: output index 0=FAKE, index 1=REAL.
        return np.clip(pred[:, 1].astype(float), 0.0, 1.0)

    raise ValueError(
        f"{model_name} produced output shape {pred.shape}; expected "
        "(n, 1) sigmoid or (n, 2) softmax."
    )


def evaluate_scores(name, y_true, scores):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) != len(y_true):
        raise ValueError(
            f"{name}: got {len(scores)} predictions for {len(y_true)} labels."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{name}: predictions contain NaN or infinite values.")

    y_pred = (scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "model": name,
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "precision_weighted": float(precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )),
        "recall_weighted": float(recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )),
        "f1_weighted": float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )),
        "confusion_matrix": cm.tolist(),
        "threshold": 0.5,
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["log_loss"] = float(log_loss(y_true, np.column_stack([1-scores, scores]), labels=[0, 1]))
    except ValueError:
        metrics["log_loss"] = None

    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=LABEL_NAMES,
        output_dict=True, zero_division=0
    )
    metrics["classification_report"] = report

    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.lower())
    with (OUT_DIR / f"{safe_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (OUT_DIR / f"{safe_name}_classification_report.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write(classification_report(
            y_true, y_pred, labels=[0, 1], target_names=LABEL_NAMES,
            zero_division=0
        ))

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{name} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{safe_name}_confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=LABEL_NAMES,
        zero_division=0
    ))
    print("Confusion matrix (rows=actual, columns=predicted; FAKE, REAL):")
    print(cm)
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Macro F1       : {metrics['f1_macro']:.4f}")
    print(f"ROC-AUC        : {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}")
    print(f"Log loss       : {metrics['log_loss'] if metrics['log_loss'] is not None else 'N/A'}")
    return metrics


def evaluate_logistic_regression(texts, y):
    print("\nLoading Logistic Regression...")
    model_path = find_existing(
        ARTIFACTS["logistic_regression"]["model"], "Logistic Regression model"
    )
    vectorizer_path = find_existing(
        ARTIFACTS["logistic_regression"]["vectorizer"], "TF-IDF vectorizer"
    )
    model = load_pickle(model_path)
    vectorizer = load_pickle(vectorizer_path)

    try:
        check_is_fitted(model)
        check_is_fitted(vectorizer)
    except Exception as exc:
        warnings.warn(f"Could not verify fitted state: {exc}")

    X = vectorizer.transform(texts.astype(str))
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The Logistic Regression artifact has no predict_proba(). "
            "It may not be a LogisticRegression classifier."
        )
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"LR model classes are {classes}; expected class 1=REAL.")
    scores = probabilities[:, classes.index(1)]
    print(f"Using model     : {model_path}")
    print(f"Using vectorizer: {vectorizer_path}")
    return evaluate_scores("Logistic Regression", y, scores)


def evaluate_neural_model(kind, texts, y):
    print(f"\nLoading {kind.upper()}...")
    cfg = ARTIFACTS[kind]
    model_path = find_existing(cfg["model"], f"{kind.upper()} Keras model")
    tokenizer_path = find_existing(cfg["tokenizer"], f"{kind.upper()} tokenizer")
    config_path = find_existing(cfg["config"], f"{kind.upper()} config")

    model = load_keras_model(model_path)
    tokenizer = load_pickle(tokenizer_path)
    config = load_pickle(config_path)

    max_length = int(get_config_value(
        config, ["max_length", "max_len", "sequence_length"], 300
    ))
    max_words = int(get_config_value(
        config, ["max_words", "vocab_size", "num_words"], 10000
    ))

    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except Exception as exc:
        raise RuntimeError("Could not import Keras pad_sequences.") from exc

    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    X = pad_sequences(
        sequences, maxlen=max_length, padding="post", truncating="post"
    )

    scores = positive_class_scores(model, X, kind.upper())
    print(f"Using model    : {model_path}")
    print(f"Using tokenizer: {tokenizer_path}")
    print(f"Using config   : {config_path}")
    print(f"Sequence length: {max_length}; max_words setting: {max_words}")
    return evaluate_scores(kind.upper(), y, scores)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    texts, y = load_test_data()

    results = []
    failures = []

    tasks = [
        ("Logistic Regression", lambda: evaluate_logistic_regression(texts, y)),
        ("CNN", lambda: evaluate_neural_model("cnn", texts, y)),
        ("BiLSTM", lambda: evaluate_neural_model("bilstm", texts, y)),
    ]

    for name, task in tasks:
        try:
            results.append(task())
        except Exception as exc:
            failures.append({"model": name, "error": str(exc)})
            print(f"\n[ERROR] {name} evaluation failed: {exc}")

    if results:
        summary = pd.DataFrame([
            {
                "Model": r["model"],
                "Samples": r["samples"],
                "Accuracy": r["accuracy"],
                "Precision (Macro)": r["precision_macro"],
                "Recall (Macro)": r["recall_macro"],
                "F1 (Macro)": r["f1_macro"],
                "ROC-AUC": r["roc_auc"],
                "Log Loss": r["log_loss"],
            }
            for r in results
        ])
        summary.to_csv(OUT_DIR / "text_models_comparison.csv", index=False)
        summary.to_json(OUT_DIR / "text_models_comparison.json", orient="records", indent=2)

        print("\n" + "=" * 72)
        print("TEXT MODEL COMPARISON")
        print("=" * 72)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    with (OUT_DIR / "evaluation_status.json").open("w", encoding="utf-8") as f:
        json.dump({"successful": [r["model"] for r in results], "failures": failures}, f, indent=2)

    print(f"\nResults folder: {OUT_DIR}")
    print("Generated per-model metrics JSON, classification reports, confusion matrices,")
    print("and a combined CSV/JSON comparison.")
    if failures:
        print("\nSome models did not evaluate. Fix the listed missing/incompatible artifact")
        print("paths and rerun. No model files were modified.")
        sys.exit(1)


if __name__ == "__main__":
    main()
