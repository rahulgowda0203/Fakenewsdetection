#!/usr/bin/env python3
"""
TruthLens AI - Audio model evaluation report generator.

This evaluates the already-saved audio test results (report JSON and confusion
matrix CSV). It does NOT retrain the audio model or pretend to re-run inference.

Run from the repository root:
    python evaluate_audio_metrics.py

Expected project artifacts (the script searches common locations):
    models/audio/secureaudio_test_report.json
    models/audio/secureaudio_confusion_matrix.csv

If your files are elsewhere, edit REPORT_CANDIDATES and CM_CANDIDATES below.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evaluation_results" / "audio"

REPORT_CANDIDATES = [
    "models/audio/secureaudio_test_report.json",
    "models/audio/audio_test_report.json",
    "audio_training/secureaudio_test_report.json",
    "secureaudio_test_report.json",
]
CM_CANDIDATES = [
    "models/audio/secureaudio_confusion_matrix.csv",
    "models/audio/audio_confusion_matrix.csv",
    "audio_training/secureaudio_confusion_matrix.csv",
    "secureaudio_confusion_matrix.csv",
]

# Rows and columns in the known project confusion matrix are [Real, Fake].
CLASS_NAMES = ["Real", "Fake"]


def first_existing(candidates):
    for item in candidates:
        p = ROOT / item
        if p.is_file():
            return p
    return None


def load_confusion_matrix(path):
    # Handle CSVs with or without row/column labels.
    df = pd.read_csv(path, header=None)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cm = numeric.to_numpy(dtype=float)
    if cm.shape == (3, 3):
        # Common format: first row/column contain class labels.
        cm = cm[1:, 1:]
    if cm.shape != (2, 2):
        raise ValueError(
            f"Expected a 2x2 confusion matrix in {path}, got {cm.shape}.\n"
            "Open the CSV and check whether it contains extra headers/indexes."
        )
    if not np.all(np.isfinite(cm)) or np.any(cm < 0):
        raise ValueError("Confusion matrix contains invalid values.")
    if not np.allclose(cm, np.round(cm)):
        raise ValueError("Confusion matrix must contain integer counts.")
    return np.round(cm).astype(int)


def metrics_from_cm(cm):
    # Matrix layout: rows=actual [Real, Fake], columns=predicted [Real, Fake]
    y_true, y_pred = [], []
    for actual in range(2):
        for predicted in range(2):
            count = int(cm[actual, predicted])
            y_true.extend([actual] * count)
            y_pred.extend([predicted] * count)

    accuracy = accuracy_score(y_true, y_pred)
    return {
        "samples": int(cm.sum()),
        "accuracy": float(accuracy),
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
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = first_existing(REPORT_CANDIDATES)
    cm_path = first_existing(CM_CANDIDATES)

    if not cm_path:
        print("[ERROR] Audio confusion-matrix CSV was not found.")
        print("Searched:")
        for p in CM_CANDIDATES:
            print("  ", ROOT / p)
        print("\nPut your saved audio confusion matrix at:")
        print("  models/audio/secureaudio_confusion_matrix.csv")
        sys.exit(1)

    cm = load_confusion_matrix(cm_path)
    result = metrics_from_cm(cm)

    # Include the saved report as source metadata, without assuming a schema.
    saved_report = None
    if report_path:
        try:
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARNING] Could not parse saved report JSON: {exc}")

    result["evaluation_source"] = "saved test confusion matrix; no new inference run"
    result["confusion_matrix_file"] = str(cm_path)
    result["saved_report_file"] = str(report_path) if report_path else None
    result["saved_report_contents"] = saved_report

    # Save metrics JSON and readable report.
    (OUT / "audio_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    report_text = classification_report(
        [0] * int(cm[0].sum()) + [1] * int(cm[1].sum()),
        [0] * int(cm[0, 0]) + [1] * int(cm[0, 1]) +
        [0] * int(cm[1, 0]) + [1] * int(cm[1, 1]),
        labels=[0, 1], target_names=CLASS_NAMES, zero_division=0
    )
    (OUT / "audio_classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    # Confusion matrix plot.
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Audio Detection - Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    fig.tight_layout()
    fig.savefig(OUT / "audio_confusion_matrix.png", dpi=220)
    plt.close(fig)

    # Metric comparison plot.
    names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    values = [
        result["accuracy"],
        result["precision_macro"],
        result["recall_macro"],
        result["f1_macro"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Audio Detection - Evaluation Metrics")
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUT / "audio_metrics.png", dpi=220)
    plt.close(fig)

    print("\n" + "=" * 58)
    print("TRUTHLENS AUDIO DETECTION - EVALUATION RESULTS")
    print("=" * 58)
    print(f"Confusion matrix file : {cm_path}")
    print(f"Saved report found    : {bool(report_path)}")
    print(f"Test samples          : {result['samples']}")
    print(f"Accuracy              : {result['accuracy']:.4%}")
    print(f"Macro precision       : {result['precision_macro']:.4f}")
    print(f"Macro recall          : {result['recall_macro']:.4f}")
    print(f"Macro F1-score        : {result['f1_macro']:.4f}")
    print(f"Weighted F1-score     : {result['f1_weighted']:.4f}")
    print("\nConfusion matrix (rows=actual, columns=predicted; Real, Fake):")
    print(cm)
    print("\nClassification report:")
    print(report_text)
    print(f"\nAll output files saved in: {OUT}")
    print("Note: these metrics are calculated from your saved test confusion matrix.")
    print("They are not a fresh evaluation of raw audio files.")


if __name__ == "__main__":
    main()
