from pathlib import Path
import json
import warnings

import numpy as np
import soundfile as sf
import joblib

from scipy.fftpack import dct
from scipy.signal import stft

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "audio" / "secureaudio"
MODEL_DIR = ROOT / "models" / "audio"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 40


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def extract_features(file_path):
    audio, sr = sf.read(file_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected 16 kHz audio, got {sr} Hz")

    if len(audio) < 400:
        raise ValueError("Audio file is too short")

    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # Short-time Fourier transform
    _, times, z = stft(
        audio,
        fs=SAMPLE_RATE,
        nperseg=400,
        noverlap=240,
        nfft=512,
        boundary=None
    )

    power = np.abs(z) ** 2
    frequencies = np.linspace(0, SAMPLE_RATE / 2, power.shape[0])

    # Build mel filter bank
    mel_points = np.linspace(
        hz_to_mel(0),
        hz_to_mel(SAMPLE_RATE / 2),
        N_MELS + 2
    )
    hz_points = mel_to_hz(mel_points)

    filters = np.zeros((N_MELS, len(frequencies)))

    for m in range(N_MELS):
        left, center, right = hz_points[m:m + 3]

        filters[m] = np.maximum(
            0,
            np.minimum(
                (frequencies - left) / (center - left + 1e-8),
                (right - frequencies) / (right - center + 1e-8)
            )
        )

    mel_energy = filters @ power
    log_mel = np.log(mel_energy + 1e-8)

    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:N_MFCC]

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Additional spectral features
    magnitude = np.abs(z)
    total_energy = np.sum(magnitude, axis=0) + 1e-8

    centroid = np.sum(
        frequencies[:, None] * magnitude, axis=0
    ) / total_energy

    bandwidth = np.sqrt(
        np.sum(
            ((frequencies[:, None] - centroid) ** 2) * magnitude,
            axis=0
        ) / total_energy
    )

    cumulative = np.cumsum(power, axis=0)
    threshold = 0.85 * cumulative[-1]
    rolloff_indices = np.argmax(cumulative >= threshold, axis=0)
    rolloff = frequencies[rolloff_indices]

    frame_length = 400
    hop = 160
    zcr_values = []

    for start in range(0, max(1, len(audio) - frame_length + 1), hop):
        frame = audio[start:start + frame_length]
        if len(frame) < frame_length:
            continue
        zcr_values.append(
            np.mean(np.abs(np.diff(np.sign(frame)))) / 2
        )

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [
            np.mean(centroid),
            np.std(centroid),
            np.mean(bandwidth),
            np.std(bandwidth),
            np.mean(rolloff),
            np.std(rolloff),
            np.mean(zcr_values) if zcr_values else 0,
            np.std(zcr_values) if zcr_values else 0,
            np.mean(np.sqrt(np.mean(audio ** 2))),
        ]
    ])

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def load_split(split):
    X, y = [], []
    skipped = 0

    for label, class_id in [("real", 0), ("fake", 1)]:
        folder = DATASET / split / label

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        files = sorted(folder.glob("*.wav"))

        for index, file_path in enumerate(files, start=1):
            try:
                X.append(extract_features(file_path))
                y.append(class_id)
            except Exception as exc:
                skipped += 1
                print(f"Skipped {file_path.name}: {exc}")

            if index % 100 == 0:
                print(f"{split}/{label}: {index}/{len(files)} processed")

    print(f"{split}: loaded {len(X)} files; skipped {skipped}")

    return np.asarray(X), np.asarray(y)


def main():
    print("\nLoading training audio...")
    X_train_all, y_train_all = load_split("train")

    print("\nLoading independent test audio...")
    X_test, y_test = load_split("test")

    if len(X_train_all) == 0 or len(X_test) == 0:
        raise RuntimeError("No audio features were extracted.")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=0.20,
        random_state=42,
        stratify=y_train_all
    )

    print("\nTraining SVM classifier...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    # Validation results
    val_predictions = model.predict(X_val)

    print("\n========== VALIDATION RESULTS ==========")
    print("Accuracy:", round(accuracy_score(y_val, val_predictions) * 100, 2), "%")
    print(classification_report(
        y_val,
        val_predictions,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        zero_division=0
    ))

    # Final independent test
    test_predictions = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_predictions)

    print("\n========== FINAL TEST RESULTS ==========")
    print("Accuracy:", round(test_accuracy * 100, 2), "%")

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        output_dict=True,
        zero_division=0
    )

    print(classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])

    print("Confusion Matrix:")
    print(cm)
    print("Rows = actual [Real, Fake]; columns = predicted [Real, Fake]")

    model_path = MODEL_DIR / "secureaudio_svm.joblib"
    report_path = MODEL_DIR / "secureaudio_test_report.json"
    matrix_path = MODEL_DIR / "secureaudio_confusion_matrix.csv"

    joblib.dump(model, model_path)

    with open(report_path, "w") as f:
        json.dump({
            "dataset": "SecureAudio standardized corpus",
            "feature_type": "MFCC and spectral statistics",
            "classifier": "RBF SVM",
            "training_files": int(len(X_train)),
            "validation_files": int(len(X_val)),
            "test_files": int(len(X_test)),
            "test_accuracy": float(test_accuracy),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }, f, indent=4)

    np.savetxt(
        matrix_path,
        cm,
        delimiter=",",
        fmt="%d"
    )

    print("\nSaved model:", model_path)
    print("Saved report:", report_path)
    print("Saved confusion matrix:", matrix_path)
    print("\nAudio training and testing completed!")


if __name__ == "__main__":
    main()
