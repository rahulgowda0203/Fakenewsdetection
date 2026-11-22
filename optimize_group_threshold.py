import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

train_df = pd.read_csv("dataset/group_train.csv")
val_df = pd.read_csv("dataset/group_validation.csv")

X_train = train_df["title"].fillna("").astype(str)
y_train = train_df["label"]

X_val = val_df["title"].fillna("").astype(str)
y_val = val_df["label"]

print("Training TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

print("Training Logistic Regression...")

model = LogisticRegression(
    C=4.0,
    class_weight=None,
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# Get fake-news probability using the actual class labels
fake_index = list(model.classes_).index(0)
fake_probabilities = model.predict_proba(X_val_tfidf)[:, fake_index]

thresholds = np.arange(0.20, 0.551, 0.025)

results = []

print("\nGROUP-AWARE VALIDATION RESULTS")
print("=" * 75)
print(
    f"{'Threshold':<12}{'Accuracy':<12}"
    f"{'Fake Precision':<17}{'Fake Recall':<15}"
    f"{'Fake F1':<12}{'Macro F1':<12}"
)

for threshold in thresholds:
    predictions = (fake_probabilities >= threshold).astype(int)
    # Model labels: 0 = Fake, 1 = Real
    predictions = np.where(predictions == 1, 0, 1)

    accuracy = accuracy_score(y_val, predictions)
    fake_precision = precision_score(
        y_val, predictions, pos_label=0, zero_division=0
    )
    fake_recall = recall_score(
        y_val, predictions, pos_label=0, zero_division=0
    )
    fake_f1 = f1_score(
        y_val, predictions, pos_label=0, zero_division=0
    )
    macro_f1 = f1_score(
        y_val, predictions, average="macro", zero_division=0
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,
        "macro_f1": macro_f1
    })

    print(
        f"{threshold:<12.3f}{accuracy:<12.4f}"
        f"{fake_precision:<17.4f}{fake_recall:<15.4f}"
        f"{fake_f1:<12.4f}{macro_f1:<12.4f}"
    )

best = max(results, key=lambda x: x["macro_f1"])

print("\nBEST VALIDATION THRESHOLD")
print("=" * 40)
for key, value in best.items():
    print(f"{key}: {value:.4f}")

print("\nClassification report at selected threshold:")

best_predictions = np.where(
    fake_probabilities >= best["threshold"], 0, 1
)

print(classification_report(
    y_val,
    best_predictions,
    target_names=["Fake News", "Real News"],
    zero_division=0
))

# Save the group-trained model and vectorizer
import pickle
from pathlib import Path

Path("model").mkdir(exist_ok=True)

with open("model/logistic_regression_group.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer_group.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully.")
