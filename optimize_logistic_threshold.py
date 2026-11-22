import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

df = pd.read_csv("dataset/train.csv")

X = df["title"].fillna("").astype(str)
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

model = LogisticRegression(
    C=4.0,
    class_weight=None,
    max_iter=1000,
    random_state=42
)

model.fit(X_train_vec, y_train)

probabilities = model.predict_proba(X_val_vec)[:, 0]

print("\nTHRESHOLD COMPARISON")
print("-" * 65)

for threshold in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]:
    predictions = (probabilities >= threshold).astype(int)
    predictions = 1 - predictions

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy_score(y_val, predictions):.4f} | "
        f"Macro F1: {f1_score(y_val, predictions, average='macro'):.4f}"
    )

    print(classification_report(
        y_val,
        predictions,
        target_names=["Fake News", "Real News"],
        zero_division=0
    ))
