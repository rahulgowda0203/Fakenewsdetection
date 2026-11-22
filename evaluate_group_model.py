import pandas as pd
import pickle
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

with open("model/logistic_regression_group.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer_group.pkl", "rb") as f:
    vectorizer = pickle.load(f)

test_df = pd.read_csv("dataset/test.csv")

X_test = test_df["title"].fillna("").astype(str)
y_test = test_df["label"]

X_test_tfidf = vectorizer.transform(X_test)

fake_index = list(model.classes_).index(0)
fake_probabilities = model.predict_proba(X_test_tfidf)[:, fake_index]

THRESHOLD = 0.325

predictions = np.where(fake_probabilities >= THRESHOLD, 0, 1)

print("\nFINAL GROUP-AWARE TEST RESULTS")
print("=" * 45)

print("Threshold:", THRESHOLD)
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Macro F1:", round(
    f1_score(y_test, predictions, average="macro"), 4
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Fake News", "Real News"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nEvaluation completed.")
