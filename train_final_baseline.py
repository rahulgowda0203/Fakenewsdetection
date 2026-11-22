import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load datasets
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

X_train = train_df["title"].fillna("").astype(str)
y_train = train_df["label"]

X_test = test_df["title"].fillna("").astype(str)
y_test = test_df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Best validation configuration
model = LogisticRegression(
    C=4.0,
    class_weight=None,
    max_iter=1000,
    random_state=42
)

# Train
model.fit(X_train_tfidf, y_train)

# Final test evaluation
predictions = model.predict(X_test_tfidf)

print("\nFINAL TEST RESULTS")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Macro F1:", round(f1_score(y_test, predictions, average="macro"), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Fake News", "Real News"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save final model
with open("model/logistic_regression_final.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer_final.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nFinal model and vectorizer saved successfully!")