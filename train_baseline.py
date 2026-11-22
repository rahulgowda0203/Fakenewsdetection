
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

X_train = train["title"].fillna("")
y_train = train["label"]

X_test = test["title"].fillna("")
y_test = test["label"]

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

print("Training Logistic Regression...")
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

print("\nBASELINE MODEL RESULTS")
print("=" * 50)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Macro F1:", round(f1_score(
    y_test, predictions, average="macro"
), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Fake", "Real"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "model/logistic_regression.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
