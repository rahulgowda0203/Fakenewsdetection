
import re
import pandas as pd
from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

df = pd.read_csv("dataset/train.csv")

def normalize_title(value):
    return re.sub(r"\s+", " ", str(value).lower().strip())

def normalize_url(value):
    if pd.isna(value) or not str(value).strip():
        return None
    value = str(value).strip()
    if not re.match(r"^https?://", value, re.I):
        value = "https://" + value
    parts = urlsplit(value)
    domain = re.sub(r"^www\.", "", parts.netloc.lower())
    return f"{domain}{parts.path.rstrip('/')}".lower()

df["title_group"] = df["title"].apply(normalize_title)
df["url_group"] = df["news_url"].apply(normalize_url)

# Keep connected headline/URL groups together
parent = list(range(len(df)))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    a, b = find(a), find(b)
    if a != b:
        parent[b] = a

for column in ["title_group", "url_group"]:
    for indices in df.groupby(column, dropna=True).groups.values():
        indices = list(indices)
        for idx in indices[1:]:
            union(indices[0], idx)

groups = [find(i) for i in range(len(df))]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(df, df["label"], groups=groups)
)

train = df.iloc[train_idx]
val = df.iloc[val_idx]

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train["title"].fillna(""))
X_val = vectorizer.transform(val["title"].fillna(""))

y_train = train["label"]
y_val = val["label"]

print("Training records:", len(train))
print("Validation records:", len(val))

results = []

for weight in [None, "balanced"]:
    for c in [0.5, 1.0, 2.0, 4.0]:
        model = LogisticRegression(
            C=c,
            max_iter=1000,
            class_weight=weight,
            random_state=42
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        results.append({
            "C": c,
            "class_weight": str(weight),
            "accuracy": accuracy_score(y_val, pred),
            "macro_f1": f1_score(
                y_val, pred, average="macro"
            ),
            "fake_f1": f1_score(
                y_val, pred, pos_label=0
            )
        })

results_df = pd.DataFrame(results).sort_values(
    "macro_f1", ascending=False
)

print("\nVALIDATION COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "baseline_tuning_results.csv",
    index=False
)

print("\nResults saved to baseline_tuning_results.csv")
