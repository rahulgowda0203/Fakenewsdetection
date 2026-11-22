import pandas as pd
import re
from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit

INPUT = "dataset/train.csv"

df = pd.read_csv(INPUT)

def normalize_title(title):
    title = str(title).lower().strip()
    return re.sub(r"\s+", " ", title)

def normalize_url(url):
    if pd.isna(url) or not str(url).strip():
        return None

    url = str(url).strip()

    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    try:
        parts = urlsplit(url)
        domain = re.sub(r"^www\.", "", parts.netloc.lower())
        path = parts.path.rstrip("/")
        return f"{domain}{path}".lower()
    except Exception:
        return None

df["title_group"] = df["title"].apply(normalize_title)
df["url_group"] = df["news_url"].apply(normalize_url)

# Union-find: connect records sharing a headline or URL
parent = list(range(len(df)))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    root_a = find(a)
    root_b = find(b)
    if root_a != root_b:
        parent[root_b] = root_a

for column in ["title_group", "url_group"]:
    for _, indices in df.groupby(column, dropna=True).groups.items():
        indices = list(indices)
        first = indices[0]

        for idx in indices[1:]:
            union(first, idx)

df["split_group"] = [find(i) for i in range(len(df))]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=123
)

train_idx, val_idx = next(
    splitter.split(df, df["label"], groups=df["split_group"])
)

train = df.iloc[train_idx].copy()
val = df.iloc[val_idx].copy()

# Verify no headline or URL overlap
assert set(train["title_group"]).isdisjoint(set(val["title_group"]))

train_urls = set(train["url_group"].dropna())
val_urls = set(val["url_group"].dropna())

assert train_urls.isdisjoint(val_urls)

columns = [
    "id", "news_url", "title", "tweet_ids",
    "label", "source_file"
]

train[columns].to_csv("dataset/group_train.csv", index=False)
val[columns].to_csv("dataset/group_validation.csv", index=False)

print("\nGROUP-AWARE VALIDATION CREATED")
print("-" * 45)
print("Training records:", len(train))
print("Validation records:", len(val))

print("\nTraining class distribution:")
print(train["label"].value_counts().sort_index())

print("\nValidation class distribution:")
print(val["label"].value_counts().sort_index())

print("\nHeadline overlap:", len(
    set(train["title_group"]) & set(val["title_group"])
))

print("URL overlap:", len(train_urls & val_urls))

print("\nSaved:")
print("dataset/group_train.csv")
print("dataset/group_validation.csv")
