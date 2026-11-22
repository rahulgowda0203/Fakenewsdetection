
import pandas as pd
import re
from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit

INPUT = "dataset/model_ready_news.csv"

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

# Join records that share either a headline or URL.
# Connected records remain in the same split.
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
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, df["label"], groups=df["split_group"])
)

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

# Verify that no headline or URL crosses the split.
assert set(train["title_group"]).isdisjoint(
    set(test["title_group"])
)

train_urls = set(train["url_group"].dropna())
test_urls = set(test["url_group"].dropna())

assert train_urls.isdisjoint(test_urls)

columns = [
    "id", "news_url", "title", "tweet_ids",
    "label", "source_file"
]

train[columns].to_csv("dataset/train.csv", index=False)
test[columns].to_csv("dataset/test.csv", index=False)

print("=" * 50)
print("LEAKAGE-SAFE SPLIT COMPLETED")
print("=" * 50)

print("Total records:", len(df))
print("Train records:", len(train))
print("Test records:", len(test))
print("Train percentage:", round(len(train) / len(df) * 100, 2))
print("Test percentage:", round(len(test) / len(df) * 100, 2))

print("\nTraining class distribution:")
print(train["label"].value_counts().sort_index())

print("\nTesting class distribution:")
print(test["label"].value_counts().sort_index())

print("\nUnique groups:", df["split_group"].nunique())
print("Headline overlap:", len(
    set(train["title_group"]) & set(test["title_group"])
))
print("URL overlap:", len(train_urls & test_urls))

print("\nSaved dataset/train.csv")
print("Saved dataset/test.csv")
