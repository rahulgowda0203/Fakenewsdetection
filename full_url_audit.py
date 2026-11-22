
import pandas as pd
from urllib.parse import urlparse, urlunparse

files = {
    "politifact_fake.csv": 0,
    "politifact_real.csv": 1,
    "gossipcop_fake.csv": 0,
    "gossipcop_real.csv": 1
}

frames = []

for filename, label in files.items():
    df = pd.read_csv(f"dataset/{filename}")

    df["label"] = label
    df["source"] = filename

    frames.append(df)

data = pd.concat(frames, ignore_index=True)


def normalize_url(url):
    if pd.isna(url) or not str(url).strip():
        return ""

    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    path = parsed.path.rstrip("/")

    return urlunparse((
        "https",
        domain,
        path,
        "",
        parsed.query,
        ""
    ))


data["canonical_url"] = data["news_url"].apply(normalize_url)

valid = data[data["canonical_url"] != ""].copy()

url_counts = valid.groupby("canonical_url")["label"].nunique()

conflicting_urls = url_counts[url_counts > 1].index

results = valid[
    valid["canonical_url"].isin(conflicting_urls)
].sort_values("canonical_url")

print("\n========== FULL DATASET URL AUDIT ==========")

print("Total dataset records:", len(data))
print("Records with valid URLs:", len(valid))
print("Unique canonical URLs:", valid["canonical_url"].nunique())
print("URLs with conflicting labels:", len(conflicting_urls))
print("Records involved:", len(results))

print("\nLabel distribution in conflicting records:")
print(results["label"].value_counts())

results.to_csv("full_url_conflicts.csv", index=False)

print("\nReport saved as full_url_conflicts.csv")
