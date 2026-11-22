
import pandas as pd
from urllib.parse import urlparse, urlunparse

df = pd.read_csv("conflicting_headlines_report.csv")

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

    # Preserve query parameters because they may identify
    # different articles.
    return urlunparse((
        "https",
        domain,
        path,
        "",
        parsed.query,
        ""
    ))

df["canonical_url"] = df["news_url"].apply(normalize_url)

df = df[df["canonical_url"] != ""]

# Find URLs appearing with both labels
url_label_counts = df.groupby("canonical_url")["label"].nunique()

conflicting_urls = url_label_counts[
    url_label_counts > 1
].index

results = df[
    df["canonical_url"].isin(conflicting_urls)
].sort_values("canonical_url")

print("\n========== URL CONFLICT ANALYSIS ==========")
print("Unique canonical URLs:", df["canonical_url"].nunique())
print("URLs with conflicting labels:", len(conflicting_urls))
print("Records involved:", len(results))

print("\nSample conflicts:")

for url, group in results.groupby("canonical_url"):
    print("\nURL:", url)
    print(group[["title", "label", "source"]].to_string(index=False))
    print("-" * 70)

results.to_csv("same_url_conflicts.csv", index=False)

print("\nSaved report: same_url_conflicts.csv")
