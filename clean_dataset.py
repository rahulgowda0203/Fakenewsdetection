
import pandas as pd
from urllib.parse import urlparse, urlunparse
from pathlib import Path

# Original dataset files and their labels
FILES = {
    "politifact_fake.csv": 0,
    "politifact_real.csv": 1,
    "gossipcop_fake.csv": 0,
    "gossipcop_real.csv": 1
}

DATA_DIR = Path("dataset")
DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. Load original datasets
# --------------------------------------------------

frames = []

for filename, label in FILES.items():
    path = DATA_DIR / filename

    df = pd.read_csv(path)
    df["label"] = label
    df["source"] = filename

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

print("Original records:", len(data))


# --------------------------------------------------
# 2. Normalize URLs
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Identify URLs with conflicting labels
# --------------------------------------------------

valid_urls = data[data["canonical_url"] != ""]

label_counts = (
    valid_urls.groupby("canonical_url")["label"].nunique()
)

conflicting_urls = label_counts[
    label_counts > 1
].index

conflict_data = valid_urls[
    valid_urls["canonical_url"].isin(conflicting_urls)
].copy()


# --------------------------------------------------
# 4. Classify conflicting URL groups
# --------------------------------------------------

url_categories = {}

for url, group in conflict_data.groupby("canonical_url"):

    titles = group["title"].dropna().astype(str).unique()

    if "wikipedia.org" in url.lower():
        category = "GENERAL REFERENCE PAGE"

    elif len(titles) == 1:
        category = "SAME TITLE - CONFLICTING LABELS"

    else:
        category = "MULTIPLE TITLES - SAME URL"

    url_categories[url] = category


# --------------------------------------------------
# 5. Separate records
# --------------------------------------------------

data["conflict_category"] = data["canonical_url"].map(
    url_categories
)

# Exclude the 88 same-title conflicts and
# 22 general reference page groups.
excluded = data[
    data["conflict_category"].isin([
        "SAME TITLE - CONFLICTING LABELS",
        "GENERAL REFERENCE PAGE"
    ])
].copy()

# Keep the 3 multiple-title groups for manual review.
manual_review = data[
    data["conflict_category"] ==
    "MULTIPLE TITLES - SAME URL"
].copy()

# Clean training data excludes all conflicting URL groups.
cleaned = data[
    data["conflict_category"].isna()
].copy()


# --------------------------------------------------
# 6. Save outputs
# --------------------------------------------------

# Remove temporary columns from training data.
cleaned = cleaned.drop(
    columns=["canonical_url", "conflict_category"]
)

cleaned.to_csv(
    DATA_DIR / "cleaned_news.csv",
    index=False
)

excluded.to_csv(
    DATA_DIR / "excluded_conflicts.csv",
    index=False
)

manual_review.to_csv(
    DATA_DIR / "manual_review_urls.csv",
    index=False
)


# --------------------------------------------------
# 7. Generate cleaning report
# --------------------------------------------------

report = f"""
FAKE NEWS DATASET CLEANING REPORT
=================================

Original records: {len(data)}

Unique conflicting URLs: {len(conflicting_urls)}

Records excluded from training:
{len(excluded)}

Records reserved for manual review:
{len(manual_review)}

Records retained for training:
{len(cleaned)}

Original CSV files were not modified.

Cleaning strategy:
- Same-title conflicting URL groups excluded.
- Wikipedia reference URL groups excluded.
- Multiple-title conflicting URL groups reserved for review.
- Records without URLs retained.

Final class distribution:
{cleaned['label'].value_counts().to_string()}
"""

with open(DATA_DIR / "cleaning_report.txt", "w",
          encoding="utf-8") as file:
    file.write(report)

print(report)

print("\nFiles created:")
print("1. dataset/cleaned_news.csv")
print("2. dataset/excluded_conflicts.csv")
print("3. dataset/manual_review_urls.csv")
print("4. dataset/cleaning_report.txt")
