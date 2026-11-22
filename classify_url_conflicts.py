
import pandas as pd

df = pd.read_csv("full_url_conflicts.csv")

summary = []

for url, group in df.groupby("canonical_url"):

    titles = group["title"].dropna().astype(str).unique()
    labels = set(group["label"].dropna().astype(int))

    title_count = len(titles)
    fake_count = (group["label"] == 0).sum()
    real_count = (group["label"] == 1).sum()

    if "wikipedia.org" in url.lower():
        category = "GENERAL REFERENCE PAGE"
    elif title_count == 1:
        category = "SAME TITLE - CONFLICTING LABELS"
    else:
        category = "MULTIPLE TITLES - SAME URL"

    summary.append({
        "canonical_url": url,
        "category": category,
        "total_records": len(group),
        "unique_titles": title_count,
        "fake_records": fake_count,
        "real_records": real_count,
        "titles": " | ".join(titles),
        "sources": " | ".join(
            sorted(group["source"].dropna().unique())
        )
    })

result = pd.DataFrame(summary)

print("\n========== URL CONFLICT CLASSIFICATION ==========")

print("\nCategory counts:")
print(result["category"].value_counts().to_string())

print("\nDetailed classification:")
print(result.to_string(index=False))

result.to_csv("classified_url_conflicts.csv", index=False)

print("\nSaved: classified_url_conflicts.csv")
