
import pandas as pd

df = pd.read_csv("dataset/cleaned_news.csv")

df["normalized_title"] = (
    df["title"]
    .fillna("")
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

duplicate_rows = df[
    df.duplicated("normalized_title", keep=False)
].copy()

conflicting_titles = (
    duplicate_rows.groupby("normalized_title")["label"]
    .nunique()
)

conflicting_titles = conflicting_titles[
    conflicting_titles > 1
].index

conflicts = duplicate_rows[
    duplicate_rows["normalized_title"].isin(conflicting_titles)
].copy()

conflicts = conflicts.sort_values(
    ["normalized_title", "label"]
)

print("=" * 60)
print("CONFLICTING DUPLICATE HEADLINES")
print("=" * 60)

print("Conflicting headline groups:", len(conflicting_titles))
print("Total records involved:", len(conflicts))

for title, group in conflicts.groupby("normalized_title"):
    print("\n" + "-" * 60)
    print("HEADLINE:", title)
    print("FAKE records:", (group["label"] == 0).sum())
    print("REAL records:", (group["label"] == 1).sum())

    for _, row in group.iterrows():
        print("\nLabel:", "FAKE" if row["label"] == 0 else "REAL")
        print("Source:", row["source_file"])
        print("URL:", row["news_url"])

conflicts.to_csv(
    "dataset/conflicting_duplicate_headlines.csv",
    index=False
)

print("\nReport saved:")
print("dataset/conflicting_duplicate_headlines.csv")
