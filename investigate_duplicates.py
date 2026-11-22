
import pandas as pd

df = pd.read_csv("dataset/cleaned_news.csv")

df["normalized_title"] = (
    df["title"]
    .fillna("")
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

duplicate_groups = df[
    df.duplicated("normalized_title", keep=False)
].copy()

summary = duplicate_groups.groupby("normalized_title").agg(
    total_records=("label", "size"),
    unique_labels=("label", "nunique"),
    fake_records=("label", lambda x: (x == 0).sum()),
    real_records=("label", lambda x: (x == 1).sum())
).reset_index()

summary["conflicting_labels"] = summary["unique_labels"] > 1

print("=" * 55)
print("DUPLICATE HEADLINE INVESTIGATION")
print("=" * 55)

print("\nTotal duplicate records:", len(duplicate_groups))
print("Duplicate headline groups:", len(summary))
print(
    "Groups with conflicting labels:",
    summary["conflicting_labels"].sum()
)
print(
    "Groups with consistent labels:",
    (~summary["conflicting_labels"]).sum()
)

print("\nTop 15 duplicate headline groups:")
print(
    summary.sort_values("total_records", ascending=False)
    .head(15)
    .to_string(index=False)
)

duplicate_groups.to_csv(
    "dataset/duplicate_headlines_review.csv",
    index=False
)

summary.to_csv(
    "dataset/duplicate_headlines_summary.csv",
    index=False
)

print("\nSaved:")
print("dataset/duplicate_headlines_review.csv")
print("dataset/duplicate_headlines_summary.csv")
