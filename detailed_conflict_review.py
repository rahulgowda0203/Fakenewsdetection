
import pandas as pd

df = pd.read_csv(
    "dataset/conflicting_duplicate_headlines.csv"
)

for title, group in df.groupby("normalized_title"):
    print("\n" + "=" * 80)
    print("HEADLINE:", title)
    print("=" * 80)

    for _, row in group.iterrows():
        print("\nLabel:", "FAKE" if row["label"] == 0 else "REAL")
        print("Source:", row["source_file"])
        print("URL:", row["news_url"])

report = df[
    ["normalized_title", "title", "label", "source_file", "news_url"]
].copy()

report["label_name"] = report["label"].map({
    0: "FAKE",
    1: "REAL"
})

report.to_csv(
    "dataset/detailed_conflict_review.csv",
    index=False
)

print("\nDetailed report saved:")
print("dataset/detailed_conflict_review.csv")
