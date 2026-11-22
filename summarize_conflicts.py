
import pandas as pd

df = pd.read_csv("dataset/conflicting_duplicate_headlines.csv")

summary = df.groupby("normalized_title").agg(
    total_records=("label", "size"),
    fake_records=("label", lambda x: (x == 0).sum()),
    real_records=("label", lambda x: (x == 1).sum()),
    sources=("source_file", lambda x: ", ".join(sorted(set(x))))
).reset_index()

print("TOTAL CONFLICTING GROUPS:", len(summary))
print("\nCONFLICTING HEADLINES:\n")

for _, row in summary.iterrows():
    print("-" * 60)
    print("Headline:", row["normalized_title"])
    print("Total:", row["total_records"])
    print("Fake:", row["fake_records"])
    print("Real:", row["real_records"])
    print("Sources:", row["sources"])
