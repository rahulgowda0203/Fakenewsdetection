
import pandas as pd

df = pd.read_csv("full_url_conflicts.csv")

summary = (
    df.groupby("canonical_url")
    .agg(
        total_records=("label", "size"),
        fake_records=("label", lambda x: (x == 0).sum()),
        real_records=("label", lambda x: (x == 1).sum()),
        titles=("title", lambda x: " | ".join(
            sorted(set(x.dropna().astype(str)))
        )),
        sources=("source", lambda x: " | ".join(
            sorted(set(x.dropna().astype(str)))
        ))
    )
    .reset_index()
)

summary = summary.sort_values(
    "total_records", ascending=False
)

print("\n========== URL CONFLICT SUMMARY ==========")
print("Total conflicting URLs:", len(summary))

print("\nTop 20 conflicting URLs:")
print(summary.head(20).to_string(index=False))

summary.to_csv("url_conflict_summary.csv", index=False)

print("\nSaved: url_conflict_summary.csv")
