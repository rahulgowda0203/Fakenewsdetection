
import pandas as pd

files = {
    "politifact_fake.csv": 0,
    "politifact_real.csv": 1,
    "gossipcop_fake.csv": 0,
    "gossipcop_real.csv": 1
}

all_data = []

for filename, label in files.items():
    df = pd.read_csv(f"dataset/{filename}")
    df["label"] = label
    df["source"] = filename
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Normalize headlines
data["normalized_title"] = (
    data["title"]
    .fillna("")
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# Find headlines appearing with both labels
label_counts = data.groupby("normalized_title")["label"].nunique()
conflicting_titles = label_counts[label_counts > 1].index

conflicts = data[
    data["normalized_title"].isin(conflicting_titles)
].copy()

# Sort and export
conflicts = conflicts.sort_values(
    ["normalized_title", "label"]
)

conflicts.to_csv(
    "conflicting_headlines_report.csv",
    index=False
)

print("========== INVESTIGATION REPORT ==========")
print("Conflicting headline groups:", len(conflicting_titles))
print("Total conflicting records:", len(conflicts))

print("\nRecords by source:")
print(conflicts["source"].value_counts())

print("\nLabel distribution:")
print(conflicts["label"].value_counts())

print("\nReport saved: conflicting_headlines_report.csv")
