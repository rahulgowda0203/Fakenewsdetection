
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

# Find headlines appearing under multiple labels
conflicts = (
    data.groupby("normalized_title")["label"]
    .nunique()
)

conflict_titles = conflicts[conflicts > 1].index

conflict_records = data[
    data["normalized_title"].isin(conflict_titles)
]

print("\n========== DUPLICATE ANALYSIS ==========")

print("Total records:", len(data))

print(
    "Repeated headlines:",
    data["normalized_title"].duplicated().sum()
)

print(
    "Conflicting headline groups:",
    len(conflict_titles)
)

print(
    "Records with conflicting labels:",
    len(conflict_records)
)

print("\nSample conflicting records:")

print(
    conflict_records[
        ["title", "label", "source"]
    ].head(20).to_string(index=False)
)

print("\n========================================")
