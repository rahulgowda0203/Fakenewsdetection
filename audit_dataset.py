
import pandas as pd
import os

DATASET_FOLDER = "dataset"

files = {
    "politifact_fake.csv": 0,
    "politifact_real.csv": 1,
    "gossipcop_fake.csv": 0,
    "gossipcop_real.csv": 1
}

all_data = []

for filename, label in files.items():

    path = os.path.join(DATASET_FOLDER, filename)
    df = pd.read_csv(path)

    df["label"] = label
    df["source"] = filename

    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

print("\n========== DATASET QUALITY AUDIT ==========")

print("\nTotal records:", len(data))

print("\nMissing values:")
print(data.isnull().sum())

print("\nEmpty headlines:")
print(
    data["title"].fillna("").str.strip().eq("").sum()
)

print("\nDuplicate headlines:")
print(data["title"].dropna().duplicated().sum())

print("\nDuplicate URLs:")
print(data["news_url"].dropna().duplicated().sum())

print("\nClass distribution:")
print(data["label"].value_counts())

print("\nRecords by source:")
print(data["source"].value_counts())

print("\n========== AUDIT COMPLETED ==========")
