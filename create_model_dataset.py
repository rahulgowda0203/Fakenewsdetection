
import pandas as pd

source = "dataset/cleaned_news.csv"
conflict_file = "dataset/conflicting_duplicate_headlines.csv"

df = pd.read_csv(source)
conflicts = pd.read_csv(conflict_file)

# Identify every normalized headline with conflicting labels
conflict_titles = conflicts["normalized_title"].dropna().unique()

# Normalize titles in the main dataset
df["normalized_title"] = (
    df["title"]
    .fillna("")
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Separate records for auditing
excluded = df[df["normalized_title"].isin(conflict_titles)].copy()
model_data = df[~df["normalized_title"].isin(conflict_titles)].copy()

# Save both datasets without changing the original
excluded.to_csv(
    "dataset/headline_conflict_excluded.csv",
    index=False
)

model_data.drop(columns=["normalized_title"]).to_csv(
    "dataset/model_ready_news.csv",
    index=False
)

print("=" * 50)
print("MODEL DATASET CREATED")
print("=" * 50)

print("Original cleaned records:", len(df))
print("Conflicting headline groups:", len(conflict_titles))
print("Excluded records:", len(excluded))
print("Remaining records:", len(model_data))

print("\nRemaining class distribution:")
print(model_data["label"].value_counts().sort_index())

print("\nSaved:")
print("dataset/model_ready_news.csv")
print("dataset/headline_conflict_excluded.csv")
