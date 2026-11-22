
import pandas as pd

df = pd.read_csv("dataset/cleaned_news.csv")

print("=" * 50)
print("CLEANED DATASET VALIDATION")
print("=" * 50)

print("\n1. Dataset shape:")
print(df.shape)

print("\n2. Missing values:")
print(df.isnull().sum())

print("\n3. Empty headlines:")
print(df["title"].fillna("").str.strip().eq("").sum())

print("\n4. Exact duplicate headlines:")
print(df["title"].duplicated().sum())

print("\n5. Normalized duplicate headlines:")
normalized = (
    df["title"]
    .fillna("")
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
print(normalized.duplicated().sum())

print("\n6. Duplicate URLs:")
print(df["news_url"].dropna().duplicated().sum())

print("\n7. Class distribution:")
print(df["label"].value_counts().sort_index())

print("\n8. Source distribution:")
print(pd.crosstab(df["source_file"], df["label"]))

print("\n9. Sample headlines:")
print(df[["title", "label"]].head(10).to_string(index=False))

print("\n" + "=" * 50)
print("VALIDATION COMPLETED")
print("=" * 50)
