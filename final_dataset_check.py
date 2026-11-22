
import pandas as pd

df = pd.read_csv("dataset/model_ready_news.csv")

print("=" * 50)
print("FINAL DATASET CHECK")
print("=" * 50)

print("\nTotal records:", len(df))

print("\nExact duplicate rows:", df.duplicated().sum())

print("\nDuplicate headlines:", df["title"].duplicated().sum())

print(
    "\nDuplicate URLs:",
    df["news_url"].dropna().duplicated().sum()
)

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["label"].value_counts().sort_index())

print("\nSource distribution:")
print(pd.crosstab(df["source_file"], df["label"]))

print("\nFINAL CHECK COMPLETED")
