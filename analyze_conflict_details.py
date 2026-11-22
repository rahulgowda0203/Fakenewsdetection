
import pandas as pd

df = pd.read_csv("conflicting_headlines_report.csv")

# Show headline, ID, URL, label and source
columns = ["title", "id", "news_url", "label", "source"]

# Keep available columns only
columns = [col for col in columns if col in df.columns]

# Sort by headline
df = df.sort_values("title")

print("\n========== CONFLICT DETAILS ==========\n")

for headline, group in df.groupby("title"):
    print("\nHeadline:", headline)
    print(group[columns].to_string(index=False))
    print("-" * 80)

# Count conflicts where URLs are different
url_analysis = df.groupby("title")["news_url"].nunique(dropna=True)

print("\n========== URL ANALYSIS ==========")
print("Headline groups:", len(url_analysis))
print("Groups with multiple URLs:", (url_analysis > 1).sum())
print("Groups with one URL:", (url_analysis == 1).sum())
