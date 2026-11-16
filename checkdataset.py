import pandas as pd
import os

dataset_folder = "dataset"

files = [
    "politifact_fake.csv",
    "politifact_real.csv",
    "gossipcop_fake.csv",
    "gossipcop_real.csv"
]

for file in files:
    path = os.path.join(dataset_folder, file)

    if os.path.exists(path):
        df = pd.read_csv(path)

        print("\n" + "=" * 50)
        print("FILE:", file)
        print("=" * 50)

        print("Rows:", len(df))
        print("Columns:")
        print(df.columns.tolist())

        print("\nFirst 3 rows:")
        print(df.head(3))
    else:
        print("\nFILE NOT FOUND:", path)