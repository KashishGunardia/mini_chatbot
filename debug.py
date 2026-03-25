import pandas as pd

df = pd.read_csv("nutrition.csv")

print("Columns:", df.columns)
print("Rows:", len(df))
print(df.head())