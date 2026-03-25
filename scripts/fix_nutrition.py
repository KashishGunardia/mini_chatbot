import pandas as pd
import os
import glob
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "nutrition")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db", "nutrition")

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)



embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



nutrition_files = glob.glob(os.path.join(RAW_PATH, "*.csv"))

if not nutrition_files:
    raise FileNotFoundError("No CSV files found in data/raw/nutrition/")

dfs = [pd.read_csv(file) for file in nutrition_files]
df = pd.concat(dfs, ignore_index=True)

print(f"Total raw rows: {len(df)}")




required_cols = ["food_name", "calories", "protein"]

for col in required_cols:
    if col not in df.columns:
        df[col] = ""


df["food_name"] = df["food_name"].fillna("").astype(str)
df["calories"] = df["calories"].fillna("").astype(str)
df["protein"] = df["protein"].fillna("").astype(str)

df = df[
    (df["food_name"] != "") &
    (df["food_name"].str.lower() != "unknown") &
    (df["food_name"].str.lower() != "n/a")
]


print(f"Rows after cleaning: {len(df)}")



df["text"] = df.apply(
    lambda row: f"Food: {row['food_name']}"
                + (f", Calories: {row['calories']}" if row["calories"] else "")
                + (f", Protein: {row['protein']}" if row["protein"] else ""),
    axis=1
)


if df.empty:
    raise ValueError("No valid data after cleaning. Check your dataset!")



df.to_csv(os.path.join(PROCESSED_PATH, "nutrition_cleaned.csv"), index=False)



loader = DataFrameLoader(df, page_content_column="text")
documents = loader.load()

print(f"Documents to embed: {len(documents)}")



vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=VECTOR_DB_PATH
)

print("Nutrition embeddings created successfully!")