from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
import os

PROCESSED_PATH = "data/processed/"
VECTOR_DB_PATH = "vector_db/"

os.makedirs(VECTOR_DB_PATH, exist_ok=True)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def ingest_data(file_name, db_name):
    print(f"Processing {file_name}...")

    df = pd.read_csv(os.path.join(PROCESSED_PATH, file_name))

    
    if "text" not in df.columns:
        print(f" Skipping {file_name}, no 'text' column found")
        return

    
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    if df.empty:
        print(f"⚠️ Skipping {file_name}, no valid text rows")
        return

    print(f"{file_name} valid rows:", len(df))

    
    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()

    
    vectorstore = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=os.path.join(VECTOR_DB_PATH, db_name)
    )

    
    vectorstore.persist()
    print(f"{db_name} stored successfully!\n")


if __name__ == "__main__":
    ingest_data("tech.csv", "tech")
    ingest_data("nutrition.csv", "nutrition")
    ingest_data("movies.csv", "movies")
    ingest_data("general.csv", "general")