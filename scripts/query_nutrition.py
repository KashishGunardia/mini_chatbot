from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

VECTOR_DB_PATH = "vector_db/"


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

nutrition_db = Chroma(
    persist_directory=os.path.join(VECTOR_DB_PATH, "nutrition"),
    embedding_function=embedding_model
)

query = input("Enter your nutrition query: ")
results = nutrition_db.similarity_search(query, k=5)

print("\nTop 5 relevant nutrition entries:\n")
for i, r in enumerate(results):
    print(f"{i+1}. {r.page_content}\n")

    