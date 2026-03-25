from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

VECTOR_DB_PATH = "vector_db/"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def check_db(db_name):
    db_path = os.path.join(VECTOR_DB_PATH, db_name)

    if not os.path.exists(db_path):
        print(f"{db_name} DB not found")
        return

    db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )

    count = db._collection.count()
    print(f"{db_name} has {count} embeddings")


if __name__ == "__main__":
    check_db("tech")
    check_db("nutrition")
    check_db("movies")
    check_db("general")