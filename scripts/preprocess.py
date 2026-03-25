import pandas as pd
import os
import glob


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_PATH, exist_ok=True)

print("Raw folder contents:", os.listdir(DATA_PATH))



def clean_dataframe(df):
    return df.dropna().drop_duplicates()



tech_path = os.path.join(DATA_PATH, "valid.csv")
tech_df = pd.read_csv(tech_path)

tech_df = tech_df.dropna(subset=["Title", "Body"]).drop_duplicates()

tech_df["text"] = (
    "Question: " + tech_df["Title"].astype(str) +
    " Details: " + tech_df["Body"].astype(str) +
    " Tags: " + tech_df.get("Tags", "").astype(str)
)

tech_df.to_csv(os.path.join(PROCESSED_PATH, "tech.csv"), index=False)



nutrition_folder = os.path.join(DATA_PATH, "nutrition")
nutrition_files = glob.glob(os.path.join(nutrition_folder, "*.csv"))

if not nutrition_files:
    raise FileNotFoundError("No CSV files found inside data/raw/nutrition/")

nutrition_dfs = [pd.read_csv(file) for file in nutrition_files]
nutrition_df = pd.concat(nutrition_dfs, ignore_index=True)

nutrition_df = clean_dataframe(nutrition_df)

print("Nutrition columns:", nutrition_df.columns)

# Ensure required columns exist
for col in ["food_name", "calories", "protein"]:
    if col not in nutrition_df.columns:
        nutrition_df[col] = ""

nutrition_df["text"] = (
    "Food: " + nutrition_df["food_name"].astype(str) +
    ", Calories: " + nutrition_df["calories"].astype(str) +
    ", Protein: " + nutrition_df["protein"].astype(str)
)

nutrition_df.to_csv(os.path.join(PROCESSED_PATH, "nutrition.csv"), index=False)



movies_path = os.path.join(DATA_PATH, "imdb-movies-dataset.csv")
movies_df = pd.read_csv(movies_path)

movies_df = clean_dataframe(movies_df)

print("Movies columns:", movies_df.columns)


for col in ["Series_Title", "Genre", "Overview"]:
    if col not in movies_df.columns:
        movies_df[col] = ""

movies_df = movies_df.dropna(subset=["Series_Title", "Overview"])
movies_df = movies_df[
    (movies_df["Series_Title"].str.strip() != "") &
    (movies_df["Overview"].str.strip() != "")
]

movies_df["text"] = (
    "Movie: " + movies_df["Series_Title"].astype(str) +
    " Genre: " + movies_df["Genre"].astype(str) +
    " Description: " + movies_df["Overview"].astype(str)
)

movies_df.to_csv(os.path.join(PROCESSED_PATH, "movies.csv"), index=False)


wiki_path = os.path.join(DATA_PATH, "AllCombined.txt")

texts = []
with open(wiki_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            texts.append(line.strip())

wiki_df = pd.DataFrame({"text": texts})
wiki_df.to_csv(os.path.join(PROCESSED_PATH, "general.csv"), index=False)


print("\n Preprocessing completed successfully!")
print("Files saved in:", PROCESSED_PATH)