
import pandas as pd
import numpy as np
from movie_genre_dict import movie_genres
import pandas as pd
import numpy as np


# Multi-select categories
Q3_cats = ["Week day lunch", "Week day dinner", "Weekend lunch", "Weekend dinner", "At a party", "Late night snack"]
Q6_cats = ["soda", "other", "tea", "alcohol", "water", "soup", "juice", "milk", "unknown", "smoothie", "asian alcohol", "asian pop", "milkshake"]
Q7_cats = ["Parents", "Siblings", "Friends", "Teachers", "Strangers"]
Q8_cats = ["None", "A little (mild)", "A moderate amount (medium)", "A lot (hot)", "I will have some of this food item with my hot sauce"]

# --- Encoding Functions ---
def one_hot_encode_column(df, col, categories):
    one_hot = np.zeros((len(df), len(categories)))
    for i, val in enumerate(df[col].fillna("Unknown")):
        items = [item.strip() for item in val.split(",")]
        for j, cat in enumerate(categories):
            if cat in items:
                one_hot[i, j] = 1
    for j, cat in enumerate(categories):
        df[f"{col}_{cat}"] = one_hot[:, j]
    df.drop(columns=[col], inplace=True)

def normalize_column(df, col):
    #filter out negative values
    mask = (df[col] >= 0)
    filtered_df = df.loc[mask, col]
    mean = filtered_df.mean()
    std = filtered_df.std()
    df.loc[mask, col] = (df.loc[mask, col] - mean) / std

def categorize_movie_genre(movie_title):
    if movie_title is None or str(movie_title).lower() in ['none', 'nan'] or not str(movie_title).strip():
        return "no_movie"
    normalized = str(movie_title)
    if normalized in movie_genres:
        return movie_genres[normalized]
    for title, genre in movie_genres.items():
        if title in normalized or normalized in title:
            return genre
    return "other"

def encode_genres(df):
    df["Q5_genres"] = df["Q5"].apply(categorize_movie_genre)
    all_genres = set()
    for entry in df["Q5_genres"]:
        all_genres.update([g.strip() for g in str(entry).split(",")])
    for genre in all_genres:
        df[f"genre_{genre}"] = df["Q5_genres"].apply(lambda x: int(genre in str(x).split(",")))
    df.drop(columns=["Q5", "Q5_genres"], inplace=True)

def encode_label(df):
    label_map = {val: i for i, val in enumerate(sorted(df["Label"].dropna().unique()))}
    df["Label"] = df["Label"].map(label_map)
    return label_map

def process(path, output_path="final_processed.csv", for_bow=False):
    df = pd.read_csv(path)

    if for_bow:
        # concatenate all free-response Q1–Q8 answers into a single text blob
        df["combined_text"] = df[[f"Q{i}" for i in range(1, 9)]].astype(str).agg(" ".join, axis=1)
        df = df[["combined_text", "Label"]]
    else:
        normalize_column(df, "Q1")
        normalize_column(df, "Q2")
        normalize_column(df, "Q4")
        one_hot_encode_column(df, "Q3", Q3_cats)
        one_hot_encode_column(df, "Q6", Q6_cats)
        one_hot_encode_column(df, "Q7", Q7_cats)
        one_hot_encode_column(df, "Q8", Q8_cats)
        encode_genres(df)
        pass

    #encode_label(df)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    process("data/cleaned_data.csv", "data/final_processed.csv", for_bow=False)
    process("data/cleaned_data.csv", "data/bow_processed.csv", for_bow=True)

