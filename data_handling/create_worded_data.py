import pandas as pd
import re
from movie_genre_dict import movie_genres  # update path if needed

# --- Paste the functions you already have ---
# categorize_complexity, categorize_ingredients, categorize_price, categorize_movie_genre

# Function to categorize Q1
def categorize_complexity(value):
    try:
        value = int(value)
        return ["very simple", "simple", "moderate", "complex", "very complex"][value - 1]
    except:
        return "unknown"

# Function to categorize Q2
def categorize_ingredients(value):
    try:
        value_str = str(value).lower()
        if "-" in value_str:
            parts = value_str.split("-")
            value = (int(parts[0].strip()) + int(parts[1].strip())) / 2
        else:
            match = re.search(r'(\d+)', value_str)
            if match:
                value = int(match.group(1))
            else:
                return "unknown"
        if value <= 3:
            return "very few"
        elif value <= 6:
            return "few"
        elif value <= 10:
            return "moderate"
        elif value <= 20:
            return "many"
        else:
            return "very many"
    except:
        return "unknown"

# Function to categorize Q4
def categorize_price(value):
    try:
        value_str = str(value)
        match = re.search(r'(\d+(\.\d+)?)', value_str)
        if not match:
            return "unknown"
        value = float(match.group(1))
        if value <= 5:
            return "very cheap"
        elif value <= 10:
            return "cheap"
        elif value <= 15:
            return "moderate"
        elif value <= 25:
            return "expensive"
        else:
            return "very expensive"
    except:
        return "unknown"

# Movie → Genre
def categorize_movie_genre(movie_title):
    if pd.isna(movie_title) or str(movie_title).strip().lower() in ['none', 'nan', '']:
        return "no_movie"
    movie_title = str(movie_title).strip()
    if movie_title in movie_genres:
        return movie_genres[movie_title]
    for title, genre in movie_genres.items():
        if title.lower() in movie_title.lower():
            return genre
    return "other"

# === Main processing function ===
def clean_data(input_file, output_file, delimiter=','):
    try:
        df = pd.read_csv(input_file, delimiter=delimiter)

        # Auto-detect relevant Q columns
        q1_col = next((col for col in df.columns if "Q1" in col), None)
        q2_col = next((col for col in df.columns if "Q2" in col), None)
        q4_col = next((col for col in df.columns if "Q4" in col), None)
        q5_col = next((col for col in df.columns if "Q5" in col), None)

        if not all([q1_col, q2_col, q4_col, q5_col]):
            print("Required columns not found (Q1–Q5).")
            return

        print("Cleaning columns...")
        df[q1_col] = df[q1_col].apply(categorize_complexity)
        df[q2_col] = df[q2_col].apply(categorize_ingredients)
        df[q4_col] = df[q4_col].apply(categorize_price)
        df[q5_col] = df[q5_col].apply(categorize_movie_genre)

        # Save a word-based CSV (Naive Bayes compatible)
        word_columns = ['id', q1_col, q2_col, 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Label']
        existing = [col for col in word_columns if col in df.columns]
        df[existing].to_csv(output_file, index=False)
        print(f"Saved worded data to {output_file}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    clean_data("data/cleaned_data.csv", "data/worded_data.csv")
