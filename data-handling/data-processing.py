import pandas as pd
import numpy as np
from movie_genre_dict import movie_genres

# Multi-select categories (used in both branches)
Q3_cats = ["Week day lunch", "Week day dinner", "Weekend lunch", "Weekend dinner", "At a party", "Late night snack"]
Q6_cats = ["soda", "other", "tea", "alcohol", "water", "soup", "juice", "milk", "unknown", "smoothie", "asian alcohol",
           "asian pop", "milkshake"]
Q7_cats = ["Parents", "Siblings", "Friends", "Teachers", "Strangers"]
Q8_cats = ["None", "A little (mild)", "A moderate amount (medium)", "A lot (hot)",
           "I will have some of this food item with my hot sauce"]

# Import cleaning functions for NN branch
# (Make sure final_docs is a package with an __init__.py file)
from final_docs.final_cleaning import clean as final_clean


# Naive Bayes–specific cleaning helper: use categorize_movie_genre instead of clean_movie_text
def categorize_movie_genre(movie_title):
    if movie_title is None or str(movie_title).lower() in ['none', 'nan'] or not str(movie_title).strip():
        return "no_movie"
    normalized = str(movie_title).strip()
    if normalized in movie_genres:
        return movie_genres[normalized]
    for title, genre in movie_genres.items():
        if title in normalized or normalized in title:
            return genre
    return "other"


# Naive Bayes version of processing (without encoding Q5 into dummies)
def process_nb(path, output_path, for_bow):
    df = pd.read_csv(path)
    # Rename columns as in your final cleaning
    df.rename(columns={
        'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': 'Q1',
        'Q2: How many ingredients would you expect this food item to contain?': 'Q2',
        'Q3: In what setting would you expect this food to be served? Please check all that apply': 'Q3',
        'Q4: How much would you expect to pay for one serving of this food item?': 'Q4',
        'Q5: What movie do you think of when thinking of this food item?': 'Q5',
        'Q6: What drink would you pair with this food item?': 'Q6',
        'Q7: When you think about this food item, who does it remind you of?': 'Q7',
        'Q8: How much hot sauce would you add to this food item?': 'Q8'
    }, inplace=True)
    # Clean numeric columns
    from final_docs.final_cleaning import clean_Q2, clean_price, clean_drink_text  # reuse these functions
    df['Q2'] = df['Q2'].apply(clean_Q2)
    df['Q2'] = pd.to_numeric(df['Q2'], errors='coerce')
    df["Q4"] = df["Q4"].apply(clean_price)
    # For Q5, use categorize_movie_genre for NB
    if 'Q5' in df.columns:
        df["Q5"] = df["Q5"].apply(categorize_movie_genre)
    df["Q6"] = df["Q6"].apply(clean_drink_text)
    # (Assume Q1, Q3, Q7, Q8 remain as is for now)
    df = df.dropna(subset=["Q1", "Q2", "Q4", "Q5", "Q6", "Q7", "Q8"])

    if for_bow:
        # For BoW approach, combine Q1–Q8 into a single text blob.
        df["combined_text"] = df[[f"Q{i}" for i in range(1, 9)]].astype(str).agg(" ".join, axis=1)
        df = df[["combined_text", "Label"]]
    else:
        # Otherwise, you can call your NB-specific encoding (if you have one) or leave Q5 intact.
        # For now, let's assume you leave it as is.
        pass
    df.to_csv(output_path, index=False)
    print(f"Saved processed NB data to {output_path}")
    return df


# Main process function that branches on model_type.
def process(path, output_path, for_bow=False, model_type="nn"):
    if model_type == "nn":
        # Use the final_clean and encode_data functions from final_cleaning.py
        df = final_clean(path)  # This cleans and encodes data for neural network
    elif model_type == "nb":
        df = process_nb(path, output_path, for_bow)
        return df  # Already saved in process_nb
    else:
        raise ValueError("model_type must be either 'nn' or 'nb'")

    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    return df


if __name__ == "__main__":
    # Example calls:
    # For neural network processing:
    process("../data/cleaned_data.csv", "../data/final_processed_nn.csv", for_bow=False, model_type="nn")
    # For naive bayes processing (BoW branch):
    process("../data/cleaned_data.csv", "../data/bow_processed_nb.csv", for_bow=True, model_type="nb")
