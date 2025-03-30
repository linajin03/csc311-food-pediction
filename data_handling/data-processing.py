# data_processing.py

import pandas as pd
import numpy as np

def process_dt(raw_path, output_path="../data/final_processed_dt.csv"):
    """
    Reads the raw file (with Q1..Q8 in text form) and produces a CSV
    suitable for the Decision Tree approach, using partial manual
    encoding for Q3 and Q7, and an ordinal numeric column for Q8.
    """
    df = pd.read_csv(raw_path)
    print("=== Decision Tree Processing ===")
    print("Raw columns:", df.columns.tolist())

    # 1) Rename columns to short "Q1..Q8"
    df.rename(
        columns={
            "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": "Q1",
            "Q2: How many ingredients would you expect this food item to contain?": "Q2",
            "Q3: In what setting would you expect this food to be served? Please check all that apply": "Q3",
            "Q4: How much would you expect to pay for one serving of this food item?": "Q4",
            "Q5: What movie do you think of when thinking of this food item?": "Q5",
            "Q6: What drink would you pair with this food item?": "Q6",
            "Q7: When you think about this food item, who does it remind you of?": "Q7",
            "Q8: How much hot sauce would you add to this food item?": "Q8",
        },
        inplace=True
    )

    # 2) Convert Q2 and Q4 to numeric
    #    remove '$' or other extra characters
    df["Q2"] = df["Q2"].astype(str)
    # e.g. parse "6-7" or "3 or more" -> you can do your existing logic or keep it simple
    # For example, if you want to remove non-digits except dashes:
    # or skip advanced logic and just attempt float parse:
    def parse_q2(val):
        import re
        # e.g. handle range "6-7"
        match = re.search(r"(\d+)\s*-\s*(\d+)", val)
        if match:
            lo, hi = match.groups()
            return (float(lo)+float(hi))/2
        # handle "3 or more"
        match2 = re.search(r"(\d+)", val)
        if match2:
            return float(match2.group(1))
        return np.nan

    df["Q2"] = df["Q2"].apply(parse_q2)

    df["Q4"] = df["Q4"].astype(str).str.replace("$","",regex=False)
    df["Q4"] = pd.to_numeric(df["Q4"], errors="coerce")

    # 3) Convert Q8 to ordinal 0..4
    q8_map = {
        "None": 0,
        "A little (mild)": 1,
        "A moderate amount (medium)": 2,
        "A lot (hot)": 3,
        "I will have some of this food item with my hot sauce": 4
    }
    df["Q8"] = df["Q8"].map(q8_map)
    # fill missing with -1 or drop
    df["Q8"] = df["Q8"].fillna(-1)

    # 4) We keep Q3, Q6, Q7 as text for manual "1 if 'Week day lunch' in str(q3)" approach
    # no expansions for decision tree

    # 5) Drop any row with missing Q1, Q2, Q4, Q8 or Label
    df = df.dropna(subset=["Q1","Q2","Q4","Q8","Label"])
    df.to_csv(output_path, index=False)
    print(f"Saved DT data to {output_path}")

def process_nn(raw_path, output_path="../data/final_processed_nn.csv"):
    """
    Reads raw CSV (with Q1..Q8 in text form),
    does numeric parse for Q2,Q4, and reindexes one-hot expansions
    so that forced columns like 'Q6_other', 'Q8_None', 'genre_Thriller' etc.
    always appear in the final CSV (even if they're all zeros).
    """
    import pandas as pd
    import numpy as np
    import re

    df = pd.read_csv(raw_path)
    print("=== Neural Network Processing with forced reindex ===")
    print("Raw columns:", df.columns.tolist())

    # 1) Rename columns to Q1..Q8
    df.rename(
        columns={
            "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": "Q1",
            "Q2: How many ingredients would you expect this food item to contain?": "Q2",
            "Q3: In what setting would you expect this food to be served? Please check all that apply": "Q3",
            "Q4: How much would you expect to pay for one serving of this food item?": "Q4",
            "Q5: What movie do you think of when thinking of this food item?": "Q5",
            "Q6: What drink would you pair with this food item?": "Q6",
            "Q7: When you think about this food item, who does it remind you of?": "Q7",
            "Q8: How much hot sauce would you add to this food item?": "Q8",
        },
        inplace=True
    )

    # 2) Parse Q2, Q4 numerically (example logic)
    def parse_q2(val):
        match = re.search(r"(\d+)\s*-\s*(\d+)", str(val))
        if match:
            lo, hi = match.groups()
            return (float(lo)+float(hi))/2
        match2 = re.search(r"(\d+)", str(val))
        if match2:
            return float(match2.group(1))
        return np.nan

    df["Q2"] = df["Q2"].apply(parse_q2)

    df["Q4"] = df["Q4"].astype(str).str.replace("$","",regex=False)
    df["Q4"] = pd.to_numeric(df["Q4"], errors="coerce")

    # -----------------------------------------------------------------------
    #   PART A: Q3 -> one-hot with forced columns
    # -----------------------------------------------------------------------
    # Suppose you want these 6 possible categories for Q3:
    Q3_cats = [
        "Week day lunch",
        "Week day dinner",
        "Weekend lunch",
        "Weekend dinner",
        "At a party",
        "Late night snack"
    ]
    dummy_q3 = df["Q3"].str.get_dummies(sep=',')
    # reindex to ensure all 6 appear:
    dummy_q3 = dummy_q3.reindex(columns=Q3_cats, fill_value=0)
    # rename columns to "Q3_..."
    dummy_q3.columns = ["Q3_"+col for col in dummy_q3.columns]
    df = pd.concat([df, dummy_q3], axis=1)
    df.drop(columns=["Q3"], inplace=True)

    # -----------------------------------------------------------------------
    #   PART B: Q6 -> one-hot with forced columns
    # -----------------------------------------------------------------------
    # Suppose you want these columns:
    Q6_cats = [
        "soda","other","tea","alcohol","water","soup","juice","milk","unknown",
        "smoothie","asian alcohol","asian pop","milkshake"
    ]
    dummy_q6 = df["Q6"].str.get_dummies(sep=',')
    # We unify or strip spaces:
    dummy_q6.columns = dummy_q6.columns.str.strip().str.lower()
    # Remove duplicates
    dummy_q6 = dummy_q6.groupby(level=0, axis=1).max()
    # Then reindex to ensure all Q6_cats appear:
    dummy_q6 = dummy_q6.reindex(columns=Q6_cats, fill_value=0)
    # rename columns to e.g. "Q6_soda", "Q6_other"
    rename_q6 = {cat: f"Q6_{cat}" for cat in Q6_cats}
    dummy_q6.rename(columns=rename_q6, inplace=True)

    df = pd.concat([df, dummy_q6], axis=1)
    df.drop(columns=["Q6"], inplace=True)

    # -----------------------------------------------------------------------
    #   PART C: Q7 -> one-hot. Example
    # -----------------------------------------------------------------------
    Q7_cats = ["Parents","Siblings","Friends","Teachers","Strangers"]
    dummy_q7 = df["Q7"].str.get_dummies(sep=',')
    dummy_q7 = dummy_q7.reindex(columns=Q7_cats, fill_value=0)
    dummy_q7.columns = ["Q7_"+col for col in dummy_q7.columns]
    df = pd.concat([df, dummy_q7], axis=1)
    df.drop(columns=["Q7"], inplace=True)

    # -----------------------------------------------------------------------
    #   PART D: Q8 -> one-hot with forced columns
    # -----------------------------------------------------------------------
    Q8_cats = [
        "None",
        "A little (mild)",
        "A moderate amount (medium)",
        "A lot (hot)",
        "I will have some of this food item with my hot sauce"
    ]
    dummy_q8 = df["Q8"].str.get_dummies()
    # reindex for them:
    dummy_q8 = dummy_q8.reindex(columns=Q8_cats, fill_value=0)
    # rename
    rename_q8 = {
        "None": "Q8_None",
        "A little (mild)": "Q8_A little (mild)",
        "A moderate amount (medium)": "Q8_A moderate amount (medium)",
        "A lot (hot)": "Q8_A lot (hot)",
        "I will have some of this food item with my hot sauce": "Q8_I will have some of this food item with my hot sauce"
    }
    dummy_q8.rename(columns=rename_q8, inplace=True)
    df = pd.concat([df, dummy_q8], axis=1)
    df.drop(columns=["Q8"], inplace=True)

    # -----------------------------------------------------------------------
    #   PART E: Q5 -> "genre_" columns with forced categories
    # -----------------------------------------------------------------------
    # Suppose you want a bunch of genres like "Thriller","Fantasy","Mystery", etc.
    # We'll do a short example. Expand or edit as needed.
    Q5_genres = [
        "Thriller", "Fantasy", "Mystery", "Animation", "Comedy", "Horror",
        "Documentary", "Sports", "Action", "Romance", "Superhero", "Music",
        "Family", "Crime", "Musical", "Drama", "Adventure", "Political",
        "Sci-Fi", "other"
    ]
    dummy_q5 = df["Q5"].str.get_dummies(sep=',')
    # unify spacing or case if needed:
    dummy_q5.columns = dummy_q5.columns.str.strip().str.lower()
    dummy_q5 = dummy_q5.groupby(level=0, axis=1).max()
    # reindex so these 20 genres always appear:
    dummy_q5 = dummy_q5.reindex(columns=Q5_genres, fill_value=0)
    # rename columns to "genre_Thriller", etc.
    rename_q5 = {g: f"genre_{g}" for g in Q5_genres}
    dummy_q5.rename(columns=rename_q5, inplace=True)

    df = pd.concat([df, dummy_q5], axis=1)
    df.drop(columns=["Q5"], inplace=True)

    # -----------------------------------------------------------------------
    #   PART F: Drop rows missing Q1,Q2,Q4,Label, etc. Then optionally standardize Q2,Q4
    # -----------------------------------------------------------------------
    df = df.dropna(subset=["Q1","Q2","Q4","Label"])

    # Example: standardize Q2, Q4
    df["Q2"] = (df["Q2"] - df["Q2"].mean())/df["Q2"].std()
    df["Q4"] = (df["Q4"] - df["Q4"].mean())/df["Q4"].std()

    df.to_csv(output_path, index=False)
    print(f"Saved NN data (with forced reindex) to {output_path}")
    return df

def process_nb(raw_path, output_path="../data/bow_processed_nb.csv"):
    """
    If you want to produce a single combined_text for naive bayes:
    """
    df = pd.read_csv(raw_path)
    print("=== NB Step (BoW) ===")
    print("Raw columns:", df.columns.tolist())

    # rename
    df.rename(
        columns={
            "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": "Q1",
            "Q2: How many ingredients would you expect this food item to contain?": "Q2",
            "Q3: In what setting would you expect this food to be served? Please check all that apply": "Q3",
            "Q4: How much would you expect to pay for one serving of this food item?": "Q4",
            "Q5: What movie do you think of when thinking of this food item?": "Q5",
            "Q6: What drink would you pair with this food item?": "Q6",
            "Q7: When you think about this food item, who does it remind you of?": "Q7",
            "Q8: How much hot sauce would you add to this food item?": "Q8",
        },
        inplace=True
    )
    # create combined_text
    df["combined_text"] = df[["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]].astype(str).agg(" ".join, axis=1)
    df = df[["combined_text","Label"]].dropna()
    df.to_csv(output_path, index=False)
    print(f"Created BoW data at {output_path}")

if __name__ == "__main__":
    # Example usage for each:
    process_dt("../data/cleaned_data_combined_modified.csv", "../data/final_processed_dt.csv")
    process_nn("../data/cleaned_data_combined_modified.csv", "../data/final_processed_nn.csv")
    process_nb("../data/cleaned_data_combined_modified.csv", "../data/bow_processed_nb.csv")
