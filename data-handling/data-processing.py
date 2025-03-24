import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define known multi-select categories
Q3_categories = ["Week day lunch", "Week day dinner", "Weekend lunch", "Weekend dinner", "At a party", "Late night snack"]
Q6_categories = [
    "soda", "other", "tea", "alcohol", "water", "soup", "juice", "milk",
    "unknown", "smoothie", "asian alcohol", "asian pop", "milkshake"
]
Q7_categories = ["Parents", "Siblings", "Friends", "Teachers", "Strangers"]
Q8_categories = ["None", "A little (mild)", "A moderate amount (medium)", "A lot (hot)", "I will have some of this food item with my hot sauce"]

def one_hot_encode(response_str, categories):
    result = np.zeros(len(categories))
    if isinstance(response_str, str):
        selections = response_str.split(',')
        for i, category in enumerate(categories):
            if category in selections:
                result[i] = 1
    return result

def clean_and_encode(input_path, output_path=None):
    df = pd.read_csv(input_path)

    # Fill missing string fields
    for col in ["Q3", "Q5", "Q6", "Q7", "Q8"]:
        df[col] = df[col].fillna("Unknown")

    # One-hot encode Q3, Q6, Q7, Q8
    for question, categories in [("Q3", Q3_categories), ("Q6", Q6_categories),
                                 ("Q7", Q7_categories), ("Q8", Q8_categories)]:
        encoded = np.array([one_hot_encode(val, categories) for val in df[question]])
        for i, cat in enumerate(categories):
            df[f"{question}_{cat}"] = encoded[:, i]
        df.drop(columns=[question], inplace=True)

    # Label encode Q5, Q8 (again for backup)
    for col in ["Q5", "Q8"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Handle missing numeric and normalize
    if "Q4" in df.columns:
        df["Q4"] = df["Q4"].replace({"\$": ""}, regex=True).replace("Unknown", np.nan)
        df["Q4"] = pd.to_numeric(df["Q4"], errors="coerce")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop("Label", errors='ignore')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode label
    if "Label" in df.columns:
        df["Label"] = df["Label"].fillna("Unknown")
        label_encoder = LabelEncoder()
        df["Label"] = label_encoder.fit_transform(df["Label"].astype(str))

    if output_path:
        df.to_csv(output_path, index=False)

    return df

def split_data(df, test_size=0.2):
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return train_test_split(X, y, test_size=test_size, random_state=42)

if __name__ == "__main__":
    df = clean_and_encode("data/cleaned_data.csv", "data/final_processed.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    print("Training size:", len(X_train), "| Test size:", len(X_test))
