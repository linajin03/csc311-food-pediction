"""
test prediction file for neural netowrk implementation


## Data description: 
Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)

Q2: How many ingredients would you expect this food item to contain?

Q3: In what setting would you expect this food to be served? Please check all that apply

Q4: How much would you expect to pay for one serving of this food item?

Q5: What movie do you think of when thinking of this food item?

Q6: What drink would you pair with this food item?

Q7: When you think about this food item, who does it remind you of?

Q8: How much hot sauce would you add to this food item?

Label: What is the food item?
"""

# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

def predict(model, scaler, x):
    """
    Predict the food type using the neural network model.
    """
    # Preprocess the input (scale the features)
    x_scaled = scaler.transform([x])
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    # Perform a forward pass
    with torch.no_grad():
        output = model(x_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Map the predicted class index to the food label
    food_labels = ['Pizza', 'Shawarma', 'Sushi']
    return food_labels[predicted_class]



def predict_all(filename):
    """
    Make predictions for the data in filename using the neural network model.
    """
    # Load the data
    data = pd.read_csv(filename)

    # Preprocess the data
    data = clean_data(data)

    # Select features for prediction (example: Q1, Q2_cleaned, etc.)
    feature_columns = [
    "Q1",
    "Q2",
    "Q4",
    # "Q5: What movie do you think of when thinking of this food item?", Removing for now
    "Q8",
    "Week day lunch",
    "Week day dinner",
    "Weekend lunch",
    "Weekend dinner",
    "At a party",
    "Late night snack",
    "soda",
    "other",
    "tea",
    "alcohol",
    "water",
    "soup",
    "juice",
    "milk",
    "unknown",
    "smoothie",
    "asian alcohol",
    "asian pop",
    "milkshake",
    "Parents",
    "Siblings",
    "Friends",
    "Teachers",
    "Strangers"
    ]
    features = data[feature_columns].fillna(0).values

    # Load the pretrained model and scaler
    input_size = len(feature_columns)
    hidden_size = 16
    output_size = 3
    model = load_model('model.pth', input_size, hidden_size, output_size)
    scaler = StandardScaler()
    scaler.fit(features)  # Ensure the scaler is fitted on training data during training

    # Make predictions
    predictions = []
    for feature in features:
        pred = predict(model, scaler, feature)
        predictions.append(pred)

    return predictions

def clean_Q2(q2):
    """
    Clean column Q2 in csv file, converting to all numerical discrete values.
    Handles:
    - Numerical values (e.g., "5")
    - Ranges (e.g., "4-6")
    - Textual descriptions (e.g., "around 3")
    - Spelled-out numbers (e.g., "three")
    - Ingredient lists (e.g., "bread, meat, cheese")
    - Multi-line ingredient lists (e.g., "I would expect it to contain:\n* Bread\n* Cheese")
    - #NAME? and other non-numeric values
    """
    # Handle NaN values
    if pd.isna(q2):
        return pd.NA
    
    # Convert to string if not already
    q2 = str(q2).strip()
    
    # Handle "I don't know" or similar cases
    if '#NAME?' in q2 or 'don\'t know' in q2.lower() or 'dont know' in q2.lower() or 'no idea' in q2.lower():
        print(f"Skipping Q2 value: {q2}")
        return pd.NA
    
    # Handle ranges like "4-6" or "5 to 7"
    range_match = re.search(r'(\d+)\s*[-~to]+\s*(\d+)', q2)
    if range_match:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        return (low + high) // 2  # Return the floored average
    
    # Handle single numbers
    single_number_match = re.search(r'\d+', q2)
    if single_number_match:
        return int(single_number_match.group(0))
    
    # Handle textual descriptions like "around 5" or "about 3"
    textual_match = re.search(r'(around|about|approximately|~)\s*(\d+)', q2, re.IGNORECASE)
    if textual_match:
        return int(textual_match.group(2))
    
    # Handle cases where the number is spelled out (e.g., "three")
    spelled_out_numbers = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    for word, num in spelled_out_numbers.items():
        if word in q2.lower():
            return num
    
    # Handle ingredient lists
    # Check if the text looks like a list of ingredients (may be an issue for those who answered in sentences)
    if ',' in q2 or '\n' in q2 or ' and ' in q2.lower():
        lines = q2.split('\n')
        ingredients = []
        for line in lines:
            line = re.sub(r'[*\-•]', '', line).strip()
            if line:
                parts = re.split(r'[,&]| and ', line)
                ingredients.extend([part.strip() for part in parts if part.strip()])
        unique_ingredients = set(ingredients)
        return len(unique_ingredients)
    
    # If no number or ingredients are found, return 'n/a'
    print(f"Could not parse Q2 value: {q2}")
    return pd.NA

def one_hot_encode(response_str, categories):
    result = np.zeros(len(categories))
    if isinstance(response_str, str):  # Check if it's a string (not NaN)
        selections = response_str.split(',')
        for i, category in enumerate(categories):
            if category in selections:
                result[i] = 1
    return result

def clean_data(df):
    """
    Clean and format the data in the dataframe
    """
    q1_label = "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)"
    q2_label = "Q2: How many ingredients would you expect this food item to contain?"
    q3_label = "Q3: In what setting would you expect this food to be served? Please check all that apply"
    q4_label = "Q4: How much would you expect to pay for one serving of this food item?"
    q5_label = "Q5: What movie do you think of when thinking of this food item?"
    q6_label = "Q6: What drink would you pair with this food item?"
    q7_label = "Q7: When you think about this food item, who does it remind you of?"
    q8_label = "Q8: How much hot sauce would you add to this food item?"
    
    # Categorical Feature categories
    Q3_categories = ["Week day lunch", "Week day dinner", "Weekend lunch", "Weekend dinner", "At a party", "Late night snack"]
    Q6_categories = [
        "soda", "other", "tea", "alcohol", "water", "soup", "juice", "milk", 
        "unknown", "smoothie", "asian alcohol", "asian pop", "milkshake"
    ]
    Q7_categories = ["Parents", "Siblings", "Friends", "Teachers", "Strangers"]
    Q8_categories = ["None", "A little (mild)", "A moderate amount (medium)", "A lot (hot)", "I will have some of this food item with my hot sauce"]

    # Clean Q2
    df['Q2'] = df[q2_label].apply(clean_Q2)

    # Convert cleaned to numeric
    df['Q2'] = pd.to_numeric(df['Q2'], errors='coerce')

    # Extract columns for Q3, Q6, Q7, and Q8
    q3 = df[q3_label]
    q6 = df[q6_label]
    q7 = df[q7_label]
    q8 = df[q8_label]

    # One hot encode Q3, Q6, and Q7
    q3 = np.array([one_hot_encode(response, Q3_categories) for response in q3])
    q6 = np.array([one_hot_encode(response, Q6_categories) for response in q6])
    q7 = np.array([one_hot_encode(response, Q7_categories) for response in q7])

    # Create new columns for Q3, Q6, and Q7
    for i, category in enumerate(Q3_categories):
        df[category] = q3[:, i]
    for i, category in enumerate(Q6_categories):
        df[category] = q6[:, i]
    for i, category in enumerate(Q7_categories):
        df[category] = q7[:, i]

    # Convert Q8 to ordinal scale (0, 1, 2, 3, 4)
    ordinal_mapping = {category: idx for idx, category in enumerate(Q8_categories)}
    df['Q8'] = df[q8_label].map(ordinal_mapping)

    df = df.rename(columns={q4_label: 'Q4'})
    # Remove dollar signs from Q4
    df["Q4"] = df["Q4"].str.replace("$", "")

    # Drop original categorical columns
    df = df.drop(columns=[
        q3_label, q6_label, q7_label, q8_label
    ])

    # Change Q1 label
    df = df.rename(columns={q1_label: 'Q1'})
    # Make sure its numeric
    df['Q1'] = pd.to_numeric(df['Q1'], errors='coerce')

    return df

# Define a simple feedforward neural network
class FoodPredictionNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FoodPredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load the pretrained model
def load_model(model_path, input_size, hidden_size, output_size):
    model = FoodPredictionNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# For testing
if __name__ == "__main__":
    filename = "data/cleaned_data.csv"
    predictions = predict_all(filename)
    print(predictions)  

