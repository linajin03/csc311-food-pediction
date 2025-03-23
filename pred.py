"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")


## Data description: 
Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)

Q2: How many ingredients would you expect this food item to contain?

Q3: In what setting would you expect this food to be served? Please check all that apply

Q4: How much would you expect to pay for one serving of this food item?

Q5: What movie do you think of when thinking of this food item?

Q6: What drink would you pair with this food item?

Q7: When you think about this food item, who does it remind you of?

Q8: How much hot sauce would you add to this food item?

Label
"""

# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import numpy
import pandas as pd


def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    # randomly choose between the three choices: 'Pizza', 'Shawarma', 'Sushi'.
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the food items!
    y = random.choice(['Pizza', 'Shawarma', 'Sushi'])

    # return the prediction
    return y


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    data = csv.DictReader(open(filename))

    data = pd.read_csv("data/cleaned_data_combined_modified.csv")

    # Clean Q2
    data['Q2_cleaned'] = data['Q2: How many ingredients would you expect this food item to contain?'].apply(clean_Q2)

    # Convert cleaned to numeric
    data['Q2_cleaned'] = pd.to_numeric(data['Q2_cleaned'], errors='coerce')

    predictions = []
    for test_example in data:
        # obtain a prediction for this test example
        pred = predict(test_example)
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


# For testing
if __name__ == "__main__":
    df = pd.read_csv('data/cleaned_data_combined_modified.csv')
    df['Q2_cleaned'] = df['Q2: How many ingredients would you expect this food item to contain?'].apply(clean_Q2)

    # Convert cleaned to numeric
    df['Q2_cleaned'] = pd.to_numeric(df['Q2_cleaned'], errors='coerce')

    # Check for missing values
    print("Missing values in Q2 column:", df['Q2_cleaned'].isna().sum())