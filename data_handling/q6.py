import pandas as pd
import numpy as np
import re

# Load dataset
df = pd.read_csv('data/cleaned_data_combined_modified.csv')


# Helpers
def clean_drink_text(text):
    """
    Takes a raw free-text drink string, cleans and maps it to a standard category.
    Returns a string representing the drink category (e.g., 'soda', 'coffee', etc.).
    """
    # Handle missing or NaN
    if pd.isna(text):
        return 'unknown'

    # Convert to lowercase + remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    # Fix common misspellings & unify certain brand names
    text = re.sub(r'coka\s+cola', 'coca cola', text)
    text = re.sub(r'cocacola', 'coca cola', text)

    # Define brand/keyword tp category mappings
    brand_to_category = {
        # Soda
        'coke': 'soda',
        'coca cola': 'soda',
        'cola': 'soda',
        'pepsi': 'soda',
        'sprite': 'soda',
        'fanta': 'soda',
        'mountain dew': 'soda',
        'dr pepper': 'soda',

        # Asian pop
        "ramune": "asian pop",
        "yakult": "asian pop",

        # Energy drinks
        'red bull': 'energy drink',
        'monster': 'energy drink',

        # Alcohol
        'beer': 'alcohol',
        'wine': 'alcohol',
        'saporo': 'alcohol',

        # asian alcohol
        'sake': 'asian alcohol',
        'soju': 'asian alcohol',

        # Hot drinks
        'coffee': 'coffee',
        'espresso': 'coffee',
        'latte': 'coffee',
        'tea': 'tea',

        # soup ? because that's clearly a drink
        "soup": "soup",

        # Other
        'juice': 'juice',
        'water': 'water',
        'milk': 'milk',
        'smoothie': 'smoothie',
        'milkshake': 'milkshake',
    }

    # Check if multiple categories might apply
    # (e.g. user typed "coffee or tea or water")
    # collect them in a set
    categories_found = set()

    # check each key in the text
    for brand, cat in brand_to_category.items():
        if brand in text:
            categories_found.add(cat)

    # If found no matching categories, label as 'other'
    if not categories_found:
        return 'other'

    # If found multiple, join them with a comma:
    return ', '.join(sorted(categories_found))


# Apply the cleaning function
df['Q6_clean'] = df["Q6: What drink would you pair with this food item?"].apply(clean_drink_text)


print(df['Q6_clean'].value_counts())

# Currently I'm choosing to use single category only, can pick the first from the list.
# df['Q6_clean'] = df['Q6_clean'].apply(lambda x: x.split(', ')[0])

# drop or keep the original Q6 column, depending on use case of our later chosen model:
# df.drop(columns=['Q6'], inplace=True)

# Save the cleaned dataset
df.to_csv('cleaned_data_for_model.csv', index=False, na_rep='None')
