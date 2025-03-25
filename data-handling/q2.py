import pandas as pd
def clean_Q2(q2):
    """
    Clean column Q2 in csv file, converting to all numerical discrete values.
    Handles:
    - Numerical values (e.g., "5", "6.0")
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
    if '-' in q2 or 'to' in q2 or '~' in q2:
        parts = q2.replace('to', '-').replace('~', '-').split('-')
        if len(parts) == 2 and parts[0].strip().replace('.', '', 1).isdigit() and parts[1].strip().replace('.', '', 1).isdigit():
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return int((low + high) // 2)  # Return the floored average
        
    # Handle single numbers (integers or floats)
    if q2.replace('.', '', 1).isdigit():  # Check if it's a number (including floats)
        return int(float(q2))  # Convert to float first, then to integer

    # Handle textual descriptions like "around 5" or "about 3"
    textual_indicators = ["around", "about", "approximately", "~"]
    for indicator in textual_indicators:
        if indicator in q2.lower():
            parts = q2.split()
            for part in parts:
                if part.replace('.', '', 1).isdigit():  # Check if part is a number (including floats)
                    return int(float(part))  # Convert to float first, then to integer
    
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
            line = line.replace('*', '').replace('-', '').replace('•', '').strip()
            if line:
                parts = [part.strip() for part in line.replace(' and ', ',').split(',')]
                ingredients.extend([part for part in parts if part])
        unique_ingredients = set(ingredients)
        return len(unique_ingredients)
    
    # If no number or ingredients are found, return 'n/a'
    print(f"Could not parse Q2 value: {q2}")
    return pd.NA
