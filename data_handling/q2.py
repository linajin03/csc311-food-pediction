import pandas as pd
import re

def clean_Q2(q2):
    """
    Clean column Q2 in csv file, converting to all numerical discrete values.
    Handles various input formats and extracts numerical information.
    """
    # Handle NaN values
    if pd.isna(q2):
        return pd.NA
    
    # Convert to string and strip whitespace
    q2 = str(q2).strip()
    
    # Handle "I don't know" or similar cases
    if any(phrase in q2.lower() for phrase in ['#name?', 'don\'t know', 'dont know', 'no idea']):
        return pd.NA
    
    # Normalize the string
    q2_lower = q2.lower()
    
    # Spelled out number mapping
    spelled_out_numbers = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12
    }
    
    # First, try to extract numbers using regex
    def extract_number(text):
        # Try to find numbers with decimal points or whole numbers
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        
        # If numbers found, return the first one as an integer
        if numbers:
            return int(float(numbers[0]))
        
        # Check for spelled out numbers
        for word, num in spelled_out_numbers.items():
            if word in text.lower():
                return num
        
        return None
    
    # Check for ranges first (prioritize range parsing)
    range_match = re.search(r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)', q2)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return int((low + high) / 2)
    
    # Textual indicators for approximation
    approx_indicators = ['around', 'about', 'approximately', '~', 'roughly']
    
    # Check for approximation with numbers
    for indicator in approx_indicators:
        if indicator in q2_lower:
            num = extract_number(q2)
            if num is not None:
                return num
    
    # Extract direct number
    direct_num = extract_number(q2)
    if direct_num is not None:
        return direct_num
    
    # Handle ingredient lists
    # Split by common delimiters and count unique ingredients
    delimiters = [',', '\n', ' and ', ';']
    for delimiter in delimiters:
        ingredients = [ing.strip() for ing in q2.split(delimiter) if ing.strip()]
        if len(ingredients) > 1:
            return len(set(ingredients))
    
    # If no number or ingredients are found, print and return NA
    print(f"Could not parse Q2 value: {q2}")
    return pd.NA