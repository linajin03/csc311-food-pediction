import re

def clean_price(price_str):
    
    if not price_str: 
        return -1
    
    price_str = str(price_str).strip()
    
    # When it is just a number
    if price_str.isdigit():
        return int(price_str)
    
    # Find all the numbers
    numbers = re.findall(r'\d+\.?\d*', price_str)
    
    if not numbers:
        return -1
    
    if len(numbers) >= 2:
        # finding the median 
        num1 = float(numbers[0])
        num2 = float(numbers[1])
        median = (num1 + num2) / 2
        return round(median)
    
    # whole numbers
    return round(float(numbers[0]))

