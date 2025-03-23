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

Label: 
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

def one_hot_encode(response_str, categories):
    result = np.zeros(len(categories))
    if isinstance(response_str, str):  # Check if it's a string (not NaN)
        selections = response_str.split(',')
        for i, category in enumerate(categories):
            if category in selections:
                result[i] = 1
    return result

def categorize_movie_genre(movie_title):
    # Dictionary mapping movie titles to genres
    movie_genres = {
        "Cloudy With A Chance Of Meatballs": "Animation,Comedy",
        "None": "Unknown",
        "Action Movie": "Action",
        "Mamma Mia": "Musical,Comedy",
        "Dragon": "Animation",
        "Rick And Morty": "Animation,Comedy",
        "Home Alone": "Comedy,Family",
        "The Big Lebowski": "Comedy,Crime",
        "Spider-Man": "Superhero,Action",
        "Goodfellas": "Crime,Drama",
        "Dead Silence": "Horror,Thriller",
        "La La Land": "Musical,Drama,Romance",
        "Harry Potter": "Fantasy,Adventure",
        "Transformer": "Action,Sci-Fi",
        "Teenage Mutant Ninja Turtles": "Animation,Action",
        "High School Musical": "Musical,Comedy,Family",
        "Despicable Me": "Animation,Comedy",
        "Toy Story": "Animation,Adventure",
        "The Godfather": "Crime,Drama",
        "Fast And Furious": "Action,Thriller",
        "The Garfield Movie": "Animation,Comedy",
        "Ratatouille": "Animation,Comedy",
        "Five Nights At Freddies": "Horror",
        "Back To The Future": "Sci-Fi,Adventure",
        "Mystic Pizza": "Comedy,Drama,Romance",
        "Iron Man": "Superhero,Action",
        "Life Of Pi": "Adventure,Drama",
        "Avengers": "Superhero,Action",
        "King Kong": "Action,Adventure,Drama",
        "Whiplash": "Drama,Music",
        "Inside Out": "Animation,Drama",
        "Superbad": "Comedy",
        "Stranger Things": "Sci-Fi,Horror,Drama",
        "Air Bud": "Family,Comedy",
        "Deadpool": "Action,Comedy",
        "A Quiet Place": "Horror,Thriller",
        "Finding Nemo": "Animation,Adventure",
        "My Cousin Vinny": "Comedy,Crime",
        "Eat Pray Love": "Drama,Romance",
        "Pulp Fiction": "Crime,Thriller",
        "Star Wars": "Sci-Fi,Adventure",
        "Batman": "Superhero,Action",
        "Shrek": "Animation,Comedy",
        "Scooby Doo": "Animation,Adventure",
        "Princess Diaries": "Comedy,Family,Romance",
        "Wall-E": "Animation,Sci-Fi",
        "The Hangover": "Comedy",
        "Breaking Bad": "Crime,Drama",
        "Interstellar": "Sci-Fi,Adventure,Drama",
        "Rush Hour": "Action,Comedy",
        "The Truman Show": "Drama,Sci-Fi",
        "Futurama": "Animation,Comedy,Sci-Fi",
        "Godfather": "Crime,Drama",
        "The Dictator": "Comedy,Political",
        "Borat": "Comedy,Political",
        "Mission: Impossible": "Action,Thriller",
        "Avengers: Endgame": "Superhero,Action",
        "Titanic": "Drama,Romance",
        "Dangal": "Drama,Sports",
        "Kung Fu Panda": "Animation,Action",
        "The Mummy": "Action,Adventure",
        "The Invisible Guest": "Thriller,Mystery",
        "Squid Game": "Drama,Thriller",
        "Parasite": "Thriller,Drama",
        "Blade Runner": "Sci-Fi,Thriller",
        "Spider-Man: Into The Spider-Verse": "Animation,Action",
        "Everything Everywhere All At Once": "Sci-Fi,Action,Adventure",
        "Barbie": "Comedy,Family",
        "Jurassic Park": "Adventure,Sci-Fi",
        "Ponyo": "Animation,Fantasy",
        "My Neighbor Totoro": "Animation,Family",
        "Kill Bill": "Action,Thriller",
        "Jiro Dreams Of Sushi": "Documentary",
        "Naruto": "Animation,Action",
        "Frozen": "Animation,Adventure,Family",
        "Shawshank Redemption": "Drama",
        "Mad Max": "Action,Sci-Fi",
        "The Lion King": "Animation,Adventure,Drama",
        "Your Name": "Animation,Romance,Fantasy",
        "Memoirs Of A Geisha": "Drama,Romance",
        "Godzilla": "Action,Sci-Fi",
        "Shazam": "Superhero,Action",
        "The Grinch": "Animation,Comedy,Family",
        "Zootopia": "Animation,Adventure,Comedy",
        "The Godfather Part II": "Crime,Drama",
        "The Social Network": "Drama",
        "The Big Sick": "Comedy,Drama,Romance",
        "Die Hard": "Action,Thriller",
        "Taxi Driver": "Crime,Drama,Thriller",
        "Fast & Furious": "Action,Thriller",
        "The Karate Kid": "Drama,Family",
        "John Wick": "Action,Thriller",
        "Bladerunner": "Sci-Fi,Thriller",
        "Parasite": "Thriller,Drama",
        "Gone Girl": "Thriller,Drama",
        "Inception": "Sci-Fi,Thriller",
        "The Breakfast Club": "Comedy,Drama",
        "The Lego Movie": "Animation,Adventure,Comedy",
        "Spider-Man: Far From Home": "Superhero,Action",
        "Space Jam": "Animation,Comedy",
        "Spongebob": "Animation,Comedy",
        "Toy Story 4": "Animation,Adventure",
        "Green Book": "Drama",
        "Madagascar": "Animation,Adventure,Comedy",
        "The Mario Movie": "Animation,Adventure",
        "The Big Lebowski": "Comedy,Crime",
        "The Road To Fallujah": "Documentary",
        "Kingdom Of Heaven": "Action,Adventure,Drama",
        "The Dictator": "Comedy,Political",
        "Rush Hour": "Action,Comedy",
        "Breaking Bad": "Crime,Drama",
        "The Boys": "Superhero,Action",
        "Wicked": "Musical,Fantasy",
        "Eternal Sunshine Of The Spotless Mind": "Drama,Romance,Sci-Fi",
        "The Grinch": "Animation,Comedy,Family",
        "Ratatouille": "Animation,Comedy",
        "Star Wars: The Last Jedi": "Sci-Fi,Action",
        "The Dictator": "Comedy",
        "The Lego Movie 2": "Animation,Comedy,Adventure",
        "Dune": "Sci-Fi,Adventure"
    }

    # Handle None or empty strings
    if movie_title is None or str(movie_title).lower() in ['none', 'nan'] or not str(movie_title).strip():
        return "no_movie"

    # Normalize the title for comparison
    normalized_title = str(movie_title)

    # Check for direct matches
    if normalized_title in movie_genres:
        return movie_genres[normalized_title]

    # Check for partial matches
    for title, genre in movie_genres.items():
        if title in normalized_title or normalized_title in title:
            return genre

    # If no match is found
    return "other"

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
    # df['Q2'] = df[q2_label].apply(clean_Q2)

    # Convert cleaned to numeric
    # df['Q2'] = pd.to_numeric(df['Q2'], errors='coerce')

    # Extract columns for Q3, Q6, Q7, and Q8
    q3 = df[q3_label]
    q5 = df[q5_label]
    q6 = df[q6_label]
    q7 = df[q7_label]
    q8 = df[q8_label]

    df[q5_label] = q5.apply(categorize_movie_genre)
    df_genres = df[q5_label].str.get_dummies(sep=',')

    # Concatenate the new genre columns with the original DataFrame
    df = pd.concat([df, df_genres], axis=1)
    df = df.drop(columns=[q5_label])

    # One hot encode Q3, Q5, Q6, and Q7
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

    # Change labels of Q1, Q2, and Q4
    df = df.rename(columns={q1_label: 'Q1', q2_label: 'Q2', q4_label: 'Q4'})
    
    # Make sure its an integer
    df['Q1'] = df['Q1'].astype(int, errors='ignore')
    df['Q2'] = df['Q2'].astype(int, errors='ignore')
    df['Q4'] = df['Q4'].astype(int, errors='ignore')

    return df

# For testing
if __name__ == "__main__":
    df = pd.read_csv('data/cleaned_data_combined_modified.csv')
    df['Q2_cleaned'] = df['Q2: How many ingredients would you expect this food item to contain?'].apply(clean_Q2)

    # Convert cleaned to numeric
    df['Q2_cleaned'] = pd.to_numeric(df['Q2_cleaned'], errors='coerce')

    # Check for missing values
    print("Missing values in Q2 column:", df['Q2_cleaned'].isna().sum())