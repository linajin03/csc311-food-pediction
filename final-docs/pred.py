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

import csv
import numpy as np
import pandas as pd
import os
import sys
import zipfile

# Extract code.zip to a temporary directory
with zipfile.ZipFile('code.zip', 'r') as zip_ref:
    zip_ref.extractall('temp_code')

# Add the extracted directory to Python path
sys.path.insert(0, os.path.abspath('temp_code'))

weights_path = os.path.join('temp_code', 'food_nn_weights.npz')
weights = np.load(weights_path) # Load the weights

from food_NN import FoodNeuralNetwork 
from final_cleaning import clean_data 

# Label mapping (should match your training)
LABEL_MAPPING = {
    0: 'Pizza',
    1: 'Shawarma', 
    2: 'Sushi'
}

# Load  pre-trained model
def load_model(weights):
    """Load the pre-trained model with its weights"""
    # Initialize model with same architecture as training
    model = FoodNeuralNetwork(num_features=332, num_hidden=256, num_classes=3)
    
    # Load the weights
    model.W1 = weights['W1']
    model.b1 = weights['b1']
    model.W2 = weights['W2']
    model.b2 = weights['b2']
    
    return model

feature_columns = ['Q1', 'Q2', 'Q4', '1001nights', '11sep', '13hours', '2012', '21jumpstreet', '30minutesorless', '3idiots', '47ronin', '7samurai', '9', 'actionmovie', 'airbud', 'aladdin', 'alien', 'alitathewarrior', 'americanpie', 'anchorman', 'angrybirds', 'anime', 'anjaanaanjaani', 'aquaman', 'aquietplace', 'arcane', 'argo', 'asilentvoice', 'avengers', 'babylon', 'backinaction', 'backtothefuture', 'badboys', 'bahen', 'barbie', 'batman', 'bighero6', 'billionstvshow', 'blackhawkdown', 'bladerunner', 'bollywood', 'borat', 'breakaway', 'breakingbad', 'bullettrain', 'burnt', 'captainamerica', 'carryon', 'cars', 'casablanca', 'chandnichowktochina', 'chef', 'chinesezodiac', 'cityhunter', 'cleopatra', 'cloudywithachanceofmeatballs', 'coco', 'comedy', 'coraline', 'crayonshinchan', 'crazyrichasians', 'crazystupidlove', 'dabba', 'dangal', 'deadpool', 'deadsilence', 'despicableme', 'diaryofawimpykid', 'dictator', 'diehard', 'djangounchained', 'doraemon', 'dotherightthing', 'dragon', 'drange', 'drishyam', 'drive', 'dune', 'eastsidesushi', 'eatpraylove', 'emojimovie', 'eternalsunshineofthespotlessmind', 'evangelion', 'everythingeverywhereallatonce', 'fallenangels', 'fastandfurious', 'ferrisbuellersdayoff', 'fightclub', 'findingnemo', 'fivenightsatfreddys', 'foodwars', 'freeguy', 'friday', 'friends', 'frozen', 'futurama', 'garfield', 'gijoe', 'girlstrip', 'gladiator', 'godfather', 'godzilla', 'gonegirl', 'goodfellas', 'goodwillhunting', 'gossipgirl', 'granturismo', 'greenbook', 'grownups', 'haikyu', 'hangover', 'happygilmore', 'haroldandkumar', 'harrypoter', 'harrypotter', 'hawkeye', 'heretic', 'highschoolmusical', 'hitman', 'homealone', 'horror', 'housemd', 'howlsmovingcastle', 'howtoloseaguyin10days', 'hunger', 'idk', 'idontknow', 'inception', 'indianajones', 'insideout', 'interstellar', 'ipman', 'ironman', 'isleofdogs', 'italianjon', 'jamesbond', 'jaws', 'jirodreamsofsush', 'johnnyenglish', 'johnwick', 'jurassicpark', 'karatekid', 'khabikhushikhabigham', 'kikisdeliveryservice', 'killbill', 'kingdomofheaven', 'kingkong', 'koenokatachi', 'kungfupanda', 'lalaland', 'lastsamurai', 'lawrenceofarabia', 'lifeisbeautiful', 'lifeofpi', 'lionking', 'liquoricepizza', 'lordoftherings', 'lostintranslation', 'lovehard', 'luca', 'lucy', 'madagascar', 'madeinabyssdawnofthedeepsoul', 'madmax', 'mammamia', 'mandoob', 'maninblack', 'mariomovie', 'mazerunner', 'meangirls', 'meitanteikonan', 'memoirsofageisha', 'memoryofageisha', 'meninblack', 'middleeasternmovie', 'middleeastmovie', 'midnightdiner', 'midnightdinner', 'minions', 'moneyball', 'monster', 'montypython', 'mrdeeds', 'mulan', 'murderontheorientexpress', 'mycousinvinny', 'myheroacademia', 'myneighbourtotoro', 'mysticpizza', 'na', 'neverendingstory', 'no', 'oldboy', 'onepiece', 'oppenheimer', 'pacificrim', 'parasite', 'passengers2016', 'pearlharbour', 'piratesofthecaribbean', 'piratesofthecarribeans', 'pizza', 'pokemon', 'ponyo', 'princeofegypt', 'princessdiaries', 'probablysomemiddleeatmovies', 'probablysomethingwithamiddleeasternsetting', 'pulpfiction', 'pursuitofhappyness', 'ratatouille', 'ratatoullie', 'rickandmorty', 'romancemovie', 'romanticmovies', 'runningman', 'rurounikenshin', 'rushhour', 'samuraijack', 'savingprivateryan', 'scarymovie', 'scifi', 'scoobydoo', 'scottpilgrim', 'setitup', 'sevensamurai', 'shangchi', 'sharktale', 'shawarmalegend', 'shawshankredemption', 'shazam', 'shogun', 'shortmoviebecauseieatitfast', 'shrek', 'someghiblimovie', 'sonicthehedgehog', 'soul', 'southpark', 'spacejam', 'spiderman', 'spiritedaway', 'spongebob', 'spykids', 'squidgame', 'starwars', 'stepbrothers', 'strangerthings', 'suits', 'superbad', 'suzume', 'talented', 'thebiglebowski', 'thebigshort', 'thebigsick', 'theboyandtheheron', 'theboys', 'thebreakfastclub', 'thedavincicode', 'thedoramovie', 'thegentlemen', 'thegoofymovie', 'thegrinch', 'theintern', 'theinvisibleguest', 'theitalianjob', 'thekiterunner', 'thelegomovie', 'thelittlemermaid', 'themeg', 'themenu', 'themummy', 'thepacific', 'theperfectstorm', 'theproposal', 'theritual', 'theroom', 'thesamurai', 'thesocialnetwork', 'thespynextdoor', 'thetrumanshow', 'thewhale', 'thisistheend', 'thosecommercialsbythetorontofoodmanzabebsiisthebest', 'threeidiots', 'timeofhappiness', 'titanic', 'tokyostory', 'totoro', 'toystory', 'traintobusan', 'transformer', 'turningred', 'ultramanrising', 'unclegrandpa', 'unknown', 'us', 'wags', 'walle', 'waynesworld', 'weatheringwithyou', 'whiplash', 'whoamijackiechan', 'wicked', 'wolfofwallstreet', 'wolverine', 'yakuza', 'yehjawaanihaideewani', 'youdontmesswiththezohan', 'zootopia', 'Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack', 'soda', 'other', 'tea', 'alcohol', 'water', 'soup', 'juice', 'milk', 'smoothie', 'asian alcohol', 'asian pop', 'milkshake', 'Parents', 'Siblings', 'Friends', 'Teachers', 'Strangers']

def preprocess_input(data):
    """
    Preprocess input data to match training format
    Args:
        data: Can be a dict (single sample) or DataFrame (multiple samples)
    Returns:
        Processed DataFrame ready for prediction
    """
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    data_cleaned = clean_data(data)
    
    for col in feature_columns:
        if col not in data_cleaned.columns:
            data_cleaned[col] = 0  # Add missing columns with default value 0

    # Reorder columns to match the training data
    data_cleaned = data_cleaned[feature_columns]
    
    return data_cleaned

def predict(x):
    """
    Helper function to make prediction for a given input x
    Args:
        x: Input data (dict or DataFrame)
    Returns:
        Predicted label (str)
    """
    model = load_model(weights)
    processed = preprocess_input(x)
    pred = model.predict(processed)[0]
    return LABEL_MAPPING.get(pred, "Unknown")  # Default to "Unknown" if prediction not in mapping

def predict_all(filename):
    """
    Main prediction function that takes a CSV file and returns predictions
    Args:
        filename: Path to CSV file containing test data
    Returns:
        list: Predictions for each row in the CSV file
    """
    # Read the CSV file
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        test_data = [row for row in reader]
    
    predictions = []
    for test_example in test_data:
        pred = predict(test_example)
        predictions.append(pred)
    
    return predictions

# For local testing only
if __name__ == "__main__":
    # Example usage
    test_predictions = predict_all("example_test_set.csv")
    print(test_predictions)