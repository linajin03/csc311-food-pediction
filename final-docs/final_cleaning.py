import pandas as pd
import numpy as np
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
# Function to clean Q4
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

# Function to clean Q5
def clean_movie_text(text):
    """
    Takes a raw free-text movie string, cleans and maps it to a standard category.
    Returns a string representing a known movie.
    """
    movies = ['1001nights', '11sep', '13hours', '2012', '21jumpstreet', '30minutesorless', '3idiots', '47ronin', '7samurai', '9', 'actionmovie', 'airbud', 'aladdin', 'alien', 'alitathewarrior', 'americanpie', 'anchorman', 'angrybirds', 'anime', 'anjaanaanjaani', 'aquaman', 'aquietplace', 'arcane', 'argo', 'asilentvoice', 'avengers', 'avengers:endgame', 'babylon', 'backinaction', 'backtothefuture', 'badboys', 'bahen', 'barbie', 'batman', 'bighero6', 'billionstvshow', 'blackhawkdown', 'bladerunner', 'bollywood', 'borat', 'breakaway', 'breakingbad', 'bullettrain', 'burnt', 'captainamerica', 'carryon', 'cars', 'casablanca', 'chandnichowktochina', 'chef', 'chinesezodiac', 'cityhunter', 'cleopatra', 'cloudywithachanceofmeatballs', 'coco', 'comedy', 'coraline', 'crayonshinchan', 'crazyrichasians', 'crazystupidlove', 'dabba', 'dangal', 'deadpool', 'deadsilence', 'despicableme', 'detectiveconan', 'diaryofawimpykid', 'dictator', 'diehard', 'djangounchained', 'doraemon', 'dotherightthing', 'dragon', 'drange', 'drishyam', 'drive', 'dune', 'eastsidesushi', 'eatpraylove', 'emojimovie', 'eternalsunshineofthespotlessmind', 'evangelion', 'everythingeverywhereallatonce', 'fallenangels', 'fast&furious', 'fastandfurious', 'ferrisbuellersdayoff', 'fightclub', 'findingnemo', 'fivenightsatfreddys', 'foodwars', 'forrestgump', 'freeguy', 'friday', 'friends', 'frozen', 'futurama', 'garfield', 'gijoe', 'girlstrip', 'gladiator', 'godfather', 'godzilla', 'gonegirl', 'goodfellas', 'goodwillhunting', 'gossipgirl', 'granturismo', 'greenbook', 'grownups', 'haikyu', 'hangover', 'happygilmore', 'haroldandkumar', 'harrypoter', 'harrypotter', 'hawkeye', 'heretic', 'highschoolmusical', 'hitman', 'homealone', 'horror', 'housemd', 'howlsmovingcastle', 'howtoloseaguyin10days', 'hunger', 'idk', 'idontknow', 'inception', 'indianajones', 'insideout', 'interstellar', 'ipman', 'ironman', 'isleofdogs', 'italianjon', 'jamesbond', 'jaws', 'jirodreamsofsush', 'jirodreamsofsushi', 'johnnyenglish', 'johnwick', 'jurassicpark', 'karatekid', 'khabikhushikhabigham', 'kikisdeliveryservice', 'killbill', 'kingdomofheaven', 'kingkong', 'koenokatachi', 'kungfupanda', 'lalaland', 'lastsamurai', 'lawrenceofarabia', 'legendofshawama', 'lifeisbeautiful', 'lifeofpi', 'lionking', 'liquoricepizza', 'lizandthebluebird', 'lordoftherings', 'lostintranslation', 'lovehard', 'luca', 'lucy', 'madagascar', 'madeinabyssdawnofthedeepsoul', 'madmax', 'mammamia', 'mandoob', 'maninblack', 'mariomovie', 'masterchef', 'mazerunner', 'meangirls', 'meitanteikonan', 'memoirsofageisha', 'memoryofageisha', 'meninblack', 'middleeasternmovie', 'middleeasternmovies', 'middleeastmovie', 'midnightdiner', 'midnightdinner', 'minions', 'mission:impossible', 'moneyball', 'monster', 'monsterhouse', 'monsterinc', 'monstersinc', 'montypython', 'mrdeeds', 'mulan', 'murdermystery', 'murderontheorientexpress', 'mycousinvinny', 'myheroacademia', 'myneighbourtotoro', 'mysticpizza', 'na', 'naruto', 'neverendingstory', 'no', 'nosferatu', 'nothing', 'notsure', 'nottinghill', 'oldboy', 'onceuponatimeinhollywood', 'onepiece', 'oppenheimer', 'pacificrim', 'parasite', 'passengers2016', 'pearlharbour', 'piratesofthecaribbean', 'piratesofthecarribeans', 'pizza', 'pizza2012', 'pokemon', 'pokemonthefirstmovie', 'ponyo', 'princeofegypt', 'princessdiaries', 'probablysomemiddleeatmovies', 'probablysomenichemovieorsomemoviespecifictoacountryandnotanenglishmovie', 'probablysomethingwithamiddleeasternsetting', 'pulpfiction', 'pursuitofhappyness', 'ratatouille', 'ratatoullie', 'rattatouieeventhoughthereisntpizza', 'readyplayeronethereispacmaninsideandpizzalookslikeone', 'relaxingcomedy', 'rickandmorty', 'romancemovie', 'romanticmovies', 'runningman', 'rurounikenshin', 'rushhour', 'samuraijack', 'savingprivateryan', 'scarymovie', 'scarymovie42006', 'scifi', 'scoobydoo', 'scottpilgrim', 'setitup', 'sevensamurai', 'shangchi', 'shanghainoon', 'sharktale', 'shawarmalegend', 'shawshankredemption', 'shazam', 'shogun', 'shortmoviebecauseieatitfast', 'shrek', 'slumdogmillionaire', 'snowpiercer', 'someghiblimovie', 'sonicthehedgehog', 'sonofbabylon', 'soul', 'southpark', 'spacejam', 'spiderman', 'spiritedaway', 'spongebob', 'spykids', 'squidgame', 'starwars', 'starwars:thelastjedi', 'stepbrothers', 'strangerthings', 'suits', 'superbad', 'suzume', 'talented', 'taxidriver', 'teenagemutantninjaturtles', 'terminator', 'thebiglebowski', 'thebigshort', 'thebigsick', 'theboyandtheheron', 'theboys', 'thebreakfastclub', 'thedavincicode', 'thedictator', 'thedoramovie', 'thegarfieldmovie', 'thegentlemen', 'thegodfather', 'thegodfatherpartii', 'thegoofymovie', 'thegrinch', 'thehangover', 'thehungergames', 'theintern', 'theinvisibleguest', 'theitalianjob', 'thekiterunner', 'thelastsamurai', 'thelegomovie', 'thelegomovie2', 'thelionking', 'thelittlemermaid', 'themariomovie', 'themeg', 'themenu', 'themummy', 'thepacific', 'theperfectstorm', 'theprinceofegypt', 'theproposal', 'theritual', 'theroadtofallujah', 'theroom', 'thesamurai', 'thesocialnetwork', 'thespynextdoor', 'thetrumanshow', 'thewhale', 'thisistheend', 'thosecommercialsbythetorontofoodmanzabebsiisthebest', 'threeidiots', 'timeofhappiness', 'titanic', 'tokyostory', 'totoro', 'toystory', 'toystory4', 'traintobusan', 'transformer', 'turningred', 'ultramanrising', 'unclegrandpa', 'unknown', 'us', 'venom', 'wags', 'walle', 'waynesworld', 'weatheringwithyou', 'whiplash', 'whoami2005', 'whoamijackiechan', 'wicked', 'wizardsofwaverlyplacemovie', 'wolfofwallstreet', 'wolverine', 'yakuza', 'yehjawaanihaideewani', 'youdontmesswiththezohan', 'yourname', 'zootopia']


    # Handle missing or NaN
    if pd.isna(text):
        return 'unknown'

    # Convert to lowercase + remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r' ', '', text)
    
    #check that text is a recognized move, otherwise label with unknown
    text = next((movie for movie in movies if movie in text), "unknown")

    return text

def one_hot_encode(response_str, categories):
    result = np.zeros(len(categories))
    if isinstance(response_str, str):  # Check if it's a string (not NaN)
        selections = response_str.split(',')
        for i, category in enumerate(categories):
            if category in selections:
                result[i] = 1
    return result

# Function to clean Q6
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

def clean_data(df):
    """
    Clean and format the data in the dataframe
    """
    df.rename(columns={'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': 'Q1', 
                       'Q2: How many ingredients would you expect this food item to contain?': 'Q2', 
                       'Q3: In what setting would you expect this food to be served? Please check all that apply': 'Q3',
                       'Q4: How much would you expect to pay for one serving of this food item?': 'Q4',
                       'Q5: What movie do you think of when thinking of this food item?': 'Q5',
                       'Q6: What drink would you pair with this food item?': 'Q6',
                       'Q7: When you think about this food item, who does it remind you of?': 'Q7',
                       'Q8: How much hot sauce would you add to this food item?': 'Q8',
                       }, inplace=True)

    #cleaning Q2
    df['Q2'] = df['Q2'].apply(clean_Q2)
    df['Q2'] = pd.to_numeric(df['Q2'], errors='coerce')

    #Q3 requires no cleaning

    #cleaning Q4 
    df["Q4"] = df["Q4"].apply(clean_price)

    #cleaning Q5
    df["Q5"] = df["Q5"].apply(clean_movie_text)
    #cleaning Q6
    df["Q6"] = df["Q6"].apply(clean_drink_text)

    #Q7 requires no cleaning
    #Q8 requires no cleaning

    # Drop rows with missing values
    df = df.dropna()

    # Categorical Feature categories
    Q3_categories = ["Week day lunch", "Week day dinner", "Weekend lunch", "Weekend dinner", "At a party", "Late night snack"]
    
    Q6_categories = [
        "soda", "other", "tea", "alcohol", "water", "soup", "juice", "milk", 
        "unknown", "smoothie", "asian alcohol", "asian pop", "milkshake"
    ]
    Q7_categories = ["Parents", "Siblings", "Friends", "Teachers", "Strangers"]
    Q8_categories = ["None", "A little (mild)", "A moderate amount (medium)", "A lot (hot)", "I will have some of this food item with my hot sauce"]
    
    df_genres = df['Q5'].str.get_dummies(sep=',')

    # Concatenate the new genre columns with the original DataFrame
    df = pd.concat([df, df_genres], axis=1)
    df = df.drop(columns=['Q5'])

    # One hot encode Q3, Q5, Q6, and Q7
    q3 = np.array([one_hot_encode(response, Q3_categories) for response in df['Q3']])
    q6 = np.array([one_hot_encode(response, Q6_categories) for response in df['Q6']])
    q7 = np.array([one_hot_encode(response, Q7_categories) for response in df['Q7']])

    # Create new columns for Q3, Q6, and Q7
    for i, category in enumerate(Q3_categories):
        df[category] = q3[:, i]
    for i, category in enumerate(Q6_categories):
        df[category] = q6[:, i]
    for i, category in enumerate(Q7_categories):
        df[category] = q7[:, i]

    # Convert Q8 to ordinal scale (0, 1, 2, 3, 4)
    ordinal_mapping = {category: idx for idx, category in enumerate(Q8_categories)}
    df['Q8'] = df['Q8'].map(ordinal_mapping)

    # Remove dollar signs from Q4
    df["Q4"] = df["Q4"].apply(lambda x: str(x).replace("$", "") if isinstance(x, str) else x)

    # Drop original categorical columns
    df = df.drop(columns=[ 'Q3', 'Q6', 'Q7', 'Q8'])

    # Make sure its an integer
    df['Q1'] = df['Q1'].astype(int, errors='ignore')
    df['Q2'] = df['Q2'].astype(int, errors='ignore')
    df['Q4'] = df['Q4'].astype(int, errors='ignore')

    # Remove outliers from Q4 that are larger than 60
    df = df[df['Q4'] < 30]

    # Normalize Q4
    df['Q4'] = (df['Q4'] - df['Q4'].mean()) / df['Q4'].std()

    # Remove outliers from Q2 that are larger than 15
    df = df[df['Q2'] < 15]

    # Normalize Q2
    df['Q2'] = (df['Q2'] - df['Q2'].mean()) / df['Q2'].std()

    return df


def train_sgd(model, X_train, t_train,
              alpha=0.1, n_epochs=0, batch_size=100,
              X_valid=None, t_valid=None,
              w_init=None, plot=True):
    '''
    Given model - an instance of MLPModel
          X_train - the data matrix to use for training
          t_train - the target vector to use for training
          alpha - the learning rate.
                    From our experiments, it appears that a larger learning rate
                    is appropriate for this task.
          n_epochs - the number of **epochs** of gradient descent to run
          batch_size - the size of each mini batch
          X_valid - the data matrix to use for validation (optional)
          t_valid - the target vector to use for validation (optional)
          w_init - the initial w vector (if None, use a vector of all zeros)
          plot - whether to track statistics and plot the training curve

    Solves for model weights via stochastic gradient descent,
    using the provided batch_size.

    Return weights after niter iterations.
    '''
    def make_onehot(indicies, total=128):
        I = np.eye(total)
        return I[indicies]
    # as before, initialize all the weights to zeros
    w = np.zeros(X_train.shape[1])

    train_loss = [] # for the current minibatch, tracked once per iteration
    valid_loss = [] # for the entire validation data set, tracked once per epoch

    # track the number of iterations
    niter = 0

    # we will use these indices to help shuffle X_train
    N = X_train.shape[0] # number of training data points
    indices = list(range(N))
    try:
        for e in range(n_epochs):
            random.shuffle(indices) # for creating new minibatches

            for i in range(0, N, batch_size):
                if (i + batch_size) > N:
                    # At the very end of an epoch, if there are not enough
                    # data points to form an entire batch, then skip this batch
                    continue

                indices_in_batch = indices[i: i+batch_size]
                X_minibatch = X_train[indices_in_batch, :]
                t_minibatch = make_onehot(t_train[indices_in_batch], model.num_classes)

                # gradient descent iteration
                model.cleanup()
                model.forward(X_minibatch)
                model.backward(t_minibatch)
                model.update(alpha)

                if plot:
                    # Record the current training loss values
                    train_loss.append(model.loss(t_minibatch))
                niter += 1

            # compute validation data metrics, if provided, once per epoch
            if plot and (X_valid is not None) and (t_valid is not None):
                model.cleanup()
                model.forward(X_valid)
                valid_loss.append((niter, model.loss(make_onehot(t_valid, model.num_classes))))
     
    except Exception as e:
        print(f"Training interrupted: {e}")
    if plot:
        plt.title("SGD Training Curve Showing Loss at each Iteration")
        plt.plot(train_loss, label="Training Loss")
        if (X_valid is not None) and (t_valid is not None): # compute validation data metrics, if provided
            plt.plot([iter for (iter, loss) in valid_loss],
                     [loss for (iter, loss) in valid_loss],
                     label="Validation Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        print("Final Training Loss:", train_loss[-1])
        if (X_valid is not None) and (t_valid is not None):
            print("Final Validation Loss:", valid_loss[-1])

    # return weights 
    return w