import pandas as pd
import numpy as np
import re
from data_handling.movie_genre_dict import movie_genres


def clean_Q2(q2):
    """
    Clean column Q2 in csv file, converting to all numerical discrete values.
    Handles various input formats and extracts numerical information.
    """
    if pd.isna(q2):
        return pd.NA
    q2 = str(q2).strip()
    if any(phrase in q2.lower() for phrase in ['#name?', 'don\'t know', 'dont know', 'no idea']):
        return pd.NA

    q2_lower = q2.lower()

    spelled_out_numbers = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12
    }

    def extract_number(text):
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if numbers:
            return int(float(numbers[0]))
        for word, num in spelled_out_numbers.items():
            if word in text.lower():
                return num
        return None

    range_match = re.search(r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)', q2)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return int((low + high) / 2)

    approx_indicators = ['around', 'about', 'approximately', '~', 'roughly']
    for indicator in approx_indicators:
        if indicator in q2_lower:
            num = extract_number(q2)
            if num is not None:
                return num

    direct_num = extract_number(q2)
    if direct_num is not None:
        return direct_num

    # Attempt to interpret as a list of ingredients
    delimiters = [',', '\n', ' and ', ';']
    for delimiter in delimiters:
        ingredients = [ing.strip() for ing in q2.split(delimiter) if ing.strip()]
        if len(ingredients) > 1:
            return len(set(ingredients))

    print(f"Could not parse Q2 value: {q2}")
    return pd.NA


def clean_price(price_str):
    if not price_str:
        return -1
    price_str = str(price_str).strip()
    if price_str.isdigit():
        return int(price_str)
    numbers = re.findall(r'\d+\.?\d*', price_str)
    if not numbers:
        return -1
    if len(numbers) >= 2:
        num1 = float(numbers[0])
        num2 = float(numbers[1])
        median = (num1 + num2) / 2
        return round(median)
    return round(float(numbers[0]))


def clean_movie_text(text):
    """
    Takes a raw free-text movie string, cleans and maps it to a standard category.
    Returns a string representing a known movie or 'unknown'.
    """
    movies = [
        '1001nights', '11sep', '13hours', '2012', '21jumpstreet', '30minutesorless', '3idiots', '47ronin', '7samurai',
        '9',
        'actionmovie', 'airbud', 'aladdin', 'alien', 'alitathewarrior', 'americanpie', 'anchorman', 'angrybirds',
        'anime',
        'anjaanaanjaani', 'aquaman', 'aquietplace', 'arcane', 'argo', 'asilentvoice', 'avengers', 'avengers:endgame',
        'babylon',
        'backinaction', 'backtothefuture', 'badboys', 'bahen', 'barbie', 'batman', 'bighero6', 'billionstvshow',
        'blackhawkdown',
        'bladerunner', 'bollywood', 'borat', 'breakaway', 'breakingbad', 'bullettrain', 'burnt', 'captainamerica',
        'carryon',
        'cars', 'casablanca', 'chandnichowktochina', 'chef', 'chinesezodiac', 'cityhunter', 'cleopatra',
        'cloudywithachanceofmeatballs', 'coco', 'comedy', 'coraline', 'crayonshinchan', 'crazyrichasians',
        'crazystupidlove',
        'dabba', 'dangal', 'deadpool', 'deadsilence', 'despicableme', 'detectiveconan', 'diaryofawimpykid', 'dictator',
        'diehard',
        'djangounchained', 'doraemon', 'dotherightthing', 'dragon', 'drange', 'drishyam', 'drive', 'dune',
        'eastsidesushi',
        'eatpraylove', 'emojimovie', 'eternalsunshineofthespotlessmind', 'evangelion', 'everythingeverywhereallatonce',
        'fallenangels', 'fast&furious', 'fastandfurious', 'ferrisbuellersdayoff', 'fightclub', 'findingnemo',
        'fivenightsatfreddys',
        'foodwars', 'forrestgump', 'freeguy', 'friday', 'friends', 'frozen', 'futurama', 'garfield', 'gijoe',
        'girlstrip', 'gladiator',
        'godfather', 'godzilla', 'gonegirl', 'goodfellas', 'goodwillhunting', 'gossipgirl', 'granturismo', 'greenbook',
        'grownups',
        'haikyu', 'hangover', 'happygilmore', 'haroldandkumar', 'harrypoter', 'harrypotter', 'hawkeye', 'heretic',
        'highschoolmusical',
        'hitman', 'homealone', 'horror', 'housemd', 'howlsmovingcastle', 'howtoloseaguyin10days', 'hunger', 'idk',
        'idontknow',
        'inception', 'indianajones', 'insideout', 'interstellar', 'ipman', 'ironman', 'isleofdogs', 'italianjon',
        'jamesbond', 'jaws',
        'jirodreamsofsush', 'jirodreamsofsushi', 'johnnyenglish', 'johnwick', 'jurassicpark', 'karatekid',
        'khabikhushikhabigham',
        'kikisdeliveryservice', 'killbill', 'kingdomofheaven', 'kingkong', 'koenokatachi', 'kungfupanda', 'lalaland',
        'lastsamurai',
        'lawrenceofarabia', 'legendofshawama', 'lifeisbeautiful', 'lifeofpi', 'lionking', 'liquoricepizza',
        'lizandthebluebird',
        'lordoftherings', 'lostintranslation', 'lovehard', 'luca', 'lucy', 'madagascar', 'madeinabyssdawnofthedeepsoul',
        'madmax',
        'mammamia', 'mandoob', 'maninblack', 'mariomovie', 'masterchef', 'mazerunner', 'meangirls', 'meitanteikonan',
        'memoirsofageisha',
        'memoryofageisha', 'meninblack', 'middleeasternmovie', 'middleeasternmovies', 'middleeastmovie',
        'midnightdiner', 'midnightdinner',
        'minions', 'mission:impossible', 'moneyball', 'monster', 'monsterhouse', 'monsterinc', 'monstersinc',
        'montypython', 'mrdeeds',
        'mulan', 'murdermystery', 'murderontheorientexpress', 'mycousinvinny', 'myheroacademia', 'myneighbourtotoro',
        'mysticpizza',
        'na', 'naruto', 'neverendingstory', 'no', 'nosferatu', 'nothing', 'notsure', 'nottinghill', 'oldboy',
        'onceuponatimeinhollywood',
        'onepiece', 'oppenheimer', 'pacificrim', 'parasite', 'passengers2016', 'pearlharbour', 'piratesofthecaribbean',
        'piratesofthecarribeans',
        'pizza', 'pizza2012', 'pokemon', 'pokemonthefirstmovie', 'ponyo', 'princeofegypt', 'princessdiaries',
        'probablysomemiddleeatmovies',
        'probablysomenichemovieorsomemoviespecifictoacountryandnotanenglishmovie',
        'probablysomethingwithamiddleeasternsetting',
        'pulpfiction', 'pursuitofhappyness', 'ratatouille', 'ratatoullie', 'rattatouieeventhoughthereisntpizza',
        'readyplayeronethereispacmaninsideandpizzalookslikeone', 'relaxingcomedy', 'rickandmorty', 'romancemovie',
        'romanticmovies',
        'runningman', 'rurounikenshin', 'rushhour', 'samuraijack', 'savingprivateryan', 'scarymovie', 'scarymovie42006',
        'scifi', 'scoobydoo',
        'scottpilgrim', 'setitup', 'sevensamurai', 'shangchi', 'shanghainoon', 'sharktale', 'shawarmalegend',
        'shawshankredemption', 'shazam',
        'shogun', 'shortmoviebecauseieatitfast', 'shrek', 'slumdogmillionaire', 'snowpiercer', 'someghiblimovie',
        'sonicthehedgehog',
        'sonofbabylon', 'soul', 'southpark', 'spacejam', 'spiderman', 'spiritedaway', 'spongebob', 'spykids',
        'squidgame', 'starwars',
        'starwars:thelastjedi', 'stepbrothers', 'strangerthings', 'suits', 'superbad', 'suzume', 'talented',
        'taxidriver', 'teenagemutantninjaturtles',
        'terminator', 'thebiglebowski', 'thebigshort', 'thebigsick', 'theboyandtheheron', 'theboys', 'thebreakfastclub',
        'thedavincicode',
        'thedictator', 'thedoramovie', 'thegarfieldmovie', 'thegentlemen', 'thegodfather', 'thegodfatherpartii',
        'thegoofymovie', 'thegrinch',
        'thehangover', 'thehungergames', 'theintern', 'theinvisibleguest', 'theitalianjob', 'thekiterunner',
        'thelastsamurai', 'thelegomovie',
        'thelegomovie2', 'thelionking', 'thelittlemermaid', 'themariomovie', 'themeg', 'themenu', 'themummy',
        'thepacific', 'theperfectstorm',
        'theprinceofegypt', 'theproposal', 'theritual', 'theroadtofallujah', 'theroom', 'thesamurai',
        'thesocialnetwork', 'thespynextdoor',
        'thetrumanshow', 'thewhale', 'thisistheend', 'thosecommercialsbythetorontofoodmanzabebsiisthebest',
        'threeidiots', 'timeofhappiness',
        'titanic', 'tokyostory', 'totoro', 'toystory', 'toystory4', 'traintobusan', 'transformer', 'turningred',
        'ultramanrising', 'unclegrandpa',
        'unknown', 'us', 'venom', 'wags', 'walle', 'waynesworld', 'weatheringwithyou', 'whiplash', 'whoami2005',
        'whoamijackiechan', 'wicked',
        'wizardsofwaverlyplacemovie', 'wolfofwallstreet', 'wolverine', 'yakuza', 'yehjawaanihaideewani',
        'youdontmesswiththezohan', 'yourname', 'zootopia'
    ]

    if pd.isna(text):
        return 'unknown'
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r' ', '', text)

    return next((m for m in movies if m in text), "unknown")


def clean_drink_text(text):
    if pd.isna(text):
        return 'unknown'
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'coka\s+cola', 'coca cola', text)
    text = re.sub(r'cocacola', 'coca cola', text)

    brand_to_category = {
        'coke': 'soda', 'coca cola': 'soda', 'cola': 'soda', 'pepsi': 'soda', 'sprite': 'soda',
        'fanta': 'soda', 'mountain dew': 'soda', 'dr pepper': 'soda', 'ramune': 'asian pop', 'yakult': 'asian pop',
        'red bull': 'energy drink', 'monster': 'energy drink', 'beer': 'alcohol', 'wine': 'alcohol',
        'saporo': 'alcohol',
        'sake': 'asian alcohol', 'soju': 'asian alcohol', 'coffee': 'coffee', 'espresso': 'coffee', 'latte': 'coffee',
        'tea': 'tea', 'soup': 'soup', 'juice': 'juice', 'water': 'water', 'milk': 'milk',
        'smoothie': 'smoothie', 'milkshake': 'milkshake'
    }
    categories_found = set()
    for brand, cat in brand_to_category.items():
        if brand in text:
            categories_found.add(cat)

    if not categories_found:
        return 'other'
    return ', '.join(sorted(categories_found))


def one_hot_encode(response_str, categories):
    result = np.zeros(len(categories), dtype=float)  # keep as float
    if isinstance(response_str, str):
        selections = response_str.split(',')
        for i, category in enumerate(categories):
            if category in selections:
                result[i] = 1.0
    return result


def clean_data(df):
    """
    Apply cleaning to each Q-column, then:
      - Drop or fill missing
      - One-hot Q3, Q6, Q7
      - Convert Q8 to ordinal
      - Create dummies for Q5
      - Standardize Q2 and Q4 (keep them as floats, no int-cast)
    """
    print("Raw columns:", df.columns.tolist())
    for col in df.columns:
        print(repr(col))

    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('"', '')
    df.rename(
        columns={
            'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': 'Q1',
            'Q2: How many ingredients would you expect this food item to contain?': 'Q2',
            'Q3: In what setting would you expect this food to be served? Please check all that apply': 'Q3',
            'Q4: How much would you expect to pay for one serving of this food item?': 'Q4',
            'Q5: What movie do you think of when thinking of this food item?': 'Q5',
            'Q6: What drink would you pair with this food item?': 'Q6',
            'Q7: When you think about this food item, who does it remind you of?': 'Q7',
            'Q8: How much hot sauce would you add to this food item?': 'Q8',
        },
        inplace=True
    )


    # Clean Q2, Q4, Q5, Q6
    df['Q2'] = df['Q2'].apply(clean_Q2)
    df['Q2'] = pd.to_numeric(df['Q2'], errors='coerce')
    df['Q4'] = df['Q4'].apply(clean_price)
    df['Q5'] = df['Q5'].apply(clean_movie_text)
    df['Q6'] = df['Q6'].apply(clean_drink_text)

    # Drop rows with missing Q2/Q4
    df = df.dropna(subset=['Q2', 'Q4'])

    # Q1 is on a simple 1..5 scale; keep as int if you like:
    df['Q1'] = pd.to_numeric(df['Q1'], errors='coerce')
    # Fill missing Q1 with -1 if needed
    df['Q1'] = df['Q1'].fillna(-1).astype(int)

    # Q3, Q7, Q8 can be left as strings except we will do one-hot or ordinal
    df['Q3'] = df['Q3'].fillna("Unknown")
    df['Q7'] = df['Q7'].fillna("Unknown")
    df['Q8'] = df['Q8'].fillna("Unknown")

    # Make dummies for Q5 (the "movie" column)
    Q5_CATEGORIES = list(movie_genres.keys())
    df_genres = df['Q5'].str.get_dummies(sep=',')
    df_genres = df_genres.reindex(columns=Q5_CATEGORIES, fill_value=0)
    df = pd.concat([df, df_genres], axis=1)
    df.drop(columns=['Q5'], inplace=True)

    # One-hot Q3, Q6, Q7
    Q3_categories = ["Week day lunch", "Week day dinner", "Weekend lunch", "Weekend dinner", "At a party",
                     "Late night snack"]
    Q6_categories = ["soda", "other", "tea", "alcohol", "water", "soup", "juice", "milk", "unknown", "smoothie",
                     "asian alcohol", "asian pop", "milkshake"]
    Q7_categories = ["Parents", "Siblings", "Friends", "Teachers", "Strangers"]

    def one_hot(df, col, cats):
        arr = np.array([one_hot_encode(val, cats) for val in df[col]])
        for i, c in enumerate(cats):
            df[f"{c}"] = arr[:, i]
        df.drop(columns=[col], inplace=True)

    one_hot(df, "Q3", Q3_categories)
    one_hot(df, "Q6", Q6_categories)
    one_hot(df, "Q7", Q7_categories)

    # Map Q8
    Q8_CATEGORIES = [
        "None",
        "A little (mild)",
        "A moderate amount (medium)",
        "A lot (hot)",
        "I will have some of this food item with my hot sauce"
    ]

    df_q8 = pd.get_dummies(df['Q8'], prefix='Q8')
    # Force all expected Q8 columns to appear:
    all_q8_columns = [f"Q8_{cat}" for cat in Q8_CATEGORIES]
    df_q8 = df_q8.reindex(columns=all_q8_columns, fill_value=0)

    df = pd.concat([df, df_q8], axis=1)
    df.drop(columns=["Q8"], inplace=True)

    # Remove outliers
    df = df[df['Q4'] < 30]
    df = df[df['Q2'] < 15]

    # Standardize Q2, Q4 as floats
    df['Q4'] = (df['Q4'] - df['Q4'].mean()) / df['Q4'].std()
    df['Q2'] = (df['Q2'] - df['Q2'].mean()) / df['Q2'].std()

    return df


def clean(path, output_path="cleaned_data.csv"):
    df = pd.read_csv(path)
    df = clean_data(df)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")
    return df


if __name__ == "__main__":
    # Example usage
    df = clean("../data/cleaned_data_combined_modified.csv", "../data/cleaned_data.csv")
    print(df.head())
