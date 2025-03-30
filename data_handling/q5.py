import pandas as pd
import numpy as np
import re

# Load dataset
df = pd.read_csv('data/cleaned_data_combined_modified.csv')

movies = ['1001nights', '11sep', '13hours', '2012', '21jumpstreet', '30minutesorless', '3idiots', '47ronin', '7samurai', '9', 'actionmovie', 'airbud', 'aladdin', 'alien', 'alitathewarrior', 'americanpie', 'anchorman', 'angrybirds', 'anime', 'anjaanaanjaani', 'aquaman', 'aquietplace', 'arcane', 'argo', 'asilentvoice', 'avengers', 'avengers:endgame', 'babylon', 'backinaction', 'backtothefuture', 'badboys', 'bahen', 'barbie', 'batman', 'bighero6', 'billionstvshow', 'blackhawkdown', 'bladerunner', 'bollywood', 'borat', 'breakaway', 'breakingbad', 'bullettrain', 'burnt', 'captainamerica', 'carryon', 'cars', 'casablanca', 'chandnichowktochina', 'chef', 'chinesezodiac', 'cityhunter', 'cleopatra', 'cloudywithachanceofmeatballs', 'coco', 'comedy', 'coraline', 'crayonshinchan', 'crazyrichasians', 'crazystupidlove', 'dabba', 'dangal', 'deadpool', 'deadsilence', 'despicableme', 'detectiveconan', 'diaryofawimpykid', 'dictator', 'diehard', 'djangounchained', 'doraemon', 'dotherightthing', 'dragon', 'drange', 'drishyam', 'drive', 'dune', 'eastsidesushi', 'eatpraylove', 'emojimovie', 'eternalsunshineofthespotlessmind', 'evangelion', 'everythingeverywhereallatonce', 'fallenangels', 'fast&furious', 'fastandfurious', 'ferrisbuellersdayoff', 'fightclub', 'findingnemo', 'fivenightsatfreddys', 'foodwars', 'forrestgump', 'freeguy', 'friday', 'friends', 'frozen', 'futurama', 'garfield', 'gijoe', 'girlstrip', 'gladiator', 'godfather', 'godzilla', 'gonegirl', 'goodfellas', 'goodwillhunting', 'gossipgirl', 'granturismo', 'greenbook', 'grownups', 'haikyu', 'hangover', 'happygilmore', 'haroldandkumar', 'harrypoter', 'harrypotter', 'hawkeye', 'heretic', 'highschoolmusical', 'hitman', 'homealone', 'horror', 'housemd', 'howlsmovingcastle', 'howtoloseaguyin10days', 'hunger', 'idk', 'idontknow', 'inception', 'indianajones', 'insideout', 'interstellar', 'ipman', 'ironman', 'isleofdogs', 'italianjon', 'jamesbond', 'jaws', 'jirodreamsofsush', 'jirodreamsofsushi', 'johnnyenglish', 'johnwick', 'jurassicpark', 'karatekid', 'khabikhushikhabigham', 'kikisdeliveryservice', 'killbill', 'kingdomofheaven', 'kingkong', 'koenokatachi', 'kungfupanda', 'lalaland', 'lastsamurai', 'lawrenceofarabia', 'legendofshawama', 'lifeisbeautiful', 'lifeofpi', 'lionking', 'liquoricepizza', 'lizandthebluebird', 'lordoftherings', 'lostintranslation', 'lovehard', 'luca', 'lucy', 'madagascar', 'madeinabyssdawnofthedeepsoul', 'madmax', 'mammamia', 'mandoob', 'maninblack', 'mariomovie', 'masterchef', 'mazerunner', 'meangirls', 'meitanteikonan', 'memoirsofageisha', 'memoryofageisha', 'meninblack', 'middleeasternmovie', 'middleeasternmovies', 'middleeastmovie', 'midnightdiner', 'midnightdinner', 'minions', 'mission:impossible', 'moneyball', 'monster', 'monsterhouse', 'monsterinc', 'monstersinc', 'montypython', 'mrdeeds', 'mulan', 'murdermystery', 'murderontheorientexpress', 'mycousinvinny', 'myheroacademia', 'myneighbourtotoro', 'mysticpizza', 'na', 'naruto', 'neverendingstory', 'no', 'nosferatu', 'nothing', 'notsure', 'nottinghill', 'oldboy', 'onceuponatimeinhollywood', 'onepiece', 'oppenheimer', 'pacificrim', 'parasite', 'passengers2016', 'pearlharbour', 'piratesofthecaribbean', 'piratesofthecarribeans', 'pizza', 'pizza2012', 'pokemon', 'pokemonthefirstmovie', 'ponyo', 'princeofegypt', 'princessdiaries', 'probablysomemiddleeatmovies', 'probablysomenichemovieorsomemoviespecifictoacountryandnotanenglishmovie', 'probablysomethingwithamiddleeasternsetting', 'pulpfiction', 'pursuitofhappyness', 'ratatouille', 'ratatoullie', 'rattatouieeventhoughthereisntpizza', 'readyplayeronethereispacmaninsideandpizzalookslikeone', 'relaxingcomedy', 'rickandmorty', 'romancemovie', 'romanticmovies', 'runningman', 'rurounikenshin', 'rushhour', 'samuraijack', 'savingprivateryan', 'scarymovie', 'scarymovie42006', 'scifi', 'scoobydoo', 'scottpilgrim', 'setitup', 'sevensamurai', 'shangchi', 'shanghainoon', 'sharktale', 'shawarmalegend', 'shawshankredemption', 'shazam', 'shogun', 'shortmoviebecauseieatitfast', 'shrek', 'slumdogmillionaire', 'snowpiercer', 'someghiblimovie', 'sonicthehedgehog', 'sonofbabylon', 'soul', 'southpark', 'spacejam', 'spiderman', 'spiritedaway', 'spongebob', 'spykids', 'squidgame', 'starwars', 'starwars:thelastjedi', 'stepbrothers', 'strangerthings', 'suits', 'superbad', 'suzume', 'talented', 'taxidriver', 'teenagemutantninjaturtles', 'terminator', 'thebiglebowski', 'thebigshort', 'thebigsick', 'theboyandtheheron', 'theboys', 'thebreakfastclub', 'thedavincicode', 'thedictator', 'thedoramovie', 'thegarfieldmovie', 'thegentlemen', 'thegodfather', 'thegodfatherpartii', 'thegoofymovie', 'thegrinch', 'thehangover', 'thehungergames', 'theintern', 'theinvisibleguest', 'theitalianjob', 'thekiterunner', 'thelastsamurai', 'thelegomovie', 'thelegomovie2', 'thelionking', 'thelittlemermaid', 'themariomovie', 'themeg', 'themenu', 'themummy', 'thepacific', 'theperfectstorm', 'theprinceofegypt', 'theproposal', 'theritual', 'theroadtofallujah', 'theroom', 'thesamurai', 'thesocialnetwork', 'thespynextdoor', 'thetrumanshow', 'thewhale', 'thisistheend', 'thosecommercialsbythetorontofoodmanzabebsiisthebest', 'threeidiots', 'timeofhappiness', 'titanic', 'tokyostory', 'totoro', 'toystory', 'toystory4', 'traintobusan', 'transformer', 'turningred', 'ultramanrising', 'unclegrandpa', 'unknown', 'us', 'venom', 'wags', 'walle', 'waynesworld', 'weatheringwithyou', 'whiplash', 'whoami2005', 'whoamijackiechan', 'wicked', 'wizardsofwaverlyplacemovie', 'wolfofwallstreet', 'wolverine', 'yakuza', 'yehjawaanihaideewani', 'youdontmesswiththezohan', 'yourname', 'zootopia']

# Helpers
def clean_movie_text(text):
    """
    Takes a raw free-text movie string, cleans and maps it to a standard category.
    Returns a string representing a known movie.
    """

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

# Apply the cleaning function
df['Q5_clean'] = df["Q5: What movie do you think of when thinking of this food item?"].apply(clean_movie_text)

print(df['Q5_clean'].value_counts())

# drop or keep the original Q6 column, depending on use case of our later chosen model:
#df.drop(columns=['Q5: What movie do you think of when thinking of this food item?'], inplace=True)

# Save the cleaned dataset
df.to_csv('cleaned_data_for_model.csv', index=False, na_rep='None')
