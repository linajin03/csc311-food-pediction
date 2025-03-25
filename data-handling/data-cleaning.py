import pandas as pd
import numpy as np

from q2 import clean_Q2
from q4 import clean_price
from q5 import clean_movie_text
from q6 import clean_drink_text


def clean(path, output_path="cleaned_data.csv"):
    """
    This function takes the dogwater data given to us and applies our seperate cleaning functions to each column: q2(lina), q4(jihyuk), q5(vincent), q6(fiona)
    The functions are imported from seperate files for readiability

    Any missing textual data is repalced with "Unknown", and any missing numerical data is replaced with -1

    """
    
    df = pd.read_csv(path)

    #rename columns
    df.rename(columns={'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': 'Q1', 
                       'Q2: How many ingredients would you expect this food item to contain?': 'Q2', 
                       'Q3: In what setting would you expect this food to be served? Please check all that apply': 'Q3',
                       'Q4: How much would you expect to pay for one serving of this food item?': 'Q4',
                       'Q5: What movie do you think of when thinking of this food item?': 'Q5',
                       'Q6: What drink would you pair with this food item?': 'Q6',
                       'Q7: When you think about this food item, who does it remind you of?': 'Q7',
                       'Q8: How much hot sauce would you add to this food item?': 'Q8',
                       }, inplace=True)

    #clean each column of the dataframe (soz this is a little messy :P - vince)

    #Q1 requires no cleaning
    
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

    #fill any missing values in df with Unknown or -1
    for col in ["Q3", "Q7", "Q8"]:
        df[col] = df[col].fillna("Unknown")
    for col in ["Q1", "Q2", "Q4"]:
        df[col] =  df[col].fillna(-1)

    #encode_label(df)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    clean("data/cleaned_data_combined_modified.csv", "data/cleaned_data.csv")
    

