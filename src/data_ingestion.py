import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

url = 'https://raw.githubusercontent.com/entbappy/Branching-tutorial/refs/heads/master/tweet_emotions.csv'
def load_data(data_url:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        return df
    except pd.errors.ParserError as e:
        print(f'Error: faild to parse the csv file from {data_url}')
        print(e)
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading the data.")

