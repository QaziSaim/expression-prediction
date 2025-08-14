import numpy as np
import pandas as pd

import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')

nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_number(text):
    text = [n for n in text if not n.isdigit()]
    return ''.join(text)

def lower_case(text):
    text = text.split()
    text = [n.lower() for n in text]
    return " ".join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()
def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stopwords(content))
    df.content=df.content.apply(lambda content : removing_number(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df

train_data_process = normalize_text(train_data)
test_data_process = normalize_text(test_data)

data_path = os.path.join('E:\MLOPOS\DVC-RUN\data','processed')

os.makedirs(data_path)

train_data_process.to_csv(os.path.join(data_path,'train_processed.csv'))
test_data_process.to_csv(os.path.join(data_path,'test_processed.csv'))

