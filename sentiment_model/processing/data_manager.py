import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
from pathlib import Path


import io
import re
import json
import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

import joblib
import pandas as pd
from datetime import datetime

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder

from sentiment_model import __version__ as _version
from sentiment_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


##  Pre-Pipeline Preparation

def removeUnwantedWords(text):
   #print("Removing unwanted words")
   stop_words = set(stopwords.words('english'))
   #common_words=nltk.FreqDist(text)
   words = [w for w in text if not w in stop_words]
   return "".join(words)


# Extract year and month from the date column and create two another columns
def pre_model_preparation(*, df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing data")
    #drop nulls
    df.dropna(inplace=True,axis=0)
    print("Adding sentiment column")
    df['Sentiment']=np.where(df['Score'] > 3, 'positive', 'negative')
   
    #drop duplicates   
    df['is_duplicate']=df.duplicated()
    df.drop(df[df.is_duplicate == True].index, inplace=True)
    df.drop(['is_duplicate'],axis=1,inplace=True)
    #convert unix time to normal time
    df['Time']=pd.to_datetime(df['Time'], unit='s')
    print("Applying text transformations")
    df['Text'] = df['Text'].apply(lambda x:re.sub('<.*?>',' ',x, flags=re.MULTILINE)) #remove html tags
    df['Text'] = df['Text'].apply(lambda x:re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', x,flags=re.MULTILINE)) #remove punctuation
    df['Text']=df['Text'].apply(removeUnwantedWords)
   
    print("Dropping unnecessary fields")
    # Drop unnecessary fields
    for field in config.model_config.unused_fields:
        if field in df.columns:
            df.drop(labels = field, axis=1, inplace=True)   

    le = LabelEncoder()
    #transformed train reviews
    df['Sentiment']=le.fit_transform(df['Sentiment'])
    
    return df


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    transformed=None
    try:
        dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
        transformed = pre_model_preparation(df = dataframe)
    except Exception as e:
        print(e)
    print("Data Transformation complete")
    return transformed


def save_model(*, model_to_persist: Model) -> None:
    """Persist the model.
    Saves the versioned model, and overwrites any previous saved models. 
    This ensures that when the package is published, there is only one trained model that 
    can be called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.model_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_model(files_to_keep=[save_file_name])
    joblib.dump(model_to_persist, save_path)


def load_model(*, file_name: str) -> Model:
    """Load a persisted model."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_model(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model models.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
