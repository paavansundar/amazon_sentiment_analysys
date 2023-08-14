import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
import io

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

from tensorflow import keras

from sentiment_model import __version__ as _version
from sentiment_model.config.core import config
from sentiment_model.processing.data_manager import load_model
#from sentiment_model.processing.data_manager import pre_model_preparation
#from sentiment_model.processing.validation import validate_inputs


model_file_name = f"{config.app_config.model_save_file}{_version}.pkl"
model = load_model(file_name = model_file_name)

def load_tokenizer():
    tokenizer=None
    with open(str(parent.absolute())+"/trained_tokenizer/tokenizer.json") as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer

    

#print(type(tokenizer))

def make_prediction():
    tokenizer=load_tokenizer()
    #Let us test some  samples
    test_sample_1 = "This product is fantastic! I really like it because it is so good!"


    test_samples_tokens = tokenizer.texts_to_sequences(test_samples)
    test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=100)

    # predict
    pred = model.predict(x=test_samples_tokens_pad)
    assert lent(pred) > 0
