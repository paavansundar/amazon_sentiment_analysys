import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
import numpy as np
import json
import io

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers


from sentiment_model.config.core import config
from sentiment_model.processing.data_manager import load_dataset, save_model

def run_training() -> None:
    print("Running training")  
    """
    Train the model.
    """

    # read training data
    print("Reading data")
    data = load_dataset(file_name = config.app_config.training_data_file)
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data.Text,     # predictors
        data.Sentiment,       # target
        test_size = config.model_config.test_size,
        random_state=config.model_config.random_state,   # set the random seed here for reproducibility
    )
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    print("X_train shape",X_train.shape)
    tokenizer_json_dump=save_tokenizer(tokenizer)

    X_train_tok = tokenizer.texts_to_sequences(X_train)
    X_test_tok = tokenizer.texts_to_sequences(X_test)
 
# Find the vocabulary size and perform padding on both train and test set
    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 100
    X_train_pad = pad_sequences(X_train_tok, padding='post', maxlen=maxlen, truncating='post')
    X_test_pad = pad_sequences(X_test_tok, padding='post', maxlen=maxlen, truncating='post')

    print("Training tokens ",X_train_pad.shape)
    EMBEDDING_DIM = 32

    print('Build model...')

    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = EMBEDDING_DIM, input_length=maxlen))
    model.add(LSTM(units=40,  dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

# Try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Summary of the built model...')
    print(model.summary())
    # model fitting
    print("Fitting model with tokens",X_train_pad.shape,y_train.shape)
    history = model.fit(X_train_pad, y_train, batch_size=128, epochs=1, verbose=1, validation_split=0.2)
    print("model was fit")
    # Calculate the score/error
    #Final evaluation of the model on test data
    print('Testing...')
    y_test = np.array(y_test)
    score, acc = model.evaluate(X_test_pad, y_test, batch_size=128)

    print('Test score:', score)
    print('Test accuracy:', acc)

    print("Accuracy: {0:.2%}".format(acc))

    # persist trained model
    save_model(model_to_persist = model)
    #save_tokenizer(tokenizer)


def save_tokenizer(tokenizer_to_save):
  json_tokenizer=tokenizer_to_save.to_json()
  with io.open(str(parent.absolute())+"/trained_tokenizer/tokenizer.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(json_tokenizer, ensure_ascii=False))
 

   
if __name__ == "__main__":
    run_training()
