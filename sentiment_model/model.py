import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

from sentiment_model.config.core import config
from sentiment_model.processing.features import SentimentEncoder


sentiment_pipe = Pipeline([
    
    ######## One-hot encoding ########
    ('encode_sentiment', SentimentEncoder(variable = config.model_config.target)),
    
    # Regressor
    ('model_rf', LSTM(n_estimators = config.model_config.n_estimators, 
                                       max_depth = config.model_config.max_depth,
                                      random_state = config.model_config.random_state))
    
    ])
