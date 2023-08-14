"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sentiment_model.config.core import config
from sentiment_model.predict import make_prediction


def test_age_variable_transformer():
  

    predict=make_prediction(input_data = data_in)
    # Then
    assert len(predict['predictions']) > 0

