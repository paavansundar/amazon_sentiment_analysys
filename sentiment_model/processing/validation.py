from typing import List, Optional, Tuple, Union

from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from sentiment_model.config.core import config
from sentiment_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame = input_df)
    validated_data = pre_processed
    errors = None

    return validated_data
