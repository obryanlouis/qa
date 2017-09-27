"""Defines the model types.
"""

from model.debug_model import DebugModel
from model.logistic_regression import LogisticRegression
from model.match_lstm import MatchLstm

MODEL_TYPES = {
    "debug": DebugModel,
    "logistic_regression": LogisticRegression,
    "match_lstm": MatchLstm,
}
