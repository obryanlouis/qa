"""Defines the model types.
"""

from model.debug_model import DebugModel
from model.match_lstm import MatchLstm
from model.rnet import Rnet

MODEL_TYPES = {
    "debug": DebugModel,
    "match_lstm": MatchLstm,
    "rnet": Rnet,
}
