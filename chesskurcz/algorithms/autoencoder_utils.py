import chess, chess.pgn
import numpy as np
import torch
from torch.nn.functional import one_hot
import jaxtyping as jax

from chesskurcz.algorithms.utils import *

DEF_REPR_T_SIZE = (4, 1, 8) # 5 channels, one row and one-hot encoding for eight classes (row or column)
NULL_MOVE = torch.zeros(DEF_REPR_T_SIZE)

def default_uci_move_repr(uci_move: str) -> jax.Float[torch.Tensor, "4 1 8"]:

    if len(uci_move) == 0: return torch.zeros(DEF_REPR_T_SIZE)
    if uci_move in set(POSSIBLE_SCORES): return torch.zeros(DEF_REPR_T_SIZE)
    PROMOTION = 0

    # if len(uci_move) > 4:
    # 

    encoded = torch.tensor([
        ord(uci_move[0]) - ord('a'),
        int(uci_move[1]) - 1,
        ord(uci_move[2]) - ord('a'),
        int(uci_move[3]) - 1
    ])

    return one_hot(encoded, num_classes=8).reshape(DEF_REPR_T_SIZE)