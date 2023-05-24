import chess, chess.pgn
import numpy as np
import torch
from torch.nn.functional import one_hot

from chesskurcz.algorithms.utils import *

DEF_REPR_T_SIZE = (5, 1, 8) # 5 channels, one row and one-hot encoding for eight classes (row or column)

def default_uci_move_repr(uci_move: str) -> torch.Tensor:

    if len(uci_move) == 0: return torch.zeros((5, 1, 8))
    if uci_move in set(POSSIBLE_SCORES): return torch.zeros((5, 1, 8))
    PROMOTION = 0
    if len(uci_move) > 4:
        PROMOTION = PIECE_TO_INT[uci_move[4].upper()]

    encoded = torch.tensor([
        ord(uci_move[0]) - ord('a'),
        int(uci_move[1]) - 1,
        ord(uci_move[2]) - ord('a'),
        int(uci_move[3]) - 1,
        PROMOTION
    ])

    return one_hot(encoded).reshape((5, 1, 8))