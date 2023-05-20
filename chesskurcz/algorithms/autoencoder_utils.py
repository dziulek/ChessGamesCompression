import chess, chess.pgn
import numpy as np

from chesskurcz.algorithms.utils import *

def default_uci_move_repr(uci_move: str) -> np.ndarray:

    if len(uci_move) == 0: return np.zeros(5) - 1
    if uci_move in set(POSSIBLE_SCORES): return np.zeros(5) - 1
    PROMOTION = 0
    if len(uci_move) > 4:
        PROMOTION = ord(uci_move[4]) - ord('a')

    return np.array([
        ord(uci_move[0]) - ord('a'),
        int(uci_move[1]) - 1,
        ord(uci_move[2]) - ord('a'),
        int(uci_move[3]) - 1,
        PROMOTION
    ])