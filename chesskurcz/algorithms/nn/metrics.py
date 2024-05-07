import torch
import torch.nn as nn

import chess

class CompressionMetric:
    """
        Calculates the maximum number of bits required to 
        encode one move in a particular game. The result is 
        averaged across games.
    """
    def __init__(self,):

        self.sums = {}
        self.count = {}

    def update(self, game_num: torch.Tensor, 
               fen: torch.Tensor, move_uci: torch.Tensor, 
               data: torch.Tensor, targets: torch.Tensor):

        pass

    def calculate(self, fen: torch.Tensor, move_uci: torch.Tensor,
                  data: torch.Tensor, targets: torch.Tensor):

        positions = [chess.Board(f) for f in fen]
        pass

    def result(self,): 

        pass