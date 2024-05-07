import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple

import chess

class CompressionMetric:
    """
        Calculates the maximum number of bits required to 
        encode one move in a particular game. The result is 
        averaged across games.
    """
    def __init__(self,):

        self.max = {}
        self.sums = {}
        self.count_per_game = {}
        self.count = 0

    def update(self, game_nums: List[int], 
               fen: List[str], move_uci: torch.Tensor, 
               predicted: torch.Tensor, targets: torch.Tensor):

        metric = []
        for i, game_num, position, move in enumerate(zip(game_nums, fen, move_uci)):
            if game_num not in self.max.keys():
                self.max[game_num] = 0
                self.sums[game_num] = 0
                self.count_per_game[game_num] = 0 

            metric_val = self.calculate(position, move, predicted[i], targets[i])
            self.max[game_num] = max(self.max[game_num], metric_val)
            self.sums[game_num] += metric_val
            self.count_per_game[game_num] += 1
            self.count += 1

    def calculate(self, fen: str, move_uci: str,
                  predicted: torch.Tensor, target: torch.Tensor) -> int:

        position = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)

        rank = self.__calculate_move_rank(position, predicted, target)

        return  np.floor(np.log2(rank)) + 1
    
    def __calculate_move_rank(self, position: chess.Board,
                              predicted: torch.Tensor, target: torch.Tensor) -> int:
        
        # TODO
        pass

    def result(self,) -> Tuple[float, float]: 

        if not self.count:
            return None, None 

        avg_max_bits = np.sum(list(self.max.values()), dtype=np.float32) / len(self.count_per_game)
        avg_agv_bits = np.sum(list(self.sums.values()), dtype=np.float32) / self.count

        return avg_max_bits, avg_agv_bits