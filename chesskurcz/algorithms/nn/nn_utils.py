import chess, chess.pgn
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from typing import Any, Dict, List, Tuple

from chesskurcz.algorithms.util.utils import *


class Pipe:

    def __init__(self, funcs: Callable | List[Callable]) -> None:

        if not isinstance(funcs, list):
            funcs = [funcs] 
        self.funcs = funcs
    
    def __call__(self, input: Any):

        out = input
        for func in self.funcs:
            out = func(out)
        
        return out

def bitboard_to_array(bitboard: chess.Bitboard) -> np.ndarray:

    return [int(b) for b in bin(bitboard)[2:]]


def make_move_label(board: chess.Board, move: chess.Move, 
                    algorithm: str = 'unique', **kwargs):

    assert(algorithm in ('unique', '')) 

    uci_move = str(move)

    if algorithm == 'unique':
        encoded = torch.tensor([
            ord(uci_move[0]) - ord('a'),
            int(uci_move[1]) - 1,
            ord(uci_move[2]) - ord('a'),
            int(uci_move[3]) - 1
        ])
        return one_hot(encoded, num_classes=8)

def make_input(board: chess.Board, 
               add_random_channel: bool = False, 
               device: torch.DeviceObjType = torch.cpu) -> torch.Tensor:

    color = board.turn
    input = []

    for piece_type in chess.PIECE_TYPES:

        bitboard = board.pieces_mask(piece_type, color)
        input.append(torch.tensor(bitboard_to_array(bitboard), dtype=torch.float32, device=device))
    
    if add_random_channel:
        input.append(torch.rand((8,8), dtype=torch.float32, device=device))
    
    return torch.concat(input, dim=0)

def spatial_out_dim(input_dim: Tuple, 
                    kernel_size: Tuple[int, int],
                    stride: Tuple[int, int],
                    padding: Tuple[int, int],
                    dilation: Tuple[int, int]) -> Tuple:

    return input_dim[:-2] + np.floor(
        (np.array(input_dim[-2:]) + 2 * np.array(padding) - np.array(dilation) * (np.array(kernel_size) - 1) - 1) / np.array(stride) + 1
    ) 

def infer_conv_output_dim(input_dim: Tuple,
                          conv_module: nn.Sequential) -> Tuple:

    output_dim = input_dim
    for module in conv_module.modules():
        if 'conv' in module._get_name().lower():

            output_dim = spatial_out_dim(
                input_dim,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation
            )

    return output_dim  
            