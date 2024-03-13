import chess, chess.pgn
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import jaxtyping as jax

from chesskurcz.algorithms.util.utils import *

DEF_REPR_T_SIZE = (4, 1, 8) # 4 channels, one row and one-hot encoding for eight classes (row or column)
NULL_MOVE = torch.zeros(DEF_REPR_T_SIZE)

def default_uci_move_repr(uci_move: str) -> jax.Float[torch.Tensor, "4 1 8"]:

    if len(uci_move) == 0: return torch.zeros(DEF_REPR_T_SIZE)
    if uci_move in set(POSSIBLE_SCORES): return torch.zeros(DEF_REPR_T_SIZE)

    encoded = torch.tensor([
        ord(uci_move[0]) - ord('a'),
        int(uci_move[1]) - 1,
        ord(uci_move[2]) - ord('a'),
        int(uci_move[3]) - 1
    ])

    return one_hot(encoded, num_classes=8).reshape(DEF_REPR_T_SIZE)

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
            