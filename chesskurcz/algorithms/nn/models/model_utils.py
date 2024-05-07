import torch
import torch.nn as nn
import numpy as np


def create_model(base_type: str = 'convnet', *args, **kwargs) -> nn.Module:

    pass

def calculate_params_num(model: nn.Module) -> int:

    params_num = 0
    for param in model.parameters():
        if param.requires_grad:
            factor = np.dtype(param.dtype).alignment
            params_num += np.prod(param.shape) * factor
         
    return params_num