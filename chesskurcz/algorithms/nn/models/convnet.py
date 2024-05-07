import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, Union, List

from chesskurcz.algorithms.nn.nn_utils import infer_conv_output_dim

class ConvNet(nn.Module):

    def __init__(self, input_dim: Tuple, 
                 channels: Union[List, Tuple], 
                 kernel_sizes: Union[List, Tuple], 
                 dense_num_neurons: Union[List, Tuple]) -> None:

        super().__init__()

        assert(len(input_dim) == 3)
        assert(len(channels) == len(kernel_sizes) == len(dense_num_neurons))

        self.num_conv_layers = len(channels)
        self.num_dense_layers = len(dense_num_neurons)

        c_in, h_in, w_in = input_dim
        self.input_dim = input_dim
        self.channels_num = channels
        self.kernel_sizes = kernel_sizes
        self.dense_num_neurons = dense_num_neurons

        self.conv_layers = nn.Sequential(
            *[nn.Conv2d(in_channels=c_in, 
                        out_channels=self.channels_num[i], 
                        kernel_size=self.kernel_sizes[i]) for i in range(self.num_conv_layers)],
        )

        conv_output_dim = infer_conv_output_dim(self.input_dim, self.conv_layers)
        assert(conv_output_dim > self.dense_num_neurons[0])

        self.dense_layers = nn.Sequential(
            *[nn.Linear(conv_output_dim, 
                        self.dense_num_neurons[i]) for i in range(self.num_dense_layers)]
        )

    def forward(self, x):

        x = self.conv_layers(x)
        x = nn.Flatten()(x)
        x = self.dense_layers(x)

        return x
