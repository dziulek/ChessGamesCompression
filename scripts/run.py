import torch
import torch.nn as nn
import argparse

from chesskurcz.algorithms.autoencoder import AutoEncoder
from chesskurcz.algorithms.dataset. import BasicDataset

def run_experiment(model: torch.Module,
                   dataset: ):

