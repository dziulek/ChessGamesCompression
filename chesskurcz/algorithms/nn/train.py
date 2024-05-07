import torch
import torch.nn as nn
import argparse
import chess
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Any

from algorithms.nn.models.convnet import ConvNet
from algorithms.nn.dataset import ChessGameDataset, ChessGameStreamDataset 
from algorithms.nn.nn_utils import Pipe, make_move_label, make_input
from algorithms.nn.models.model_utils import create_model

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--model-config', type=str, default=None)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--loss', '-l', type=str, default=None)
    parser.add_argument('--optimizer', '-o', type=str, default='adam')
    parser.add_argument('--batch-size', '-bs', type=int, default=8)
    parser.add_argument('--loss', type=str, default='cross_entropy')

    args = parser.parse_args()

    dataset_path = args.dataset
    if not Path(args.dataset).is_absolute():
        dataset_path = ChessGameDataset.ROOT_DATASET / dataset_path

    dataset = ChessGameDataset(dataset_path, 
                               transorm_label=Pipe(funcs=make_move_label),
                               transform_data=Pipe(funcs=[lambda fen: chess.Board(fen), make_input]))

    if args.loss == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()

    model_config: str = args.model_config
    if model_config.endswith('.json'):
        with open(model_config, 'r') as f:
            model_config = json.load(f) 
    else:
        model_config = eval(model_config)

    model: nn.Module = create_model(base_type=args.model, **model_config)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters())    

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1) 
    ctx = None
    
    train_loop(args.epochs, dataset, optimizer, model, loss_fn, lr_scheduler, ctx)

def train_loop(epochs: int, dataset: torch.data.Dataset, 
               optimizer: torch.optim.Optimizer,
               model: nn.Module, loss_fn: torch.nn.modules.loss._Loss, 
               lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
               ctx: Dict[str, Any] = None):

    for i in range(epochs):

        start = time.time()
        epoch_step(i, dataset, optimizer, model, loss_fn, lr_scheduler, None, None)
        epoch_duration = time.time() - start



def epoch_step(epoch_num: int, dataset: torch.data.Dataset, 
               optimizer: torch.optim.Optimizer, 
               model: nn.Module, loss_fn: torch.nn.modules.loss._Loss,
               lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
               post_func: Callable = None, ctx: Dict[str, Any] = None):

    for game_num, input, target in dataset:

        optimizer.zero_grad()
        output = model(input)

        loss: torch.Tensor = loss_fn(output, target)
        loss.backward()

        optimizer.step()

    lr_scheduler.step()