import torch
import numpy as np
import json
import os
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Iterable, Callable, Optional, Union
from pathlib import Path
from threading import Semaphore

class ChessGameStreamDataset(torch.utils.data.IterableDataset):

    def __init__(self, data_stream: Union[str,Path], 
                 read_batch_size: int = 8192,
                 transform: Optional[Callable] = None):
        super(ChessGameStreamDataset).__init__()

        if isinstance(data_stream, str):
            data_stream = Path(data_stream)
        
        self.data_stream_sem = Semaphore(1)
        self.read_batch_size = read_batch_size
        self.data_stream = data_stream
        self.offset = 0 

        self.transform = transform

    def __iter__(self,) -> Iterable:
        ...


class ChessGameDataset(torch.utils.data.Dataset):

    ROOT_DATASET = Path(__file__).absolute().parents[2] / 'datasets'

    def __init__(self, dataset_path: str,
                 num_items: int = None, 
                 transform_label: Callable = None, 
                 transform_data: Callable = None):
        
        super().__init__()

        dataset_path: Path = Path(dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = ChessGameDataset.ROOT_DATASET / dataset_path

        assert('params.json' in os.listdir(str(dataset_path)))
        assert('data.txt' in os.listdir(str(dataset_path)))
        assert('labels.txt' in os.listdir(str(dataset_path)))

        with open(dataset_path / 'params.json', 'r') as f:
            self.metadata = json.load(f)
        
        with open(dataset_path / 'data.txt', 'r') as f:
            lines = f.readlines()
            lines: List[str] = list(filter(lambda l: l, self.data))
            if num_items is not None:
                lines = lines[:min(len(lines), num_items)]
            self.data = []
            for i, line in enumerate(lines):
                self.data += [(i, pos) for pos in line.split(sep=self.metadata['line_sep'])]
            

        with open(dataset_path / 'labels.txt', 'r') as f:
            lines = f.readlines()
            lines = list(filter(lambda l: l, lines))
            if num_items is not None:
                lines = lines[:min(len(lines), num_items)]
            self.labels = []
            for i, line in enumerate(lines):
                self.labels += [(i, move) for move in lines.split(sep=self.metadata['line_sep'])]

        assert(len(self.data) == len(self.labels))

        self.transform_label = transform_label 
        self.transform_data = transform_data
    
    def __len__(self):
        return self.data
    
    def __getitem__(self, index):
        
        game_num, position = self.data[index]
        _, target = self.labels[index]
        
        if self.transform_label:
            target = self.transform_label(target)
        if self.transform_data:
            position = self.transform_data(position)        

        return game_num, position, target