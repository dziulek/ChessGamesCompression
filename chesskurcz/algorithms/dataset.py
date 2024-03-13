import torch
import numpy as np
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

    def __init__(self, data: Iterable,
                 targets: Iterable, transform=None):
        
        super(ChessGameDataset).__init__()

        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets)
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        game = self.data[index]
        target = self.targets[index]
        
        if self.transform:
            game = self.transform(game)
        
        return game, target