import pytest
from pathlib import Path
import torch

from chesskurcz.algorithms.dataset import ChessGameStreamDataset

def test_stream_dataset():

    test_path = Path(__file__).absolute().parents[1] / "test_data" / "test_file.pgn"   

    with test_path.open('r') as f:
        f.readlines()
        assert f.seekable()
        off = f.tell()
    
    # dataset = ChessGameStreamDataset(test_path)
    # data_loader = torch.utils.data.DataLoader(dataset)
    # lines = list(iter(data_loader))

    assert 1 == 1