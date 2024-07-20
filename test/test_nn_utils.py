import pytest
import numpy as np
import torch
import chess

from chesskurcz.algorithms.nn.nn_utils import make_input, make_move_label

@pytest.mark.parametrize("uci,ref_output", [
    (
        "e3e7",
        torch.tensor([
            [0,0,0,1,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,1,0],
        ])
    ),
    (
        "a5b8",
        torch.tensor([
            [1,0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0,0],
            [0,2,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1],
        ])
    )
])
def test_make_move_label(uci: str, ref_output: torch.Tensor):

    out_uci = make_move_label(None, chess.Move.from_uci(uci))
    np.testing.assert_array_almost_equal(ref_output, out_uci)