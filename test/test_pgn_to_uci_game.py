import unittest


from chesskurcz.algorithms.utils import get_script_path, compare_games
import io, os, sys, multiprocessing

from chesskurcz.algorithms.transform import TransformOut, game_from_pgn_to_uci, game_from_uci_to_pgn

from chesskurcz.algorithms.encoder import Encoder

class Test_pgn_to_uci_game(unittest.TestCase): 

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        pass