import unittest

from chesskurcz.algorithms.util.utils import get_workspace_path, standard_png_move_extractor, POSSIBLE_SCORES
from chesskurcz.algorithms.transform import TransformIn

import re, io
import chess.pgn

class Test_transform_in(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        self.data_path = 'test_data/test_file.pgn'
        self.path = get_workspace_path()

        self.transform = TransformIn(
            standard_png_move_extractor, 
            re.compile(r'(\?|\!|\{[^{}]*\}|\[[^\[\]]*\]|\n|\#|\+)')
        )

    def test_transform(self,):

        src_data = None

        abs_path = self.path + self.data_path
        with open(abs_path, 'r') as f:
            src_data = ''.join(f.readlines())

        self.assertIsNotNone(src_data)

        games = self.transform.transform(src_data)
        self.assertEqual(len(games), 102)

        for g in games:

            # check if g is not empty
            self.assertNotEqual(0, len(g))

            # last move should be the score of the game
            self.assertEqual(True, g[-1] in set(POSSIBLE_SCORES))
            # there should be only one score
            cnt = 0
            for s in POSSIBLE_SCORES:
                cnt += g.count(s)
            self.assertEqual(1, cnt, msg=repr(''.join(g)))
            # check if g can be parsed
            try:
                _ = chess.pgn.read_game(io.StringIO(initial_value=' '.join(g)))
            except:
                self.assertEqual(True, False)

                    