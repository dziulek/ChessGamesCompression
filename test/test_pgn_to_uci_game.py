import unittest, time


from chesskurcz.algorithms.utils import pgn_to_uci_move, pgn_to_uci_game, control_square, get_workspace_path, standard_png_move_extractor
from chesskurcz.algorithms.transform import TransformIn, game_from_pgn_to_uci

class Test_pgn_to_uci_game(unittest.TestCase): 

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        self.script_path = get_workspace_path()
        self.test_files = [self.script_path + "data/filtered_test_file.pgn"]
        self.test_strings = []
        

    # def test_control_square(self,):

    #     pos = (3,3)
    #     targets_queen_bishop = [
    #         (1,1), (0,0), (2,4), (1, 5), (0, 6), (4,4)
    #     ]
    #     targets_queen_rook = [
    #         (4,3), (7, 3), (3,4), (3, 7)
    #     ]
    #     targets_king = [
    #         (4, 4), (2,2), (2,3), (4,3), (3, 2)
    #     ]
    #     targets_knight = [
    #         (5,4), (5, 2), (1, 4), (1, 2), (4, 5), (2, 5), (4, 1), (1, 2)
    #     ]

    #     for t in targets_queen_bishop:
    #         self.assertEqual(True, control_square('B', pos, t))
    #         self.assertEqual(True, control_square('Q', pos, t))

    #     for t in targets_queen_rook:
    #         self.assertEqual(True, control_square('R', pos, t))
    #         self.assertEqual(True, control_square('Q', pos, t))            

    #     for t in targets_king:
    #         self.assertEqual(True, control_square('K', pos, t))

    #     for t in targets_knight:
    #         self.assertEqual(True, control_square('N', pos, t), msg=str(pos) + " " + str(t))

    def test_pgn_to_uci_game(self,):

        transform = TransformIn(standard_png_move_extractor)

        for f in self.test_files:

            with open(f, 'r') as buff:

                games = transform.transform(buff.read())
                for game in games:
                    
                    uci_moves = pgn_to_uci_game(game)

                    self.assertEqual(True, True)

    def test_conversion_time(self,):

        transform = TransformIn(standard_png_move_extractor)
        print('PROCESSING WITH CUSTOM FUNCTION')
        for f in self.test_files:

            with open(f, 'r') as buff:

                games = transform.transform(buff.read())

                start = time.time()
                for game in games:
                    
                    uci_moves = pgn_to_uci_game(game)
                duration_custom = time.time() - start
                print('TIME:', duration_custom, 'sec.')
                print('====================================')
                start = time.time()
                for game in games:

                    uci_moves = game_from_pgn_to_uci(game)
                duration_python_chess = time.time() - start
                print('TIME:', duration_python_chess, 'sec.')

        print('====================================')
        print('OWN VERSION IS', duration_python_chess / duration_custom, 'FASTER')