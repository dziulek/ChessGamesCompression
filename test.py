import unittest
from test.test_compression_apm import Test_compression_apm
from test.test_transform_in import Test_transform_in
from test.test_compression_rank import Test_compression_rank
from test.test_compression_naive import Test_compression_naive
from test.test_encoder import Test_encoder
from test.test_pgn_to_uci_game import Test_pgn_to_uci_game

def main():

    suite = unittest.TestSuite()
    # suite.addTest(Test_transform_in("test_transform"))
    suite.addTest(Test_pgn_to_uci_game("test_conversion_time"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':

    main()