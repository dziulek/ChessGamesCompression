import pytest
from chesskurcz.algorithms.util.utils import get_workspace_path, compare_games
import io, os, sys, multiprocessing

from chesskurcz.algorithms.transform import TransformOut, \
    game_from_pgn_to_uci, game_from_uci_to_pgn
from chesskurcz.algorithms.encoder import Encoder

@pytest.mark.parametrize("algorithm", ["naive", "rank", "apm"])
class TestCompressionAlgorithms:

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.data_path = 'test_data/test_file.pgn'
        self.path = get_workspace_path()
        self.BATCH_SIZE = int(1e4)

        self.encoder_one_worker = Encoder('rank', num_workers=1, batch_size=self.BATCH_SIZE)
        self.encoder_mul_workers = Encoder('rank', num_workers=4, batch_size=self.BATCH_SIZE)

        self.transform_out = TransformOut(
            move_repr=game_from_uci_to_pgn
        )

    def test_process_one_thread(self,):

        file_path = self.path + self.data_path
        enc_file_name = '__tmp.bin'

        source_data = None

        with open(file_path, 'r') as f:
            source_data = f.read()
            source_data = self.encoder_one_worker.def_out_format.transform(
                self.encoder_one_worker.def_pgn_parser.transform(source_data)
            )

        self.assertIsNotNone(source_data)

        # encode the file
        self.encoder_one_worker.encode(
            file_path, enc_file_name, verbose=True
        )
        
        self.assertEqual(True, enc_file_name in set(os.listdir()))

        alg_output_file = '__dec.txt'
        self.encoder_one_worker.decode(
            enc_file_name, alg_output_file, verbose=True
        )

        self.assertEqual(True, alg_output_file in set(os.listdir()))

        with open(alg_output_file, 'r') as f:
            dec_data = f.readlines()
            dec_data = [g.strip() for g in dec_data]

        source_data = source_data.strip().split('\n')   

        self.assertEqual(len(source_data), len(dec_data))

        for src_g, dec_g in zip(source_data, dec_data):

            self.assertEqual(True, compare_games(src_g, dec_g), msg=repr(src_g) + '\n' + repr(dec_g))

        os.remove(alg_output_file)
        os.remove(enc_file_name)

    def test_process_mul_threads(self,):

        file_path = self.path + self.data_path
        enc_file_name = '__tmp.bin'

        source_data = None

        with open(file_path, 'r') as f:
            source_data = f.read()
            source_data = self.encoder_mul_workers.def_out_format.transform(
                self.encoder_mul_workers.def_pgn_parser.transform(source_data)
            )

        self.assertIsNotNone(source_data)

        # encode the file
        self.encoder_mul_workers.encode(
            file_path, enc_file_name, verbose=True
        )
        
        self.assertEqual(True, enc_file_name in set(os.listdir()))

        alg_output_file = '__dec.txt'
        self.encoder_mul_workers.decode(
            enc_file_name, alg_output_file, verbose=True
        )

        self.assertEqual(True, alg_output_file in set(os.listdir()))

        with open(alg_output_file, 'r') as f:
            dec_data = f.readlines()
            dec_data = [g.strip() for g in dec_data]

        source_data = source_data.strip().split('\n')  
        source_data = [' '.join(game_from_pgn_to_uci(g)) for g in source_data]

        self.assertEqual(len(source_data), len(dec_data))

        # sort the games, they can be permutated
        source_data.sort()
        dec_data.sort()

        for src_g, dec_g in zip(source_data, dec_data):

            self.assertEqual(src_g, dec_g, msg=repr(src_g) + '\n' + repr(dec_g))
                
        os.remove(alg_output_file)
        os.remove(enc_file_name)
    