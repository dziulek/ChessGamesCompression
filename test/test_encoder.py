import unittest


from chesskurcz.algorithms.util.utils import get_workspace_path, compare_games
import io, os, sys, multiprocessing

from chesskurcz.algorithms.transform import TransformOut, game_from_pgn_to_uci, game_from_uci_to_pgn
from chesskurcz.algorithms.encoder import Encoder

class Test_encoder(unittest.TestCase): 

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.data_path = 'test_data/test_file.pgn'
        self.path = get_workspace_path()
        self.BATCH_SIZE = int(1e4)

        self.encoder_one_worker = Encoder('apm', par_workers=1, batch_size=self.BATCH_SIZE)
        self.encoder_mul_workers = Encoder('apm', par_workers=4, batch_size=self.BATCH_SIZE)

        self.transform_out = TransformOut(
            move_repr=game_from_uci_to_pgn
        )

        self.N = [20, 30, 50, 75, 100]

    def test_read_batch_games_one_thread(self,) -> None:

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
            file_path, enc_file_name
        )
        
        self.assertEqual(True, enc_file_name in set(os.listdir()))
        source_data = source_data.strip().split('\n') 
        TOTAL = len(source_data)
        current = 0

        STEP = 100

        while current < TOTAL:

            alg_output_file = '__dec.txt'
            self.encoder_one_worker.decode_batch_of_games(
                enc_file_name, alg_output_file, N=STEP, verbose=True
            )

            self.assertEqual(True, alg_output_file in set(os.listdir()))

            with open(alg_output_file, 'r') as f:
                dec_data = f.readlines()
                dec_data = [g.strip() for g in dec_data]

            self.assertEqual(min(TOTAL - current, STEP), len(dec_data))

            for g_dec, g_src in zip(dec_data, source_data[current : min(TOTAL, current + STEP)]):
                self.assertEqual(g_dec, g_src,
                                msg=repr(g_dec) + '\n' + repr(g_src))

            current += STEP
            os.remove(alg_output_file)

        os.remove(enc_file_name)        
        