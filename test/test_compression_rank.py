import unittest


from src.algorithms.utils import get_script_path, compare_games
import io, os, sys, multiprocessing

from src.algorithms.transform import TransformOut, game_from_pgn_to_uci, game_from_uci_to_pgn

from src.algorithms.encoder import Encoder

class Test_compression_rank(unittest.TestCase): 

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.data_path = '../test_data/test_file.pgn'
        self.path = get_script_path()
        self.BATCH_SIZE = int(1e4)

        self.encoder_one_worker = Encoder('rank', par_workers=1, batch_size=self.BATCH_SIZE)
        self.encoder_mul_workers = Encoder('rank', par_workers=4, batch_size=self.BATCH_SIZE)

        self.transform_out = TransformOut(
            move_repr=game_from_uci_to_pgn
        )

    def __simple_worker(self, Q_in, Q_out):

        data = ''
        while 1:

            d = Q_in.get()
            if d == 'kill': break
            else: data += d
        
        return data

    def test_reader(self,):

        encoder = Encoder('apm', batch_size=self.BATCH_SIZE)

        with open(self.path + self.data_path, 'r') as f:
            source_data = f.read()

        ref_data = source_data
        Q = multiprocessing.Queue()

        reader = multiprocessing.Process(target=encoder._Encoder__reader, 
                                         args=(self.path + self.data_path, Q, False, None))
        reader.start()
        
        out_data = ''
        while 1:
            d = Q.get()
            if d == 'kill': break
            out_data += d

        reader.join()

        self.assertEqual(ref_data, out_data)

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

if __name__ == "__main__":

    unittest.main()