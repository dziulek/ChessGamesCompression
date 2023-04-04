import unittest

import src

from src.algorithms.utils import get_script_path, preprocess_lines, move_token_reg, thrash_token_reg
from src.algorithms.algorithm import Encoder
import io, os, sys

class Test_compression_rank(unittest.TestCase): 

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.data_path = '/../test_data/test_file.txt'
        self.path = get_script_path()
        self.BATCH_SIZE = int(1e4)

        self.encoder_one_thread = Encoder('apm', thread_no=1, batch_size=self.BATCH_SIZE)
        self.encoder_mul_threads = Encoder('apm', thread_no=4, batch_size=self.BATCH_SIZE)

    def test_process_one_thread(self,):

        f = open(self.path + self.data_path, 'r')
        comp = open('__tmp.bin', 'wb') 
        
        self.encoder_one_thread.encode(
            f, comp
        )

        f.close()
        comp.close()

        with open(self.path + self.data_path, 'r') as f:
            ref = '\n'.join(f.readlines())
            ref = self.encoder_one_thread.def_pgn_parser.transform(ref)
        comp = open('__tmp.bin', 'rb')
        decomp = open('__dec.txt', 'w')

        self.encoder_one_thread.decode(
            comp, decomp
        )

        comp.close()
        decomp.close()

        decomp = open('__dec.txt', 'r')        
        a = decomp.read()
        a = self.encoder_one_thread.def_pgn_parser.transform(a)

        decomp.close()
        os.remove('__tmp.bin')
        os.remove('__dec.txt')

        self.assertEqual(ref, a)

    def test_compression_validity_multiple_theads(self,):

        pass 

if __name__ == "__main__":

    unittest.main()