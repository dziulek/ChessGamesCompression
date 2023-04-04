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

        
        with open(self.path + self.data_path, 'r') as f:
            source_data = io.StringIO(initial_value=f.read())
        source_data.seek(0)

        comp = io.BytesIO()
        comp.seek(0)
        
        self.encoder_one_thread.encode(
            source_data, comp
        )
        print(comp.getvalue())
        comp.seek(0)

        ref = source_data.getvalue()
        decomp = io.StringIO()

        self.encoder_one_thread.decode(
            comp, decomp
        )
        decomp.seek(0)

        a = decomp.read()
        a = self.encoder_one_thread.def_pgn_parser.transform(a)

        comp.close()
        decomp.close()
        source_data.close()
        self.assertEqual(ref, a)

    def test_compression_validity_multiple_theads(self,):

        pass 

if __name__ == "__main__":

    unittest.main()