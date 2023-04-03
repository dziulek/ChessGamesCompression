import unittest

from src.algorithms.fast_naive import encode_naive, decode_naive
from src.algorithms.utils import get_script_path, preprocess_lines, move_token_reg, thrash_token_reg
from src.experiments import process_encode, process_decode
from src.algorithms.apm import move_transform
import io, os, sys

class Test_compression_naive(unittest.TestCase): 

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.data_path = '/../test_data/test_file.txt'
        self.path = get_script_path()
        self.BATCH_SIZE = int(1e4)
        self.transform = (
            preprocess_lines,
            {
                'regex_drop': thrash_token_reg,
                'regex_take': move_token_reg,
                'token_transform': move_transform
            }
        )

    def test_process_encode_decode(self,):

        f = open(self.path + self.data_path, 'r')
        comp = open('__tmp.bin', 'wb') 
        
        process_encode(
            f, comp, encode_naive, batch_s=self.BATCH_SIZE, pre_transform=self.transform
        )

        f.close()
        comp.close()

        with open(self.path + self.data_path, 'r') as f:
            ref = f.readlines()
            ref = preprocess_lines(ref, **self.transform[1])

        comp = open('__tmp.bin', 'rb')
        decomp = io.StringIO()

        process_decode(
            comp, decomp, decode_naive, self.BATCH_SIZE
        )

        comp.close()

        decomp = preprocess_lines(decomp.getvalue().split('\n'), **self.transform[1])
        
        self.assertEqual(ref, decomp)

        os.remove('__tmp.bin')

    def test_compression_validity_multiple_theads(self,):

        pass 

if __name__ == "__main__":

    unittest.main()