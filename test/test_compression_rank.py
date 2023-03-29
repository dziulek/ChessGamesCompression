import unittest


from src.compressors import encode_rank, decode_rank
from src.utils import get_script_path, filterLines
from experiments import process_encode, process_decode
import io

class Test_compression_rank(unittest.TestCase): 

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.data_path = '/../test_data/test_file.txt'
        self.path = get_script_path()
        self.BATCH_SIZE = int(1e4)

    def test_process_encode_decode(self,):

        f = open(self.path + self.data_path, 'r')
        comp = open('__tmp.bin', 'wb') 
        
        process_encode(
            comp, f, None, encode_rank, batch_s=self.BATCH_SIZE
        )

        f.close()
        comp.close()

        with open(self.path + self.data_path, 'r') as f:
            ref = f.readlines()
            filterLines(ref)
            ref = '\n'.join(ref)

        comp = open('__tmp.bin', 'rb')
        decomp = io.StringIO()

        games = process_decode(
            comp, decomp, None, decode_rank, self.BATCH_SIZE
        )

        comp.close()
        
        self.assertEqual(ref, decomp)

    def test_compression_validity_multiple_theads(self,):

        pass 

if __name__ == "__main__":

    unittest.main()