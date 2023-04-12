import unittest
from test.test_compression_apm import Test_compression_apm
from test.test_transform_in import Test_transform_in
from test.test_compression_rank import Test_compression_rank
from test.test_compression_naive import Test_compression_naive

def main():

    suite = unittest.TestSuite()
    # suite.addTest(Test_transform_in("test_transform"))
    suite.addTest(Test_compression_naive("test_process_mul_threads"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':

    main()