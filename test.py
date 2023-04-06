import unittest
from  test.test_compression_apm import Test_compression_rank

def main():

    suite = unittest.TestSuite()
    suite.addTest(Test_compression_rank("test_reader"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':

    main()