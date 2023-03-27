import numpy as np
from typing import List, Dict
import dahuffman
import os

HUFFMAN_TOKENS = ['O-O-O', '0-0', '1-0', '0-1', '1/2-1/2', 'x', 'N', 'B', 'K', 'Q', 'R']

def add_board_field_tokens():

    global HUFFMAN_TOKENS

    for i in range(8):
        for j in range(8):

            HUFFMAN_TOKENS.append(chr(ord('a') + i) + str(j + 1))

def main():

    add_board_field_tokens()

    file = open('../archive/Carlsen_clean.txt', 'r')
    file_base = open('../archive/Carlsen_benchmark.txt', 'r')

    lines_base = file_base.readlines()
    file_base.close()

    half_move_no = 0
    for line in lines_base:
        half_move_no += len(line.split(' ')) - 1

    print("Number of half moves in the origin data file:", half_move_no)
    data_base = ''.join(lines_base)
    print("Size in bytes of origin file", len(data_base) - len(lines_base))
    print("Size of compressed origin file", os.path.getsize('../archive/Carlsen_benchmark.zip'))

    lines = file.readlines()
    file.close()

    for line in lines:
        if line[-1] == '\n':
            line = lines[:-1]

    data = ''.join(lines)

    print(f"Size of the file after removing redundant char.: {len(data)}, decreased by {1 - len(data)/len(data_base):.2f}, bits per move {8 * len(data) / half_move_no:.2f}")

    codec = dahuffman.HuffmanCodec.from_data(data)
    encoded = codec.encode(data)

    print(f"Size of the file after Huffman encoding: {len(encoded)}, decreased by { 1 - len(encoded)/len(data_base):.2f}, bits per move {8 * len(encoded) / half_move_no:.2f}")

if __name__ == "__main__":

    main()