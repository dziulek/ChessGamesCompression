import threading
import os, io, sys
import math

from typing import List, Dict, Tuple, Callable
from src.stats import Stats
from src.compressors import REV_SCORE_MAP, SCORE_MAP
from src.utils import to_binary

import re

LOOKUP_TABLE = {
    'a': 0x0,
    'b': 0x1,
    'c': 0x2,
    'd': 0x3,
    'e': 0x4,
    'f': 0x5,
    'g': 0x6,
    'h': 0x7,
    '1': 0x8,
    '2': 0x9,
    '3': 0xA,
    '4': 0xB,
    '5': 0xC,
    '6': 0xD,
    '7': 0xE,
    '8': 0xF,
    'x': 0x10,
    'O-O': 0x11,
    'O-O-O': 0x12,
    '=': 0x13,
    '+': 0x14,
    '#': 0x15,
    'Q': 0x16,
    'K': 0x17,
    'R': 0x18,
    'B': 0x19,
    'N': 0x1A
}

# MOVE_REGEX = re.compile(r'[')

def encode_naive(games: List[str], stats: Stats) -> bytes:

    enc_data = []
    bin_data = bytes()

    for game in games:
    
        ind = 0
        f_char = -1
        
        while ind < len(game):

            if game[ind] == '{':
                while game[ind] != '}': ind+=1

            if game[ind] != ' ' and f_char == -1:
                f_char = ind
            
            if game[ind] == ' ':

                if game[f_char: ind].find('.') == -1:
                    if game[f_char: ind] == 'O-O':
                        enc_data.append(LOOKUP_TABLE['O-O'])

                    elif game[f_char: ind] == 'O-O-O':
                        enc_data.append(LOOKUP_TABLE['O-O-O'])

                    else:

                        for c in game[f_char : ind]:
                            enc_data.append(LOOKUP_TABLE[c])
                f_char = -1

            ind += 1

    out = bytes()

    occ_bits = 0
    _bin = int(0)
    _carry = int(0)
    BITS = 32

    for b in enc_data:
        
        _bin, _carry = to_binary(
            _bin, BITS, occ_bits, b, 5
        )

        if _carry >= 0:

            bin_data += _bin.to_bytes(4, 'big')
            _bin = _carry

    return bin_data

def decode_naive(r_buff: io.TextIOWrapper, batch_s: int) -> List[str]:

    pass

def main():

    pass

if __name__ == "__main__":

    main()