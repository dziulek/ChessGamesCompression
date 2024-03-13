import threading
import os, io, sys
import math
import chess
import chess.pgn
import copy
import numpy as np

from typing import List, Dict, Tuple, Callable
from chesskurcz.algorithms.util.utils import to_binary, extract_move_idx, FOUR_1, FOUR_0, BIT_S 
from chesskurcz.algorithms.util.utils import processLine, get_workspace_path, MOVE_REGEX, POSSIBLE_SCORES

import re

BATCH_SIZE = int(1e4)

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
    'N': 0x1A,
    '1-0': 0x1B,
    '1/2-1/2': 0x1C,
    '0-1': 0x1D
}

def encode_naive(games: List[List[str]]) -> bytes:

    enc_data: List[List[int]] = []
    bin_data = bytes()

    for game in games:
        
        enc_data.append([])
                
        # print(moves)
        for move in game:
            
            if move in set(POSSIBLE_SCORES + ['O-O-O', 'O-O']):
                enc_data[-1].append(LOOKUP_TABLE[move])
                continue

            for c in move:
                enc_data[-1].append(LOOKUP_TABLE[c])
            
        bits_no = 5 * len(enc_data[-1])
        bytes_no = math.ceil(bits_no / 8)
        offset = 8 - bits_no % 8
        pref = bytes_no
        pref <<= 3
        pref += offset
        enc_data[-1].insert(0, pref)

    BITS = 32

    for i, data in enumerate(enc_data):
        bin_data += data[0].to_bytes(2, 'big')

        occ_bits = 0
        _bin = int(0)
        _carry = int(0)
        for b in data[1:]:
            
            _bin, _carry, occ_bits = to_binary(
                _bin, BITS, occ_bits, b, 5
            )

            if _carry >= 0:

                bin_data += _bin.to_bytes(4, 'big')
                _bin = _carry
        
        if occ_bits > 0:
            _bin <<= ((BIT_S - occ_bits)%8)
            bin_data += _bin.to_bytes(math.ceil(occ_bits / 8), 'big')

    return bin_data

def decode_naive(enc_data: bytes, return_games=False, games_objs: List=None) -> List[str]:


    reg = re.compile(MOVE_REGEX)

    REV_LOOKUP_TABLE = dict(zip(LOOKUP_TABLE.values(), LOOKUP_TABLE.keys()))

    decoded_games = []

    b_cnt = 0

    byte_it = 0
    cnt = 0
    while byte_it < len(enc_data):

        offset = 0
        k = 5

        pref = int.from_bytes(enc_data[byte_it : byte_it + 2], byteorder='big')
        bytes_in_game = pref >> 3

        byte_it += 2

        # not necessary yet
        suff_off = pref & (FOUR_1 >> (BIT_S - 3))

        notation = []
        b_cnt = 0
        while b_cnt < bytes_in_game:   

            _take = math.ceil((offset + k) / 8)
            off_r = (BIT_S - (offset + k)) % 8

            idx = extract_move_idx(int.from_bytes(enc_data[byte_it : byte_it + _take], 'big'), off_r, k)
            notation.append(REV_LOOKUP_TABLE[idx])

            if _take > 1:
                b_cnt += 1
                byte_it += 1
                offset = (offset + k) % 8
            else:
                offset += k

            if notation[-1] in set(POSSIBLE_SCORES):
                break
            
        moves = []

        notation = ''.join(notation)
        lmatched = reg.findall(notation)
        for match in lmatched:
            moves.append(match[0])

        if return_games:
            games_objs.append(chess.pgn.read_game(io.StringIO(' '.join(moves))))

        byte_it += 1

        decoded_games.append(moves)

        cnt += 1

    return decoded_games

def read_games_naive(r_buff: io.TextIOWrapper, batch_size: int, max_games: float=np.inf) -> Tuple[bytes, int]:

    enc_data = bytearray(batch_size)
    head = 0
    b_cnt = 0
    g_cnt = 0

    while b_cnt < batch_size and g_cnt < max_games:

        byts = r_buff.read(2)
        if not byts: break

        g_cnt += 1
        b_cnt += 2
        enc_data[head : head + 2] = byts
        head += 2
        bytes_no = int.from_bytes(byts, 'big') >> 3

        enc_data[head : head + bytes_no] = r_buff.read(bytes_no)
        head += bytes_no
        b_cnt += bytes_no        

    return enc_data[:head], g_cnt