import threading
import os, io, sys
import math
import chess
import chess.pgn
import copy

from typing import List, Dict, Tuple, Callable
from src.algorithms.compressors import REV_SCORE_MAP, SCORE_MAP
from src.algorithms.utils import to_binary, read_binary, extract_move_idx, sort_moves, move_from_code
from src.algorithms.utils import processLine, get_script_path, MOVE_REGEX, POSSIBLE_SCORES

import re

BATCH_SIZE = int(1e4)

FOUR_0 = 0x00000000
FOUR_1 = 0xffffffff
BIT_S = 32

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

def encode_naive(games: List[str]) -> bytes:

    enc_data: List[List[int]] = []
    bin_data = bytes()

    for game in games:
        
        enc_data.append([])
        moves = game.split(' ')
                
        for move in moves:

            move = move.strip()
            
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

    for data in enc_data:
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

def decode_naive(buff: io.TextIOWrapper, batch_s: int, return_games=False, games_objs: List=None) -> List[str]:


    reg = re.compile(MOVE_REGEX)

    REV_LOOKUP_TABLE = dict(zip(LOOKUP_TABLE.values(), LOOKUP_TABLE.keys()))

    decoded_games = []

    enc_data = bytes()
    b_cnt = 0
    while b_cnt < batch_s:

        byts = read_binary(buff, 2)
        if not byts:
            break
        enc_data += byts
        bytes_no = int.from_bytes(byts, 'big') >> 3

        enc_data += read_binary(buff, bytes_no)
        
        b_cnt += bytes_no

    byte_it = 0
    while byte_it < len(enc_data):

        _bin = int.from_bytes(enc_data[byte_it : byte_it + 2], 'big')
        byte_it += 2

        bytes_no = _bin >> 3
        suff_off = _bin & (FOUR_1 >> (BIT_S - 3))
        offset = 0
        k = 5

        notation = []
        b_cnt = 0
        while b_cnt < bytes_no:   
            
            _take = math.ceil((offset + k) / 8)
            off_r = (BIT_S - (offset + k)) % 8

            idx = extract_move_idx(int.from_bytes(enc_data[byte_it : byte_it + _take], 'big'), off_r, k)
            # print(notation)
            notation.append(REV_LOOKUP_TABLE[idx])

            if _take > 1:
                b_cnt += 1
                byte_it += 1
                offset = (offset + k) % 8
            else:
                offset += k


            if notation[-1] == '0-1' or notation[-1] == '1-0' or notation[-1] == '1/2-1/2':
                break

        game_str = ''

        notation = ''.join(notation)
        lmatched = reg.findall(notation)
        for match in lmatched:
            game_str += match[0] + ' '
        print(game_str)

        pgn = io.StringIO(game_str)

        game = chess.pgn.read_game(pgn)

        if return_games:
            games_objs.append(game.game())

        out = str(game.game())
        byte_it += 1

        decoded_games.append(out[out.rfind('\n') + 1 : ])

        # pgn.close()

    return decoded_games

def main():

    f = open(get_script_path() + '/../data/bin_test_file.txt', 'r')
    games = f.readlines()

    games = [processLine(g) for g in games]

    ref = copy.deepcopy(games)

    c = open('__tmp.bin', 'wb')
    c.write(encode_naive(games, None)) 

    f.close()
    c.close()

    c = open('__tmp.bin', 'rb')

    dec_games = decode_naive(c, BATCH_SIZE)

    assert ref == dec_games, 'Not equal'

    c.close()

    os.remove('__tmp.bin')

if __name__ == "__main__":

    main()