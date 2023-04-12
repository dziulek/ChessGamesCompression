import chess
import chess.pgn


import numpy as np
import io

from typing import List, Dict, Tuple
from src.algorithms.utils import get_all_possible_moves, POSSIBLE_SCORES, read_binary

import os
import copy

from src.algorithms.utils import processLine, get_script_path

import re

MOVE_REGEX = r'(O-O-O|O-O|[QKRBN]?([a-h]|[1-8])?x?[a-h][1-8]([#+]|=[QRBN][+#]?)?|1/2-1/2|1-0|0-1)'
move_token_reg = re.compile(MOVE_REGEX)

THRASH_REGEX = r'(\?|\!|\{.*\})'
thrash_token_reg = re.compile(THRASH_REGEX)

BATCH_SIZE = int(1e4)

ALL_POSSIBLE_MOVES = dict(zip(get_all_possible_moves(16), [i for i in range(len(get_all_possible_moves(16)))]))
REV_ALL_POSSIBLE_MOVES = dict(
    zip(ALL_POSSIBLE_MOVES.values(), ALL_POSSIBLE_MOVES.keys())
)

def move_transform(moves: List) -> str:
    
    return [re.sub(re.compile(r'\+|\#'), '', move) for move in moves]

def encode_apm(games: List[List[str]]) -> bytes():

    enc_data = bytes()


    for game in games:

        enc_game = bytes()

        if len(game) == 0:
            continue

        for move in game:
            enc_game += int.to_bytes(ALL_POSSIBLE_MOVES[move], 2, 'big')

        bytes_no = len(enc_game)

        enc_game = int.to_bytes(bytes_no, 2, 'big') + enc_game

        enc_data += enc_game

    return enc_data


def decode_apm(data: bytes, return_games=False, games_objs: List=None) -> List[List[str]]:
        
    i = 0
    byte_no = len(data)
    output = []
    while i < byte_no:

        bytes_in_game = int.from_bytes(data[i:i + 2], 'big')
        i += 2
        start = i
        moves = []
        while i - start < bytes_in_game:

            moves.append(REV_ALL_POSSIBLE_MOVES[int.from_bytes(data[i : i + 2], 'big')])
            i += 2

        if return_games:

            game = chess.pgn.read_game(io.StringIO(' '.join(moves)))
            games_objs.append(copy.deepcopy(game.game()))

        output.append(moves)

    return output

def read_games_apm(r_buff: io.TextIOWrapper, batch_size: int, max_games: float=np.inf) -> Tuple[bytes, int]:

    enc_data = bytes()
    b_cnt = 0
    g_cnt = 0
    while b_cnt < batch_size and g_cnt < max_games:

        b = r_buff.read(2)
        if not b: break
        g_cnt += 1
        b_cnt += 2

        enc_data += b
        bytes_in_game = int.from_bytes(b, 'big')

        enc_data += r_buff.read(bytes_in_game)
        b_cnt += bytes_in_game

    return enc_data, g_cnt

def main():

    f = open(get_script_path() + '/../data/bin_test_file.txt', 'r')
    games = f.readlines()

    games = [processLine(g, regex_drop=thrash_token_reg, 
                         regex_take=move_token_reg, token_transform=move_transform) for g in games]

    ref = copy.deepcopy(games)

    c = open('__tmp.bin', 'wb')
    c.write(encode_apm(games)) 

    f.close()
    c.close()

    c = open('__tmp.bin', 'rb')

    dec_games = decode_apm(c, BATCH_SIZE)

    assert ref == dec_games, 'Not equal'

    c.close()

    os.remove('__tmp.bin')

if __name__ == "__main__":

    main()