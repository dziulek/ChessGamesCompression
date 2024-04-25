import chess
import chess.pgn


import numpy as np
import io

from typing import List, Dict, Tuple
from chesskurcz.algorithms.util.utils import get_all_possible_moves

import os
import copy

import re

BATCH_SIZE = int(1e4)

ALL_POSSIBLE_MOVES = dict(zip(get_all_possible_moves(16), [i for i in range(len(get_all_possible_moves(16)))]))
REV_ALL_POSSIBLE_MOVES = dict(
    zip(ALL_POSSIBLE_MOVES.values(), ALL_POSSIBLE_MOVES.keys())
)

def move_transform(moves: List) -> str:
    
    return [re.sub(re.compile(r'\+|\#'), '', move) for move in moves]

def encode_apm(games: List[List[str]]) -> bytes:

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

    enc_data = bytearray(2 * batch_size)
    head = 0
    b_cnt = 0
    g_cnt = 0
    while b_cnt < batch_size and g_cnt < max_games:

        b = r_buff.read(2)
        if not b: break
        g_cnt += 1
        b_cnt += 2

        enc_data[head : head + 2] = b
        head += 2
        bytes_in_game = int.from_bytes(b, 'big')

        enc_data[head : head + bytes_in_game] = r_buff.read(bytes_in_game)
        head += bytes_in_game
        b_cnt += bytes_in_game

    return enc_data[:head], g_cnt