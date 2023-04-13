import numpy as np
import os 
import sys
import io
import chess
import chess.pgn
import math
from typing import List, Dict, Tuple
import threading
import copy

from src.algorithms.utils import sort_moves, move_code, move_from_code, FOUR_0, FOUR_1, BIT_S
from src.algorithms.utils import get_script_path, to_binary, processLine, extract_move_idx

BATCH_SIZE = int(1e5)

SCORE_MAP = {'1-0': 0, '0-1': 1, '1/2-1/2': 2}
REV_SCORE_MAP = {0: '1-0', 1: '0-1', 2: '1/2-1/2'}

def encode_rank(games: List[List[str]]) -> bytes:

    enc_data_out = bytes()

    for game_notation in games:
        
        enc_data = bytes()
        pgn = io.StringIO(' '.join(game_notation))
        game: chess.pgn.ChildNode = chess.pgn.read_game(pgn)

        bits = 0
        _bin = int(0)
        _carry = int(0)

        score = SCORE_MAP[game.headers.get('Result')]

        len_bef = len(enc_data)
        while game is not None:

            board = game.board()
            
            number_of_moves = board.legal_moves.count()
            if number_of_moves == 0:
                break
            k = math.floor(math.log2(number_of_moves)) + 1

            moves = sort_moves(list(board.legal_moves))
            
            game = game.next()
            if game is None:
                break
            move_no = moves.index(move_code(game.move))  

            _bin, _carry, bits = to_binary(
                _bin, BIT_S, bits, move_no, k
            )
            if _carry >= 0:
                enc_data += _bin.to_bytes(4, 'big')
                _bin = _carry

        if bits > 0:
            _bin <<= (BIT_S - bits) % 8
            enc_data += _bin.to_bytes(math.ceil(bits / 8), 'big')

        pref = len(enc_data) - len_bef
        pref <<= 3
        pref += (BIT_S - bits) % 8
        pref <<= 2
        pref += score
        enc_data_out += (pref.to_bytes(2, 'big') + enc_data)

    return enc_data_out

def decode_rank(enc_data: bytes, return_games: int=False, games_objs=None) -> List[List[str]]:

    decoded_games = []

    byte_it = 0
    while byte_it < len(enc_data):

        game = chess.pgn.Game()
        _bin = int.from_bytes(enc_data[byte_it : byte_it + 2], 'big')
        byte_it += 2

        score = REV_SCORE_MAP[_bin & (FOUR_1 >> BIT_S - 2)]
        suff_off = (_bin >> 2) & (FOUR_1 >> BIT_S - 3)
        bytes_no = _bin >> 5
        offset = 0

        b_cnt = 0

        g_moves = []
        while b_cnt < bytes_no:
            
            moves = sort_moves(list(game.board().legal_moves))
            moves_no = len(moves)
            k = math.floor(math.log2(moves_no)) + 1

            _take = math.ceil((offset + k) / 8)
            off_r = (BIT_S - (offset + k)) % 8

            idx = extract_move_idx(int.from_bytes(enc_data[byte_it : byte_it + _take], 'big'), off_r, k)
            move = moves[idx]

            # g_moves.append(move)

            if _take > 1:
                b_cnt += 1
                byte_it += 1
                offset = (offset + k) % 8
            else:
                offset += k

            game = game.add_main_variation(move_from_code(move))
            g_moves.append(game.move.uci())

            if b_cnt + 1 == bytes_no:
                if (suff_off + offset) % 8 == 0:
                    break

        game.game().headers["Result"] = score

        if return_games:
            games_objs.append(copy.deepcopy(game.game()))

        byte_it += 1

        decoded_games.append(g_moves + [score])

    return decoded_games

def read_games_rank(r_buff: io.TextIOWrapper, batch_size: int, max_games: float=np.inf) -> Tuple[bytes, int]:

    enc_data = bytes()
    b_cnt = 0
    g_cnt = 0

    while b_cnt < batch_size and g_cnt < max_games:

        byts = r_buff.read(2)
        if not byts: break

        g_cnt += 1
        b_cnt += 2
        enc_data += byts
        bytes_no = int.from_bytes(byts, 'big') >> 5

        enc_data += r_buff.read(bytes_no)
        
        b_cnt += bytes_no        

    return enc_data, g_cnt