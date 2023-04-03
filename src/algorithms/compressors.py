import numpy as np
import os 
import sys
import io
import chess
import chess.pgn
import math
from typing import List, Dict
import threading
import copy

from src.algorithms.utils import sort_moves, move_code, move_from_code, read_binary, write_binary
from src.algorithms.utils import get_script_path, write_lines, read_lines, to_binary, processLine, extract_move_idx

BATCH_SIZE = int(1e5)

SCORE_MAP = {'1-0': 0, '0-1': 1, '1/2-1/2': 2}
REV_SCORE_MAP = {0: '1-0', 1: '0-1', 2: '1/2-1/2'}

FOUR_0 = 0x00000000
FOUR_1 = 0xffffffff
BIT_S = 32

BUFF_SEM = threading.Semaphore()
sem_write = threading.Semaphore()
sem_read = threading.Semaphore()

sem_stats = threading.Semaphore()



def encode_rank(games: List[str]) -> bytes:

    enc_data_out = bytes()

    for game_notation in games:
        
        enc_data = bytes()
        pgn = io.StringIO(game_notation)
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

def decode_rank(buff: io.TextIOWrapper, batch_size: int, return_games: int=False, games_objs=None) -> List[str]:

    decoded_games = []

    enc_data = bytes()
    b_cnt = 0
    while b_cnt < batch_size:

        byts = read_binary(buff, 2)
        if not byts:
            break
        enc_data += byts
        bytes_no = int.from_bytes(byts, 'big') >> 5

        enc_data += read_binary(buff, bytes_no)
        
        b_cnt += bytes_no

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
        while b_cnt < bytes_no:
            
            moves = sort_moves(list(game.board().legal_moves))
            moves_no = len(moves)
            k = math.floor(math.log2(moves_no)) + 1

            _take = math.ceil((offset + k) / 8)
            off_r = (BIT_S - (offset + k)) % 8

            idx = extract_move_idx(int.from_bytes(enc_data[byte_it : byte_it + _take], 'big'), off_r, k)
            move = moves[idx]

            if _take > 1:
                b_cnt += 1
                byte_it += 1
                offset = (offset + k) % 8
            else:
                offset += k

            game = game.add_main_variation(move_from_code(move))

            if b_cnt + 1 == bytes_no:
                if (suff_off + offset) % 8 == 0:
                    break

        game.game().headers["Result"] = score

        if return_games:
            games_objs.append(copy.deepcopy(game.game()))

        out = str(game.game())
        byte_it += 1

        decoded_games.append(out[out.rfind('\n') + 1 : ])

    return decoded_games


def main():

    f = open(get_script_path() + '/../data/bin_test_file.txt', 'r')
    games = f.readlines()

    games = [processLine(g) for g in games]

    ref = copy.deepcopy(games)

    c = open('__tmp.bin', 'wb')
    c.write(encode_rank(games, None)) 

    f.close()
    c.close()

    c = open('__tmp.bin', 'rb')

    dec_games = decode_rank(c, BATCH_SIZE)

    assert ref == dec_games, 'Not equal'

if __name__ == "__main__":

    main()