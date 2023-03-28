import numpy as np
import os 
import sys
import io
import chess
import chess.pgn
import math
from typing import List, Dict
import threading

from src.stats import Stats
from src.utils import sort_moves, move_code, move_from_code, read_binary, write_binary, write_lines, read_lines

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

def encode_rank(buff: io.TextIOWrapper, games: List[str], stats: Stats=None) -> bytes:

    enc_data = bytes()

    for game_notation in games:

        pgn = io.StringIO(game_notation)
        game: chess.pgn.ChildNode = chess.pgn.read_game(pgn)

        if stats is not None:
            stats.add_game(game)

        bits = 0
        _bin = int(0)
        _carry = int(0)

        score = SCORE_MAP[game.headers.get('Result')]

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

            if stats is not None:
                stats.add_move(game.move, board)

            if bits + k > BIT_S:
                _bin <<= (BIT_S - bits)
                bits = (bits + k) % BIT_S
                _carry = (FOUR_1 >> (BIT_S - bits)) & move_no
                move_no >>= (bits)

                _bin |= move_no

                enc_data += _bin.to_bytes(4, 'big')
                _bin = _carry
            else:
                _bin <<= k
                bits += k

                _bin |= move_no 

        _bin <<= (BIT_S - bits)
        enc_data += _bin.to_bytes(4, 'big')
    

    pref = len(enc_data)
    pref <<= 2
    pref += score

    return pref.to_bytes(2, 'big') + enc_data

def decode_rank(buff: io.TextIOWrapper, batch_size: int) -> List[str]:

    decoded_games = []

    enc_data = bytes
    b_cnt = 0
    while b_cnt < batch_size:

        byts = read_binary(buff, 2)
        enc_data += byts
        bytes_no = byts >> 2

        enc_data += read_binary(buff, bytes_no)
        
        b_cnt += bytes_no

    it = 0
    while it < len(enc_data):

        game = chess.pgn.Game()
        bits = BIT_S
        _carry = int(0)
        _bin = enc_data[it]
        it+=1

        score = REV_SCORE_MAP[_bin & (FOUR_1 >> BIT_S - 2)]
        bytes_no = _bin >> 2

        _bin = enc_data[it]
        it+=1
        b_cnt = 6

        while b_cnt < bytes_no:
            
            moves = sort_moves(list(game.board().legal_moves))
            moves_no = len(moves)
            k = math.floor(math.log2(moves_no)) + 1

            if bits < k:
                _carry = _bin >> (BIT_S - bits)

                _bin = enc_data[it]
                it+=1
                b_cnt += 4
                _tmp = _bin>>(BIT_S - (k - bits))
                __tmp = _carry<<(k - bits)
                move = moves[__tmp + _tmp]
                _bin &= (FOUR_1 >> (k - bits))
                _bin <<= (k - bits)
                bits = BIT_S - (k - bits)

            else:  
                move = moves[_bin >> (BIT_S - k)]
                _bin &= (FOUR_1 >> (k))
                _bin <<= (k)
                bits -= k

            game = game.add_main_variation(move_from_code(move))

        game.game().headers["Result"] = score
        out = str(game.game())

        decoded_games.append(out[out.rfind('\n') + 1 : ])

    return decoded_games

def process_encode(w_buff: io.TextIOWrapper, games: List[str], batch_size: int, sem: threading.Semaphore,
        algorithm='rank', stats: Stats=None) -> None:

    encode_rank(w_buff, games, sem, stats)
    
def process_decode(r_buff: io.TextIOWrapper, batch_size: int,
        algorithm='rank') -> List[str]:

    global sem_write

    games = decode_rank(r_buff, sem_write, BATCH_SIZE)

    return games

def compressed_file_stats(path: str, total_moves: int, total_games: int, origin_size: int):

    pass



def main():

    pass

    

if __name__ == "__main__":

    main()