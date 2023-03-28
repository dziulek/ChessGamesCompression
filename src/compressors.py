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
from src.utils import readLines, filterLines, writeLines

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

def encode(buff: io.TextIOWrapper, games: List[str], sem: threading.Semaphore, stats: Stats=None) -> None:

    enc_data = []

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

            moves: List[str] = [move.uci() for move in list(board.legal_moves)]
            moves.sort()
            game = game.next()
            if game is None:
                break
            move_no = moves.index(game.move.uci())  

            if stats is not None:
                stats.add_move(game.move, board)

            if bits + k > BIT_S:
                _bin <<= (BIT_S - bits)
                bits = (bits + k) % BIT_S
                _carry = (FOUR_1 >> (BIT_S - bits)) & move_no
                move_no >>= (bits)

                _bin |= move_no

                enc_data.append(_bin)
                _bin = _carry
            else:
                _bin <<= k
                bits += k

                _bin |= move_no 

        _bin <<= (BIT_S - bits)
        enc_data.append(_bin)
    
    sem.acquire()
    pref = 4 * len(enc_data) + 2
    pref <<= 2
    pref += score
    buff.write(pref.to_bytes(2, byteorder='big'))
    for _b in enc_data:
        buff.write(_b.to_bytes(4, byteorder='big'))
    
    sem.release()

def decode(buff: io.TextIOWrapper, sem: threading.Semaphore, batch_size: int) -> List[str]:

    decoded_games = []

    sem.acquire()

    enc_data = []
    b_cnt = 0
    while b_cnt < batch_size:

        enc_data.append(int.from_bytes(buff.read(2), 'big'))
        bytes_no = enc_data[-1]>>2
        
        for _ in range(bytes_no):
            enc_data.append(int.from_bytes(buff.read(4), 'big'))
        
        b_cnt += bytes_no

    sem.release()

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
            
            moves = [move.uci() for move in  list(game.board().legal_moves)]
            moves.sort()
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

            game = game.add_main_variation(chess.Move.from_uci(move))

        game.game().headers["Result"] = score
        out = str(game.game())

        decoded_games.append(out[out.rfind('\n') + 1 : ])

    return decoded_games

def process_encode(readBuff: io.TextIOWrapper, writeBuff: io.TextIOWrapper, batch_size: int,
        algorithm='rank', stats: Stats=None) -> None:

    global sem_write, sem_read

    lines = readLines(readBuff, BATCH_SIZE, sem_read)

    while lines is not None and len(lines) > 0:
        
        lines = filterLines(lines)

        encode(writeBuff, lines, sem_write, stats)

        lines = readLines(readBuff, BATCH_SIZE, sem_read)
    
def process_decode(readBuff: io.TextIOWrapper, batch_size: int,
        algorithm='rank') -> List[str]:

    global sem_write

    games = decode(readBuff, sem_write, BATCH_SIZE)

def compressed_file_stats(path: str, total_moves: int, total_games: int, origin_size: int):

    pass



def main():

    source_file = '/home/czewian/Studia/ChessGamesCompression/data/test_file.txt'
    dest_file = '/home/czewian/Studia/ChessGamesCompression/data/encoded_test_file.bin'

    stats = Stats(source_file, dest_file, sem_stats)

    read_buff = open(source_file, 'r')
    write_buff = open(dest_file, 'wb')

    threads: List[threading.Thread] = []
    
    for i in range(1):
        threads.append(threading.Thread(target=process_encode, args=(read_buff, write_buff, BATCH_SIZE, stats)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    read_buff.close()
    write_buff.close()

    

if __name__ == "__main__":

    main()