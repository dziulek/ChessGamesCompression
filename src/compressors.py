import numpy as np
import os 
import sys
import io
import chess
import chess.pgn
import math
from typing import List, Dict
import threading

SCORE_MAP = {'1-0': 0, '0-1': 1, '1/2-1/2': 2}
REV_SCORE_MAP = {0: '1-0', 1: '0-1', 2: '1/2-1/2'}

FOUR_0 = 0x00000000
FOUR_1 = 0xffffffff
BIT_S = 32

BUFF_SEM = threading.Semaphore()

def encode(buff: io.TextIOWrapper, games: List[str], sem: threading.Semaphore) -> None:

    enc_data = []


    for game_notation in games:

        pgn = io.StringIO(game_notation)
        game: chess.pgn.ChildNode = chess.pgn.read_game(pgn)
        bits = 0
        _bin = int(0)
        _carry = int(0)

        score = SCORE_MAP[game.headers.get('Result')]

        move_cnt = int(0)
        while game is not None:

            
            number_of_moves = game.board().legal_moves.count()
            k = math.floor(math.log2(number_of_moves)) + 1

            moves: List[str] = [move.uci() for move in list(game.board().legal_moves)]
            moves.sort()
            game = game.next()
            if game is None:
                break
            move_no = moves.index(game.move.uci())  
            move_cnt += 1

            if bits + k > BIT_S:
                _bin <<= (BIT_S - bits)
                bits = (bits + k) % BIT_S
                _carry = (FOUR_1 >> (BIT_S - bits) & move_no)
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

    move_cnt <<= 2
    move_cnt += score
    buff.write(move_cnt.to_bytes(2, byteorder='big'))
    for _b in enc_data:
        buff.write(_b.to_bytes(4, byteorder='big'))
    
    sem.release()
    
    print("success")


def decode(buff: io.TextIOWrapper) -> str:

    game = chess.pgn.Game()
    bits = BIT_S
    _carry = int(0)
    _bin = int.from_bytes(buff.read(2), 'big')

    score = REV_SCORE_MAP[_bin & (FOUR_1 >> BIT_S - 2)]
    moves_no = _bin >> 2

    _bin = int.from_bytes(buff.read(4), 'big')

    for i in range(moves_no):
        
        moves = [move.uci() for move in  list(game.board().legal_moves)]
        moves.sort()
        moves_no = len(moves)
        k = math.floor(math.log2(moves_no)) + 1

        if bits < k:
            _carry = _bin >> (BIT_S - bits)

            _bin = int.from_bytes(buff.read(4), 'big')
            move = moves[_carry<<(k - bits) + _bin>>(BIT_S - (k - bits))]
            _bin &= (FOUR_1 >> (k - bits))
            _bin <<= (k - bits)
            bits = BIT_S - (k - bits)

        else:  
            move = moves[_bin >> (BIT_S - k)]
            _bin &= (FOUR_1 >> (k))
            _bin <<= (k)
            bits -= k

        game = game.add_main_variation(chess.Move.from_uci(move))

    game.headers["Result"] = score
    return str(game)




def main():

    f = open('encode_test.bin', 'wb')

    with open('/home/czewian/Studia/ChessGamesCompression/data/bin_test_file.txt', 'r') as d:
        game_notation = d.readline()

        encode(f, [game_notation], BUFF_SEM)
        f.close
        f = open('encode_test.bin', 'rb')
        decoded = decode(f)
        
        assert decoded == game_notation

    f.close()

if __name__ == "__main__":

    main()