import numpy as np
import os 
import sys
import io
import chess
import chess.pgn
import math
from typing import List, Dict

SCORE_MAP = {'1-0': 0x00, '0-1': 0x01, '1/2-1/2': 0x2}

FOUR_0 = 0x00000000
FOUR_1 = 0xffffffff
BIT_S = 32

def encode(buff: io.TextIOWrapper, game_notation: str) -> None:

    pgn = io.StringIO(game_notation)
    game: chess.pgn.ChildNode = chess.pgn.read_game(pgn)
    bits = 0
    _bin = int(0)
    _carry = int(0)

    while game is not None:

        number_of_moves = game.board().legal_moves.count()
        k = math.floor(math.log2(number_of_moves)) + 1

        moves: List[str] = [move.uci() for move in list(game.board().legal_moves)]
        moves.sort()
        game = game.next()
        if game is None:
            break
        move_no = moves.index(game.move.uci())  

        if bits + k > BIT_S:
            _bin <<= (BIT_S - bits)
            bits = (bits + k) % BIT_S
            _carry = (FOUR_1 >> (BIT_S - bits) & move_no)
            move_no >>= (bits)

            _bin |= move_no

            buff.write(_bin.to_bytes(4, byteorder='big'))
            _bin = _carry
        else:
            _bin <<= k
            bits += k

            _bin |= move_no 

        
    print("success")


def decode(buff: io.TextIOWrapper) -> str:

    game = chess.pgn.Game()
    bits = BIT_S
    _carry = int(0)
    r = buff.read(4)
    _bin = int.from_bytes(r, 'big')

    while buff is not None:

        
        moves = [move.uci() for move in  list(game.board().legal_moves)]
        moves.sort()
        moves_no = len(moves)
        k = math.floor(math.log2(moves_no)) + 1

        if bits < k:
            _carry >>= (BIT_S - bits)

            _bin = buff.read(4)
            move = moves[_carry<<(k - bits) + _bin>>(BIT_S - (k - bits))]
            _bin <<= (k - bits)
            bits = BIT_S - (k - bits)

        else:  
            move = moves[_bin >> BIT_S - k]
            _bin <<= k
            bits -= k

        game = game.add_main_variation(chess.Move.from_uci(move))
        game = game.next()

    return str(game)




def main():

    f = open('encode_test.bin', 'wb')

    with open('/home/czewian/Studia/ChessGamesCompression/data/bin_test_file.txt', 'r') as d:
        game_notation = d.readline()

        encode(f, game_notation)
        f.close
        f = open('encode_test.bin', 'rb')
        decoded = decode(f)
        
        assert decoded == game_notation

    f.close()

if __name__ == "__main__":

    main()