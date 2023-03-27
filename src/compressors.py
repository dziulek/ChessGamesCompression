import numpy as np
import os 
import sys
import io
import chess
import chess.pgn
import math
from typing import List, Dict

SCORE_MAP = {'1-0': 0x00, '0-1': 0x01, '1/2-1/2': 0x2}

def encode(buff: io.TextIOWrapper, game_notation: List[str]) -> None:

    pgn = io.StringIO(game_notation)
    game: chess.pgn.ChildNode = chess.pgn.read_game(pgn)

    while game.is_end():

        number_of_moves = game.board().legal_moves.count()
        k = math.floor(math.log2(number_of_moves))

        moves = [move.uci() for move in list(game.board().legal_moves)].sort()

        move_no = moves.index(game.next().uci())

        rep = (0xff & move_no) 

        game = game.next()

    game_notation


def decode(buff: io.TextIOWrapper) -> str:

    pass



def main():

    pass

if __name__ == "__main__":

    main()