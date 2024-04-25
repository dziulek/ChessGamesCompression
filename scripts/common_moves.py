# The purpose of the script is to figure out
# what is an average number of moves that remain
# the same between different continuous positions
# in the game.

import os
import chess
import chess.pgn
import io
from argparse import ArgumentParser
import numpy as np
from copy import deepcopy

def calculate_common_moves(bins: list, past_moves: list, moves: list):
    
    assert len(bins) == len(past_moves)
    for i in range(len(bins)):
        num_common_moves = len(set(past_moves[i]) & set(moves))
        set_sum = len(set(past_moves[i]) | set(moves))
        bins[i].append(num_common_moves / set_sum)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--path", "-p", type=str)
    parser.add_argument("--window_size", "-w", type=int, default=1)

    args = parser.parse_args()
    path = os.path.join(os.getcwd(), args.path)

    bins = [[]] * args.window_size
    past_moves = [[]] * args.window_size
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        game = chess.pgn.read_game(io.StringIO(line))
        while game is not None:
            board_copy = game.board().copy()
            moves = []
            for color in [chess.WHITE, chess.BLACK]:
                board_copy.turn = color
                moves += [str(m) for m in list(board_copy.generate_legal_moves())]
                        
            calculate_common_moves(bins, past_moves, moves)

            past_moves.append(deepcopy(moves))
            past_moves.pop(0)
            game = game.next()

    for i in range(len(bins)):
        bins[i] = np.mean(bins[i])
    
    print(bins)
    