import chess
import chess.pgn

import numpy
import io

from typing import List, Dict, Tuple
from src.utils import get_all_possible_moves, POSSIBLE_SCORES, read_binary

import os
import copy

from src.utils import processLine, get_script_path

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

def apm_encode(games: List[str]):

    enc_data = bytes()

    for game in games:

        enc_game = bytes()
        moves = game.split(' ')

        for move in moves:
            
            if move.find('.') >= 0:
                continue
            enc_game += int.to_bytes(ALL_POSSIBLE_MOVES[move], 2, 'big')

        bytes_no = len(enc_game)

        enc_game = int.to_bytes(bytes_no, 2, 'big') + enc_game

        enc_data += enc_game

    return enc_data


def apm_decode(buff: io.TextIOWrapper, batch_s: int, return_games=False, games_objs: List=None) -> List[str]:

    b_cnt = 0
    enc_data = bytes()
    game_sizes = []
    dec_data_list = []
    while b_cnt < batch_s:
        
        pref = read_binary(buff, 2)
        if not pref:
            break
        bytes_in_game = int.from_bytes(pref, 'big')
        game_sizes.append(bytes_in_game)
        enc_data += read_binary(buff, bytes_in_game)

        b_cnt += bytes_in_game + 2
        

    i = 0
    for g in game_sizes:

        dec_data_list.append([])

        start = i
        while i - start < g:

            dec_data_list[-1].append(REV_ALL_POSSIBLE_MOVES[int.from_bytes(enc_data[i : i + 2], 'big')])
            i += 2

        if return_games:

            game = chess.pgn.read_game(io.StringIO(' '.join(dec_data_list[-1])))
            games_objs.append(copy.deepcopy(game.game()))

        dec_data_list[-1] = ' '.join(dec_data_list[-1])

    return dec_data_list

def main():

    f = open(get_script_path() + '/../data/bin_test_file.txt', 'r')
    games = f.readlines()

    games = [processLine(g, regex_drop=thrash_token_reg, 
                         regex_take=move_token_reg, token_transform=move_transform) for g in games]

    ref = copy.deepcopy(games)

    c = open('__tmp.bin', 'wb')
    c.write(apm_encode(games)) 

    f.close()
    c.close()

    c = open('__tmp.bin', 'rb')

    dec_games = apm_decode(c, BATCH_SIZE)

    assert ref == dec_games, 'Not equal'

    c.close()

    os.remove('__tmp.bin')

if __name__ == "__main__":

    main()