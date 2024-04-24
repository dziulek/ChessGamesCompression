import os, sys, io
import time
from pathlib import Path
import argparse
import uuid
from enum import Enum
from chess.pgn import read_game

DEF_OUTPUT_PATH = Path(__file__).absolute().parent[1] / 'datasets'


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i')
    parser.add_argument('--max-games', '-g', type=int, default=1000)
    parser.add_argument('--output-path', '-o', default=str(DEF_OUTPUT_PATH))
    parser.add_argument('--name', '-n', default=str(time.time()))
    parser.add_argument('--representation', '-r', default='raw')

    args = parser.parse_args()

    cwd = os.getcwd()
    input_path = os.path.join(cwd, args.input)

    mode = 'pgn'
    if input_path.endswith('.zst'):
        mode = 'zst'
    
    if mode == 'pgn':
        with open(input_path, 'r') as f:

            cnt = 0
            while cnt < args.max_games:
                game = read_game(f) 
                


    elif mode == 'zst':
        pass
    else:
        pass
