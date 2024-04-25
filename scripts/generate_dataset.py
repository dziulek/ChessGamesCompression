import os, sys, io
import json
from copy import deepcopy
import time
from pathlib import Path
import argparse
import uuid
from enum import Enum
from chess.pgn import read_game
import zstandard
import chess
import logging
from chesskurcz.logger import printProgressBar
from chesskurcz.stats import Stats

DEF_OUTPUT_PATH = Path(__file__).absolute().parents[1] / 'datasets'
READ_SIZE = 32768

def decompress_zstd_chunk(input: io.StringIO, output: io.StringIO, read_size=8192, write_size=8192):

    dctx = zstandard.ZstdDecompressor()
    return dctx.copy_stream(input, output, read_size=read_size, write_size=write_size)

def game_to_uci(game: chess.pgn.Game, copy=False, sep=' ') -> str:

    ucis = []
    if copy:
        _game = deepcopy(game)
    else: _game = game

    _game = _game.next()
    while _game is not None:

        ucis.append(str(_game.move))
        _game = _game.next()

    return sep.join(ucis)

def extract_fens(game: chess.pgn.Game, copy=False, sep=',') -> str:

    fens = []
    if copy:
        _game = deepcopy(game)
    else: _game = game

    _game = _game.next()
    while _game is not None:

        fens.append(_game.board().fen())
        _game = _game.next()

    return sep.join(fens)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i')
    parser.add_argument('--max-games', '-g', type=int, default=1000)
    parser.add_argument('--output-path', '-o', default=str(DEF_OUTPUT_PATH))
    parser.add_argument('--name', '-n', default=str(time.time()))
    parser.add_argument('--representation', '-r', default='uci')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--overwrite', '-w', action='store_true')
    parser.add_argument('--labels', '-l', type=str, default=None)
    parser.add_argument('--collect-stats', '-c', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("chess.pgn").setLevel(logging.DEBUG)
    else:
        logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

    cwd = os.getcwd()
    input_path = os.path.join(cwd, args.input)
    buffer = io.StringIO()

    if input_path.endswith('.zst'):
        mode = 'zst'
        input = open(input_path, 'rb')
        
    else:
        mode = 'pgn'
        input = open(input_path, 'r')

    dataset_name = f"{args.representation}-{args.name}"
    output_path = Path(args.output_path) / dataset_name / 'data.txt'
    dataset_params_path = Path(args.output_path) / dataset_name / 'params.json'
    with open(dataset_params_path, 'w') as f:
        json.dump({
            'representation': args.representation,
            'labels': args.labels,
            
        }, f)
    if Path.exists(output_path) and not args.overwrite:
        print('Destination file exists, if you want to overwrite specify --overwrite/-w')
        exit(1)
    
    output = open(output_path, 'w')

    cnt = 0
    buffer_len = 0
    sep = '\n'
    if args.verbose:
        printProgressBar(cnt, args.max_games)

    while cnt < args.max_games:
        
        try:
            game = read_game(buffer) 
        except Exception as e:
            if args.verbose:
                print(e)     
            continue

        if game is not None and not len(game.errors):
            cnt += 1
            if args.verbose:
                printProgressBar(cnt, args.max_games)

            # transform
            if args.representation == 'uci':
                data = game_to_uci(game)
            elif args.representation == 'fen':
                data = extract_fens(game, sep='\n')

            output.write(data)
            output.write(sep)

        else:
            buffer = io.StringIO(buffer.read())
            if mode == 'zst':
                _, bytes_written = decompress_zstd_chunk(input, buffer)
            else:
                bytes_written = buffer.write(input.read(READ_SIZE))

            buffer.seek(0)
            if not bytes_written:
                break
        
    buffer.close()
    input.close()
    output.close()


if __name__ == "__main__":
    main()
