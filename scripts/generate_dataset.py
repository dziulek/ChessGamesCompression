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
from chesskurcz.utilities.metrics import AvgMoveNumberInPosition, FenExtractor, UciExtractor,\
        MaxMoveNumberInPosition
from chesskurcz.utilities.stats_visitor import StatsVisitor 

DEF_OUTPUT_PATH = Path(__file__).absolute().parents[1] / 'datasets'
READ_SIZE = 32768

def decompress_zstd_chunk(input: io.StringIO, output: io.StringIO, read_size=8192, write_size=8192):

    dctx = zstandard.ZstdDecompressor()
    return dctx.copy_stream(input, output, read_size=read_size, write_size=write_size)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i')
    parser.add_argument('--max-games', '-g', type=int, default=1000)
    parser.add_argument('--output-path', '-o', default=str(DEF_OUTPUT_PATH))
    parser.add_argument('--name', '-n', default=str(time.time()))
    parser.add_argument('--representation', '-r', default='uci')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--overwrite', '-w', action='store_true')
    parser.add_argument('--labels', '-l', action='store_true', default=None)
    parser.add_argument('--collect-stats', '-c', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("chess.pgn").setLevel(logging.DEBUG)
    else:
        logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

    cwd = os.getcwd()
    input_path = os.path.join(cwd, args.input)
    buffer = io.StringIO("")

    if input_path.endswith('.zst'):
        mode = 'zst'
        input = open(input_path, 'rb')
        
    else:
        mode = 'pgn'
        input = open(input_path, 'r')

    dataset_name = f"{args.representation}-{args.name}"
    dataset_path = Path(args.output_path) / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    output_path = dataset_path / 'data.txt'
    labels_path = dataset_path / 'labels.txt'
    stats_path =  dataset_path / 'stats.txt'


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
    if args.labels is not None:
        output_labels = open(labels_path, 'w')
        StatsVisitor.add_metric(UciExtractor)

    if args.collect_stats:
        stats_output = open(stats_path, 'w')
        StatsVisitor.add_metric([AvgMoveNumberInPosition, MaxMoveNumberInPosition])

    if args.representation == 'fen':
        data_key = 'fen'
        StatsVisitor.add_metric(FenExtractor)
        
    elif args.representation == 'uci':
        data_key = 'uci'
        StatsVisitor.add_metric(UciExtractor)

    label_key = 'uci'

    cnt = 0
    line_sep = ' ' 
    sep = '\n'
    

    if args.verbose:
        printProgressBar(cnt, args.max_games)

    while cnt < args.max_games:

        result = read_game(buffer, Visitor=StatsVisitor) 
        if isinstance(result, tuple):
            game, stats = result
        else:
            game = result    
    
        if game is not None:
            cnt += 1
            if args.verbose:
                printProgressBar(cnt, args.max_games)

            data = stats[data_key]
            del stats[data_key]

            if args.labels is not None:
                labels = stats[label_key]
                output_labels.write(sep.join(labels))
                del stats[label_key]

            output.write(line_sep.join(data))
            output.write(sep)

            if args.collect_stats:
                json.dump(stats, stats_output)                

        else:
            buffer = io.StringIO(buffer.getvalue())
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
