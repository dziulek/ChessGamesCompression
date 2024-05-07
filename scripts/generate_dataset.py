import os, sys, io
import json
from copy import deepcopy
import pandas as pd
import numpy as np
import time
from pathlib import Path
import argparse
import uuid
from enum import Enum
from chess.pgn import read_game
import zstandard
import chess
import logging
from multiprocessing import Queue

from chesskurcz.logger import printProgressBar
from chesskurcz.utilities.metrics import AvgMoveNumberInPosition, FenExtractor, UciExtractor,\
        MaxMoveNumberInPosition, PieceTypeProbability, EmptySquaresNumMoves, GameLen
from chesskurcz.utilities.stats_visitor import StatsVisitor 

DEF_OUTPUT_PATH = Path(__file__).absolute().parents[1] / 'datasets'
READ_SIZE = 32768

def decompress_zstd_chunk(input: io.StringIO, output: io.StringIO, read_size=8192, write_size=8192):

    dctx = zstandard.ZstdDecompressor()
    return dctx.copy_stream(input, output, read_size=read_size, write_size=write_size)

def flatten_df(df: pd.DataFrame, remove_old_columns=True) -> pd.DataFrame:

    dict_columns = []
    for col in df.columns:
        if isinstance(df[col].iloc[0], dict):
            dict_columns.append(col)
            df = df.assign(**{f"{col}_{k}": v for k, v in df[col].apply(pd.Series).items()})

    if remove_old_columns:
        df.drop(dict_columns, axis=1, inplace=True)

    return df

def aggregate_statistics(per_game_df: pd.DataFrame, func='mean') -> pd.DataFrame:

    
    tmp_df = flatten_df(per_game_df)
    columns_to_aggregate = [colname for colname, dtype in dict(tmp_df.dtypes).items() \
                            if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_)]

    tmp_df = tmp_df[columns_to_aggregate] 
    tmp_df = tmp_df.astype(float)

    tmp_df = tmp_df.aggregate(func)
    return tmp_df
        

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
    parser.add_argument('--collect-stats', '-c', type=str, default=None)

    args = parser.parse_args()

    if args.collect_stats is not None:
        assert(args.collect_stats in ('general', 'granular'))

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

    dataset_name = f"{args.representation}-{args.name}-{args.max_games}"
    dataset_path = Path(args.output_path) / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    output_path = dataset_path / 'data.txt'
    labels_path = dataset_path / 'labels.txt'
    global_stats_path = dataset_path / 'global_stats.json'
    stats_path =  dataset_path / 'stats_per_game.json'

    line_sep = ';' 
    sep = '\n'

    dataset_params_path = Path(args.output_path) / dataset_name / 'params.json'
    with open(dataset_params_path, 'w') as f:
        json.dump({
            'representation': args.representation,
            'labels': args.labels,
            'line_sep': line_sep 
        }, f)
    if Path.exists(output_path) and not args.overwrite:
        print('Destination file exists, if you want to overwrite specify --overwrite/-w')
        exit(1)
    
    output = open(output_path, 'w')
    if args.labels is not None:
        output_labels = open(labels_path, 'w')
        StatsVisitor.add_metric(UciExtractor)

    if args.collect_stats:
        global_stats_output = open(global_stats_path, 'w')
        if args.collect_stats == 'granular':
            stats_output = open(stats_path, 'w')
        StatsVisitor.add_metric([
            AvgMoveNumberInPosition, MaxMoveNumberInPosition, 
            PieceTypeProbability, EmptySquaresNumMoves, GameLen
        ])

    if args.representation == 'fen':
        data_key = 'fen'
        StatsVisitor.add_metric(FenExtractor)
        
    elif args.representation == 'uci':
        data_key = 'uci'
        StatsVisitor.add_metric(UciExtractor)

    label_key = 'uci'

    per_game_stats = []
    cnt = 0
    

    printProgressBar(cnt, args.max_games)

    while cnt < args.max_games:

        result = read_game(buffer, Visitor=StatsVisitor) 
        if isinstance(result, tuple):
            game, stats = result
        else:
            game = result    
    
        if game is not None:
            cnt += 1
            printProgressBar(cnt, args.max_games)

            data = stats[data_key]
            del stats[data_key]

            if args.labels is not None:
                labels = stats[label_key]
                output_labels.write(line_sep.join(labels))
                output_labels.write(sep)
                del stats[label_key]

            output.write(line_sep.join(data))
            output.write(sep)

            if args.collect_stats:
                per_game_stats.append(stats)

        else:
            buffer = io.StringIO(buffer.getvalue())
            if mode == 'zst':
                _, bytes_written = decompress_zstd_chunk(input, buffer)
            else:
                bytes_written = buffer.write(input.read(READ_SIZE))

            buffer.seek(0)
            if not bytes_written:
                break
    
    if args.collect_stats:
        if args.collect_stats == 'granular':
            json.dump(per_game_stats, stats_output)                

        global_stats = pd.DataFrame(per_game_stats)
        global_stats = aggregate_statistics(global_stats, func='mean')
        global_stats.to_json(global_stats_output)

        
    buffer.close()
    input.close()
    output.close()
    if args.labels is not None:
        output_labels.close()
    if args.collect_stats:
        global_stats_output.close()
        if args.collect_stats == 'granular':
            stats_output.close()


if __name__ == "__main__":
    main()
