import threading
import io
import sys
import os
from typing import Dict, List, Tuple, Callable
import time
import json
import chess
import chess.pgn

from src.utils import read_lines, write_lines, atomic_operation, write_binary, read_binary, filterLines
from src.utils import time_elapsed, processLines
from src.compressors import encode_rank, decode_rank
from src.fast_naive import encode_naive, decode_naive
from src.stats import Stats

NUMBER_OF_THREADS = 6
BATCH_SIZE = int(1e4)

def process_encode(r_buff: io.TextIOWrapper, w_buff: io.TextIOWrapper, \
                    func: Callable[[List[str]], bytes], batch_s: int):

    lines = read_lines(r_buff, batch_s)

    filterLines(lines)
    lines = processLines(lines)

    while lines is not None and len(lines) > 0:
        
        enc_data = func(lines)

        write_binary(w_buff, enc_data)
        lines = read_lines(r_buff, batch_s)
        filterLines(lines)
        lines = processLines(lines)
    

def process_decode(r_buff: io.TextIOWrapper, w_buff: io.TextIOWrapper, 
                    func: Callable[[io.TextIOWrapper, int, bool, List], 
                    List[str]], batch_s: int, ret_games=None):
    
    dec_games = ['1']

    while dec_games:

        dec_games = func(r_buff, batch_s, ret_games is not None, ret_games)
        write_lines(w_buff, dec_games)

@time_elapsed()
def run_encode(r_buff: io.TextIOWrapper, w_buff: io.TextIOWrapper, 
               func: Callable[[io.TextIOWrapper, List[str], Stats], bytes]=None,
               threads_no: int=1, batch_s=BATCH_SIZE):

    threads = [
        threading.Thread(target=process_encode, args=(r_buff, w_buff, func, batch_s)) for _ in range(threads_no)
    ]

    for thread in threads:
        thread.start()


    for thread in threads:
        thread.join()

@time_elapsed()
def run_decode(r_buff: io.TextIOWrapper, w_buff: io.TextIOWrapper, 
               func: Callable[[io.TextIOWrapper, int, bool, List], List[str]]=None,
               threads_no: int=1, batch_s=BATCH_SIZE, ret_games=None):

    games_sep_list = []

    for _ in range(threads_no):
        if ret_games is not None:
            games_sep_list.append([])
        else:
            games_sep_list.append(None)
    

    threads = [
        threading.Thread(target=process_decode, args=(r_buff, w_buff, func, batch_s, games_sep_list[i])) for i in range(threads_no)
    ]

    for thread in threads: thread.start()

    for thread in threads: thread.join()
    
    if ret_games is not None:
        for i in range(threads_no): ret_games += games_sep_list[i]


def main():

    script_path = os.path.realpath(__file__)
    script_path = script_path[:script_path.rfind('/')]    

    files = ['test_file.pgn']

    algorithms: Dict[str, Tuple[function, function]] = {
        'rank': (encode_rank, decode_rank),
        'naive': (encode_naive, decode_naive)
    }
    
    dest_files = [file for file in files]
    files = [script_path + '/data/' + file for file in files]
    dest_files = [script_path + '/compressed_data/' + file for file in dest_files]

    global_stats = {}

    for file, dest_file in zip(files, dest_files):
        
        global_stats[file] = {}

        for alg in algorithms:
            
            stats = Stats(file, dest_file)

            #=======ENCODE===============================
            encoding_func, decoding_func = algorithms[alg]

            source_buff = open(file, 'r')
            dest_buff = open(dest_file, 'wb')

            compression_time = run_encode(
                source_buff,
                dest_buff,
                encoding_func, 1, BATCH_SIZE
            )

            source_buff.close()
            dest_buff.close()

            #=======DECODE================================
            games = None

            source_buff = open(dest_file, 'rb')
            dest_buff = open('__games.dec', 'w')

            decompression_time = run_decode(
                source_buff,
                dest_buff,
                decoding_func, 1, BATCH_SIZE, games
            )
            source_buff.close()
            dest_buff.close()   

            g_buff = open('__games.dec', 'r')

            games = g_buff.readlines()
            filterLines(games)
            games = processLines(games)
            g_buff.close()

            for game in games:
                g = chess.pgn.read_game(io.StringIO(game))
                stats.include_games([g])
            stats.set_compression_time(compression_time)
            stats.set_decompression_time(decompression_time)

            stats.set_metrics()
            global_stats[file][alg] = stats.get_dict()            

            #========CHECK================================
            
            ref = open(file, 'r')
            dec = open('__games.dec', 'r')
            ref_str = ref.read()
            dec_str = dec.read()

            ref.close()
            dec.close()

            # assert ref_str.replace('\n', '') == dec_str.replace('\n', ''), 'NOT EQUAL'




    with open(script_path + '/results/' + str(time.time())[-4:] + '.json', 'w') as f:

        f.write(json.dumps(global_stats))

if __name__ == "__main__":

    main()