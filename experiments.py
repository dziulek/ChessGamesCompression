import threading
import io
import sys
import os
from typing import Dict, List, Tuple, Callable
import time
import json

from src.utils import read_lines, write_lines, atomic_operation, write_binary, read_binary, filterLines
from src.compressors import encode_rank, decode_rank
from src.fast_naive import encode_naive, decode_naive
from src.stats import Stats

NUMBER_OF_THREADS = 6
BATCH_SIZE = int(1e4)

def process_encode(r_buff: io.TextIOWrapper, w_buff: io.TextIOWrapper, \
                    stats: Stats, func: Callable[[io.TextIOWrapper, List[str], Stats], bytes], batch_s: int):

    lines = read_lines(r_buff, batch_s)

    filterLines(lines)

    while lines is not None and len(lines) > 0:
        
        enc_data = func(lines, stats)

        write_binary(w_buff, enc_data)
        lines = read_lines(r_buff, batch_s)
        filterLines(lines)
        print(len(lines))
    

def process_decode(r_buff: io.TextIOWrapper, w_buff: io.TextIOWrapper, 
                    stats: Stats, func: Callable[[io.TextIOWrapper, int], 
                    List[str]], batch_s: int):
    
    dec_games = ['1']

    while dec_games:

        dec_games = func(r_buff, batch_s)
        write_lines(w_buff, dec_games)


def run_encode(r_buff: io.TextIOWrapper, w_buff: io.TextIOWrapper, 
               func: Callable[[io.TextIOWrapper, List[str], Stats], bytes]=None,
               threads_no: int=1, stats: Stats=None, batch_s=BATCH_SIZE):

    threads = [
        threading.Thread(target=process_encode, args=(r_buff, w_buff, stats, func, batch_s)) for i in range(threads_no)
    ]

    stats.start_timer()
    for thread in threads:
        thread.start()


    for thread in threads:
        thread.join()

    stats.stop_timer()

def run_decode(r_buff: io.TextIOWrapper, w_buff: io.TextIOWrapper, func: Callable=None):

    pass



def main():

    script_path = os.path.realpath(__file__)
    script_path = script_path[:script_path.rfind('/')]    

    files = ['bin_test_file.txt']

    algorithms: Dict[str, Tuple[function, function]] = {
        'rank': (encode_rank, decode_rank)
        # 'naive': (encode_naive, decode_naive)
    }
    
    dest_files = [file for file in files]
    files = [script_path + '/data/' + file for file in files]
    dest_files = [script_path + '/compressed_data/' + file for file in dest_files]

    global_stats = {}

    for file, dest_file in zip(files, dest_files):
        
        global_stats[file] = {}

        for alg in algorithms:

            encoding_func, decoding_func = algorithms[alg]

            source_buff = open(file, 'r')
            dest_buff = open(dest_file, 'wb')

            stats = Stats(file, dest_file, sem=threading.Semaphore())

            run_encode(
                source_buff,
                dest_buff,
                encoding_func, 1, stats, BATCH_SIZE
            )

            source_buff.close()
            dest_buff.close()

            stats.set_metrics()
            global_stats[file][alg] = stats.get_dict()            

    
    with open(script_path + '/results/' + str(time.time())[-4:] + '.json', 'w') as f:

        f.write(json.dumps(global_stats))

if __name__ == "__main__":

    main()