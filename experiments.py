import threading
import io
import sys
import os
from typing import Dict, List, Tuple, Callable
import time
import json
import chess
import chess.pgn
import datetime

from chesskurcz.algorithms.utils import get_script_path
from chesskurcz.stats import Stats
from chesskurcz.algorithms.encoder import Encoder


NUMBER_OF_THREADS = 6
BATCH_SIZE = int(1e4)


def main():

    script_path = get_script_path()   

    files = ['lichess_db_standard_rated_2014-10.pgn']

    algorithms: List[str] = ['apm']
    
    dest_files = [file for file in files]
    files = [script_path + '../data/' + file for file in files]
    dest_files = [script_path + '../compressed_data/' + file for file in dest_files]

    global_stats = {}

    for file, dest_file in zip(files, dest_files):
        
        global_stats[file] = {}

        for alg in algorithms:
            print('Algorithm:', alg)

            stats = Stats(file, dest_file)
            
            encoder = Encoder(alg=alg, par_workers=4, batch_size=BATCH_SIZE)

            start = time.time()
            encoder.encode(file, dest_file, verbose=True)
            compression_time = time.time() - start

            start = time.time()
            encoder.decode(dest_file, '__decoded.txt', verbose=True) 
            decompression_time = time.time() - start

            stats.set_compression_time(compression_time)
            stats.set_decompression_time(decompression_time)

            max_games = 100
            cnt = 0
            with open('__decoded.txt', 'r') as f:

                while cnt < max_games:
                    games = f.readlines(BATCH_SIZE)
                    if not games: break
                    stats.include_games(games)
            
            stats.set_metrics()
            global_stats[file][alg] = stats.get_dict()   

            os.remove('__decoded.txt')      

    output_file_name = datetime.datetime.today().strftime('%Y-%m-%d') \
              + '_' + str(time.time())[-4:] + '.json'
    
    print('Statistics saved in', output_file_name)
    with open(script_path + '/../results/' + output_file_name, 'w') as f:

        f.write(json.dumps(global_stats))
    pass

if __name__ == "__main__":

    main()