import threading
import chess
import os

from src.compressors import process_decode, process_encode, BATCH_SIZE, sem_stats
from src.stats import Stats

from typing import List, Dict

def main():

    script_path = os.path.realpath(__file__)
    script_path = script_path[:script_path.rfind('/')]

    source_file = script_path + '/data/bin_test_file.txt'
    dest_file = script_path + '/data/encoded_test_file.bin'

    stats = Stats(source_file, dest_file, sem_stats)

    read_buff = open(source_file, 'r')
    write_buff = open(dest_file, 'wb')

    threads: List[threading.Thread] = []
    
    
    for i in range(1):
        threads.append(threading.Thread(target=process_encode, args=(read_buff, write_buff, BATCH_SIZE, 'rank', stats)))

    stats.start_timer()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    stats.stop_timer()

    read_buff.close()
    write_buff.close()

    stats.set_metrics()
    print(stats.get_json())

if __name__ == "__main__":

    main()