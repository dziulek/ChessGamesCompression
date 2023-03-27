import numpy as np
import os
import io
import threading
from typing import List, Dict

POSSIBLE_SCORES = ["0-1", "1-0", "1/2-1/2"]

BATCH_SIZE = 1000

sem_write = threading.Semaphore()
sem_read = threading.Semaphore()

def readLines(streamBuff: io.TextIOWrapper, batchSize: int, sem: threading.Semaphore) -> List[str]:

    sem.acquire()

    lines = streamBuff.readlines(batchSize)

    sem.release()

    return lines

def writeLines(streamBuff: io.TextIOWrapper, lines: str, sem: threading.Semaphore) -> None:

    sem.acquire()

    streamBuff.write(lines)

    sem.release()

def clearLine(line: str) -> str:

    out = ''
    for halfMove in line.split(' '):

        ind = halfMove.find('.')
        if ind >= 0:
            halfMove = halfMove[ind + 1:]

        if halfMove == 'O-O':
            halfMove = '0-0'

        out += halfMove

    return out


def processLine(line: str) -> str:

    if line[-1] == '\n': line = line[:-1]

    if len(line) <= 1: return ''

    if line[0] == '[': return ''

    if line[0].isdigit():
        
        # check if the score is at the end of the line
        last_line = False
        for s in POSSIBLE_SCORES:
            
            if line.find(s) + 1:
                last_line = True
                break
    
        # add new line character at the end of the line
        if last_line:
            line += '\n'

        return line

    return ''

def process_batch(readBuff: io.TextIOWrapper, writeBuff: io.TextIOWrapper, batch_size: int) -> None:

    global sem_write, sem_read

    lines = readLines(readBuff, BATCH_SIZE, sem_read)

    while lines is not None and len(lines) > 0:

        transformed = ''
        for line in lines:
            transformed += clearLine(processLine(line))

        writeLines(writeBuff, transformed, sem_write)

        lines = readLines(readBuff, BATCH_SIZE, sem_read)

def main():

    read_buff = open('../data/test_file.pgn', 'r')
    write_buff = open('../data/test_file.clean.pgn', 'w')

    threads: List[threading.Thread] = []
    
    for i in range(4):
        threads.append(threading.Thread(target=process_batch, args=(read_buff, write_buff, BATCH_SIZE)))



    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    read_buff.close()
    write_buff.close()

if __name__ == "__main__":

    main()