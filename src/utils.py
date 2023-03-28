import numpy as np
import os
import io
import threading
from typing import List, Dict

POSSIBLE_SCORES = ["0-1", "1-0", "1/2-1/2"]



def readLines(streamBuff: io.TextIOWrapper, batchSize: int, sem: threading.Semaphore) -> List[str]:

    sem.acquire()

    lines = streamBuff.readlines(batchSize)

    sem.release()

    return lines

def writeLines(streamBuff: io.TextIOWrapper, lines: List[str], sem: threading.Semaphore) -> None:

    sem.acquire()

    streamBuff.write('\n'.join(lines))

    sem.release()

def clearLine(line: str) -> str:

    return line

    # out = ''
    # for halfMove in line.split(' '):

    #     ind = halfMove.find('.')
    #     if ind >= 0:
    #         halfMove = halfMove[ind + 1:]

    #     out += halfMove

    # return out

def filterLines(lines: List[str]) -> List[str]:

    for i in range(len(lines) - 1, -1, -1):

        if len(lines[i]) < 2 or lines[i][0] == '[':
            lines.pop(i)

    return lines

def processLine(line: str) -> str:

    if line[-1] == '\n':
        return line[:-1]
    return line

def main():

    pass

if __name__ == "__main__":

    main()