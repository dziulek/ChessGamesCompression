import numpy as np
import os
import io
import threading
from typing import List, Dict, Callable, Tuple

import chess
import functools

read_sem = threading.Semaphore()
write_sem = threading.Semaphore()
write_bin_sem = threading.Semaphore()
read_bin_sem = threading.Semaphore()

sem_stats = threading.Semaphore()

POSSIBLE_SCORES = ["0-1", "1-0", "1/2-1/2"]

FOUR_0 = 0x00000000
FOUR_1 = 0xffffffff
BIT_S = 32

def extract_move_idx(bin: int, off_b: int, k: int):

    mask = (bin >> off_b) & (FOUR_1 >> (BIT_S - k))

    return mask

def atomic_operation(sem: threading.Semaphore=None):

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrap_semaphore(*args, **kwargs):
            
            if sem is not None:
                sem.acquire()

            result =  func(*args, **kwargs)

            if sem is not None:
                sem.release()

            return result
        
        return wrap_semaphore

    return decorator

@atomic_operation(sem=read_sem)
def read_lines(r_buff: io.TextIOWrapper, batch_size: int) -> List[str]:

    return r_buff.readlines(batch_size)

@atomic_operation(sem=read_bin_sem)
def read_binary(r_buff: io.TextIOWrapper, batch_size: int) -> bytes:

    return r_buff.read(batch_size)

@atomic_operation(sem=write_sem)
def write_lines(w_buff: io.TextIOWrapper, lines: List[str]) -> None:

    w_buff.write('\n'.join(lines))

@atomic_operation(sem=write_bin_sem)
def write_binary(w_buff: io.TextIOWrapper, data: bytes) -> None:

    w_buff.write(data)

def clearLine(line: str) -> str:

    pass

def filterLines(lines: List[str]):

    for i in range(len(lines) - 1, -1, -1):

        if len(lines[i]) < 2 or lines[i][0] == '[':
            lines.pop(i)

def processLine(line: str) -> str:

    if line[-1] == '\n':
        return line[:-1]
    return line

def sort_moves(moves: List[chess.Move]) -> None:

    l = []
    for move in moves:
        l.append(move_code(move))

    l.sort()
    return l

def move_code(move: chess.Move) -> str:

    if move.promotion is None:
        s = '0'
    else:
        s = str(move.promotion)
    
    return move.uci() + s

def move_from_code(mov_code: str) -> chess.Move:

    move = chess.Move.from_uci(mov_code[:4])

    if mov_code[-1] != '0':
        move.promotion = int(mov_code[-1])

    return move

def to_binary(_bin: int, BITS: int, bits: int, val: int, k: int) -> Tuple[int, int, int]:

    _carry = -1

    if bits + k > BITS:
        _bin <<= (BITS - bits)
        bits = (bits + k) % BITS
        _carry = (FOUR_1 >> (BITS - bits)) & val
        val >>= (bits)

        _bin |= val

        return _bin, _carry, bits

    _bin <<= k
    bits += k

    _bin |= val 

    return _bin, _carry, bits

def get_script_path() -> str:

    path = os.path.realpath(__file__)
    return path[: path.rfind('/')]

def main():

    pass

if __name__ == "__main__":

    main()