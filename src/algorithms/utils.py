import numpy as np
import os
import io
import threading
from typing import List, Dict, Callable, Tuple

import chess.pgn

import re
import chess
import functools
import time

read_sem = threading.Semaphore()
write_sem = threading.Semaphore()
write_bin_sem = threading.Semaphore()
read_bin_sem = threading.Semaphore()

sem_stats = threading.Semaphore()

POSSIBLE_SCORES = ["0-1", "1-0", "1/2-1/2"]

FOUR_0 = 0x00000000
FOUR_1 = 0xffffffff
BIT_S = 32

MOVE_REGEX = r'(O-O-O|O-O|[QKRBN]?([a-h]|[1-8])?x?[a-h][1-8]([#+]|=[QRBN][+#]?)?|1/2-1/2|1-0|0-1)'
move_token_reg = re.compile(MOVE_REGEX)

THRASH_REGEX = r'(\?|\!|\{[^{}]*\})'
thrash_token_reg = re.compile(THRASH_REGEX)

def compare_games(true: List[str], decompressed: List[str]) -> bool:

    '''
        Since game representation may be different
        this function compares games of potentially 
        different representations. 
    '''

    for g_true, g_dec in zip(true, decompressed):

        a = str(chess.pgn.read_game(io.StringIO(g_true)))
        b = str(chess.pgn.read_game(io.StringIO(g_dec)))
        if a != b:
            
            return False
        
    return True

def preprocess_lines(lines: List[str], **kwargs):

    return processLines(filterLines(lines), **kwargs)
    

def extract_move_idx(bin: int, off_b: int, k: int):

    mask = (bin >> off_b) & (FOUR_1 >> (BIT_S - k))

    return mask

def time_elapsed():

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrap_time(*args, **kwargs):

            start = time.time()

            func(*args, **kwargs)

            return time.time() - start
        
        return wrap_time
    
    return decorator

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
    w_buff.write('\n')

@atomic_operation(sem=write_bin_sem)
def write_binary(w_buff: io.TextIOWrapper, data: bytes) -> None:

    w_buff.write(data)

def clearLine(line: str) -> str:

    pass

def filterLines(lines: List[str]) -> List[str]:

    for i in range(len(lines) - 1, -1, -1):

        if len(lines[i]) < 2 or lines[i][0] == '[':
            lines.pop(i)

    return lines

def processLine(line: str, regex_drop: re.Pattern=None,
                regex_take: re.Pattern=None, token_transform: Callable=None) -> str:
    out = line
    if regex_drop is not None:
        out = re.sub(regex_drop, '', out)
    if regex_take is not None:
        find_list = regex_take.findall(out)

        out = ' '.join(t[0] for t in find_list)

    if token_transform is not None:

        tokens = [o.strip() for o in out.split(' ')]
        tokens = token_transform(tokens)
        out = ' '.join(tokens)
    
    if out[-1] == '\n':
        return out[:-1]
    
    return out
    
def processLines(lines: List[str], regex_drop: re.Pattern=None,
                 regex_take: re.Pattern=None, token_transform: Callable=None) -> List[str]:

    out = []
    for line in lines:
        out.append(processLine(line, regex_drop, regex_take, token_transform))

    return out

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
    return path[: path.rfind('algorithms')]

def get_all_possible_moves(bits_per_move: int) -> List[str]:
    '''
        Function returns all possible moves which can be identified
        in .pgn notation. The capture of the piece represented by
        the character 'x', as well as check sign '+' and mate '#'
        could be ommited.
    '''
    moves: List[str] = []
    fa = ord('a')
    pieces = ['Q', 'K', 'R', 'B', 'N']    

    if bits_per_move >= 16:

        # we can include illegal moves also which 
        # won't be used -> easier implementation

        # pawn moves
        for i in range(8):
            for j in range(8):
                dest_field = chr(fa + i) + str(j + 1)
                # regular forward pawn move
                moves.append(dest_field)                
                # capture left pawn move
                moves.append(chr(fa + i + 1) + 'x' + dest_field)
                # capture right pawn move
                moves.append(chr(fa + i - 1) + 'x' + dest_field)

                # piece move
                for p in pieces:
                    # no capture
                    moves.append(p + dest_field)

                    #no capture with row or column index
                    for c in range(8):
                        moves.append(p + chr(fa + c) + dest_field)
                    for r in range(8):
                        moves.append(p + str(r + 1) + dest_field)

                    # capture
                    moves.append(p + 'x' + dest_field)

                    # capture with row or column index
                    for c in range(8):
                        moves.append(p + chr(fa + c) + 'x' + dest_field)
                    for r in range(8):
                        moves.append(p + str(r + 1) + 'x' + dest_field)

        # promotions
        for i in range(8):
            for p in pieces:
                # no capture
                moves.append(chr(fa + i) + '8=' + p)
                moves.append(chr(fa + i) + '1=' + p)
                # with capture left
                for j in range(8):
                    moves.append(chr(fa + j) + 'x' + chr(fa + i) + '8=' + p)
                    moves.append(chr(fa + j) + 'x' + chr(fa + i) + '1=' + p)

        moves.append('O-O-O')
        moves.append('O-O')

        for s in POSSIBLE_SCORES:
            moves.append(s)

        return moves

def main():

    pass

if __name__ == "__main__":

    main()