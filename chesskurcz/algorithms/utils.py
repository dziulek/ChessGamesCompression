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

POSSIBLE_SCORES = ["0-1", "1-0", "1/2-1/2"]

FOUR_0 = 0x00000000
FOUR_1 = 0xffffffff
BIT_S = 32

MOVE_REGEX = r'(O-O-O|O-O|[QKRBN]?([a-h]|[1-8])?x?[a-h][1-8]([#+]|=[QRBN][+#]?)?|1/2-1/2|1-0|0-1)'

PGN_MOVE_REGEX = (
    r'(?P<CASTLE_LONG>O-O-O)|(?P<CASTLE_SHORT>O-O)|(?P<PIECE_TYPE>[QKRBN]?)'
    r'(?P<START_ROW>[1-8]?)(?P<START_COLUMN>[a-h]?)x?(?P<DEST_FIELD>[a-h][1-8])([#+]|'
    r'(?P<PROMOTION>=[QRBN])[+#]?)?'
)

pgn_re = re.compile(PGN_MOVE_REGEX)


THRASH_REGEX = r'(\?|\!|\{[^{}]*\}|\n)'

EMPTY_FIELD = -1

CHESS_BOARD = np.zeros((8, 8), dtype=np.int8) - EMPTY_FIELD

WHITE_KING = 0
WHITE_QUEEN = 1
WHITE_ROOK = 2
WHITE_BISHOP = 3
WHITE_KNIGHT = 4
WHITE_PAWN = 5

BLACK_KING = 6
BLACK_QUEEN = 7
BLACK_ROOK = 8
BLACK_BISHOP = 9
BLACK_KNIGHT = 10
BLACK_PAWN = 11

PIECE_TO_INT = {
    'K': WHITE_KING,
    'Q': WHITE_QUEEN,
    'R': WHITE_ROOK,
    'B': WHITE_BISHOP,
    'N': WHITE_KNIGHT
}

BLACK_OFF = 6

def set_starting_board(board: np.ndarray):

    piece_pos_table: Dict[int, List[np.ndarray]] = {}
    # KINGS 
    board[7][4] = WHITE_KING
    board[0][4] = BLACK_KING
    piece_pos_table[WHITE_KING].append(np.array([7, 4], np.int32))
    piece_pos_table[BLACK_KING].append(np.array([0, 4], np.int32))

    board[7][3] = WHITE_QUEEN
    board[0][3] = BLACK_QUEEN
    piece_pos_table[WHITE_QUEEN].append(np.array([7, 3], np.int32))
    piece_pos_table[BLACK_QUEEN].append(np.array([0, 3], np.int32))

    board[7][0] = WHITE_ROOK
    board[7][7] = WHITE_ROOK
    board[0][0] = BLACK_ROOK
    board[0][7] = BLACK_ROOK
    piece_pos_table[WHITE_ROOK].append(np.array([7, 0], np.int32))
    piece_pos_table[WHITE_ROOK].append(np.array([7, 7], np.int32))
    piece_pos_table[BLACK_ROOK].append(np.array([0, 0], np.int32))
    piece_pos_table[BLACK_ROOK].append(np.array([0, 7], np.int32))

    board[7][2] = WHITE_BISHOP
    board[7][5] = WHITE_BISHOP
    board[0][2] = BLACK_BISHOP
    board[0][5] = BLACK_BISHOP
    piece_pos_table[WHITE_BISHOP].append(np.array([7, 2], np.int32))
    piece_pos_table[WHITE_BISHOP].append(np.array([7, 5], np.int32))
    piece_pos_table[BLACK_BISHOP].append(np.array([0, 2], np.int32))
    piece_pos_table[BLACK_BISHOP].append(np.array([0, 5], np.int32))


    board[7][1] = WHITE_KNIGHT
    board[7][6] = WHITE_KNIGHT
    board[0][1] = BLACK_KNIGHT
    board[0][6] = BLACK_KNIGHT
    piece_pos_table[WHITE_KNIGHT].append(np.array([7, 1], np.int32))
    piece_pos_table[WHITE_KNIGHT].append(np.array([7, 6], np.int32))
    piece_pos_table[BLACK_KNIGHT].append(np.array([0, 1], np.int32))
    piece_pos_table[BLACK_KNIGHT].append(np.array([0, 6], np.int32))

    for i in range(8):
        board[6][i] = WHITE_PAWN
        board[1][i] = BLACK_PAWN
        piece_pos_table[WHITE_PAWN].append(np.array([6, i], np.int32))
        piece_pos_table[BLACK_PAWN].append(np.array([1, i], np.int32))

    return piece_pos_table

def control_square(piece_type: str, piece_pos: Tuple[int, int], piece_control: Tuple[int, int]) -> bool:

    piece_pos_y, piece_pos_x = piece_pos
    piece_control_y, piece_control_x = piece_control

    if piece_type == 'K':
        
        return max(abs(piece_control_x - piece_pos_x), 
                   abs(piece_control_y - piece_pos_y)) == 1
    
    elif piece_type == 'Q':

        # row and columns
        if piece_control_x == piece_pos_x or \
            piece_control_y == piece_pos_y: return True
        
        # diagonals
        if piece_control_y + piece_control_x == piece_pos_x + piece_pos_y: return True

        if 8 - piece_control_y + piece_control_x == piece_pos_x + piece_pos_x: return True

        return False
    
    elif piece_type == 'R':

        if piece_control_x == piece_pos_x or \
            piece_control_y == piece_pos_y: return True
        
        return False
    
    elif piece_type == 'B':

        if piece_control_y + piece_control_x == piece_pos_x + piece_pos_y: return True

        if 8 - piece_control_y + piece_control_x == piece_pos_x + piece_pos_x: return True

        return False
    
    elif piece_type == 'N':
        diff_x = piece_control_x - piece_pos_x
        diff_y = piece_control_y - piece_pos_y
        return abs(diff_x) + \
            abs(diff_y) == 3 and min(diff_y, diff_x) == 1
    
    else: # pawn

        pass

def field_to_string(coords: np.ndarray) -> str:

    return chr(ord('a') + coords[1]) + str(8 - coords[0])

def uci_to_coords(uci: str) -> np.ndarray:

    return np.array([ord(uci[0]) - ord('a'), 8 - int(uci[1])], dtype=np.int32)

def pgn_to_uci_move(board: np.ndarray, pgn_move: str, piece_pos_table: Dict[int, List[np.ndarray]], turn: bool) -> str:

    _match = pgn_re.match(pgn_move)

    # STANDARD PIECE MOVE - given piece and destination
    if _match.group('CASTLE_LONG') is not None:
        if turn: 
            board[0][4] = EMPTY_FIELD
            board[0][0] = EMPTY_FIELD
            board[0][2] = BLACK_KING
            board[0][3] = BLACK_ROOK

            for p in piece_pos_table[WHITE_ROOK + turn * BLACK_OFF]:
                
                if np.sum(p) == 0: p = np.array([0, 3], np.int32)
            
            piece_pos_table[WHITE_KING + turn * BLACK_OFF][0] = np.array([0, 2], np.int32)
            return 'e8c8'
        
        board[7][4] = EMPTY_FIELD
        board[7][0] = EMPTY_FIELD
        board[7][2] = WHITE_KING
        board[7][3] = WHITE_ROOK 

        for p in piece_pos_table[WHITE_ROOK + turn * BLACK_OFF]:
            
            if np.sum(p) == 0: p = np.array([7, 3], np.int32)
        
        piece_pos_table[WHITE_KING + turn * BLACK_OFF][0] = np.array([7, 2], np.int32)

        return 'e1c1'

    if _match.group('CASTLE_SHORT') is not None:

        if turn:
            board[0][4] = EMPTY_FIELD
            board[0][7] = EMPTY_FIELD
            board[0][6] = BLACK_KING
            board[0][5] = BLACK_ROOK

            for p in piece_pos_table[WHITE_ROOK + turn * BLACK_OFF]:
                
                if np.sum(p) == 0: p = np.array([0, 5], np.int32)
            
            piece_pos_table[WHITE_KING + turn * BLACK_OFF][0] = np.array([0, 6], np.int32)

            return 'e8g8'
        
        board[7][4] = EMPTY_FIELD
        board[7][7] = EMPTY_FIELD
        board[7][6] = WHITE_KING
        board[7][5] = WHITE_ROOK

        for p in piece_pos_table[WHITE_ROOK + turn * BLACK_OFF]:
            
            if np.sum(p) == 0: p = np.array([7, 5], np.int32)
        
        piece_pos_table[WHITE_KING + turn * BLACK_OFF][0] = np.array([7, 6], np.int32)

        return 'e1g1'

    dest_field = uci_to_coords(_match.group('DEST_FIELD'))

    if _match('PIECE_TYPE') is not None:
        
        piece_type = _match('PIECE_TYPE')
        start_pos = np.array([-1, -1], dtype=np.int32)
        if _match('START_COLUMN') != '':
            start_pos[1] = int(_match('START_COLUMN')) - 1
        if _match('START_ROW') != '':
            start_pos[0] = 8 - int(_match('START_ROW'))

        for p in piece_pos_table[PIECE_TO_INT[piece_type] + turn * BLACK_OFF]:
            
            eq = p == start_pos
            if start_pos[eq == False].all() == -1 and control_square(piece_type, p, dest_field):

                board[p] = EMPTY_FIELD
                board[dest_field] = PIECE_TO_INT[piece_type] + turn * BLACK_OFF

                p = dest_field
    
                return field_to_string(p) + field_to_string(dest_field)
    
    # pawn move
    start_pos = np.ndarray([-1, -1], dtype=np.int32)
    
    if _match('START_COLUMN') != '':
        start_pos[1] = int(_match('START_COLUMN')) - 1
    
    for p in piece_pos_table[WHITE_PAWN + turn * BLACK_OFF]:

        eq  = p == start_pos
        if start_pos[eq == False].all() == -1 and control_square('P', p, dest_field):

            board[p] = EMPTY_FIELD
            board[dest_field] = PIECE_TO_INT[piece_type] + turn * BLACK_OFF

            if _match('PROMOTION') is not None:

                board[dest_field] = PIECE_TO_INT[_match('PROMOTION')] + turn * BLACK_OFF

                return field_to_string(p) + field_to_string(dest_field)
            
            return field_to_string(p) + field_to_string(dest_field)


def pgn_to_uci_game(moves: List[str]) -> List[str]:

    board = CHESS_BOARD.copy()

    uci_moves = []
    piece_pos_table = set_starting_board(board=board)
    turn = 0

    for move in moves:

        if move in POSSIBLE_SCORES:

            uci_moves.append(move)
            return uci_moves

        uci = pgn_to_uci_move(board, move, piece_pos_table, turn)
        uci_moves.append(uci)

        turn = 1 - turn

def standard_png_move_extractor(_in: str) -> List[List[str]]:

    move_token_reg = re.compile(MOVE_REGEX)
    l = [[]]
    find_list = re.findall(move_token_reg, _in)

    for f in find_list:
        if f[0] in set(POSSIBLE_SCORES):
            l[-1].append(f[0])
            l.append([])
            continue
        l[-1].append(f[0])

    if len(l[-1]) == 0: l.pop()

    return l

def compare_games(true: str, decompressed: str) -> bool:

    '''
        Since game representation may be different
        this function compares games of potentially 
        different representations. 
    '''

    a = chess.pgn.read_game(io.StringIO(true))
    b = chess.pgn.read_game(io.StringIO(decompressed))

    a = a.next()
    b = b.next()

    if a.game().headers['Result'] != b.game().headers['Result']: return False

    while a is not None:

        move_a = a.move
        if b is None: return False

        move_b = b.move

        if move_a.uci() != move_b.uci(): return False

        a = a.next()
        b = b.next()
        
    if b is not None: return False
        
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