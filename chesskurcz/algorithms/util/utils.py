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
    r'(?P<START_ROW>[1-8]?)(?P<START_COLUMN>[a-h]?)(?P<CAPTURE>x?)(?P<DEST_FIELD>[a-h][1-8])([#+]|'
    r'(?P<PROMOTION>=[QRBN])[+#]?)?'
)

WHITE = 0
BLACK = 1

pgn_re = re.compile(PGN_MOVE_REGEX)


THRASH_REGEX = r'(\?|\!|\{[^{}]*\}|\n)'

EMPTY_FIELD = -1

CHESS_BOARD = np.zeros((8, 8), dtype=np.int8) + EMPTY_FIELD

KING = 0
QUEEN = 1
ROOK = 2
BISHOP = 3
KNIGHT = 4
PAWN = 5

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

CASTLES_MOVES = {
    'CASTLE_SHORT': {
        WHITE: {
            KING: [(7, 4), (7, 6), 'e1g1'],
            ROOK: [(7, 7), (7, 5)]
        },
        BLACK: {
            KING: [(0, 4), (0, 6), 'e8g8'],
            ROOK: [(0, 7), (0, 5)]
        }
    },
    'CASTLE_LONG': {
        WHITE: {
            KING: [(7, 4), (7, 2), 'e1c1'],
            ROOK: [(7, 0), (7, 3)],            
        },
        BLACK: {
            KING: [(0, 4), (0, 2), 'e8c8'],
            ROOK: [(0, 0), (0, 3)]          
        }
          
    }
}

PIECE_TO_INT = {
    'K': WHITE_KING,
    'Q': WHITE_QUEEN,
    'R': WHITE_ROOK,
    'B': WHITE_BISHOP,
    'N': WHITE_KNIGHT,
    'P': WHITE_PAWN
}

BLACK_OFF = 6

def set_starting_board(board: np.ndarray):

    piece_pos_table: Dict[int, List[np.ndarray]] = {}
    for i in range(BLACK_PAWN + 1): piece_pos_table[i] = []
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

def free_path_board(board: np.ndarray, start: np.ndarray, end: np.ndarray, piece_type: str) -> bool:

    if piece_type == 'N': return True

    _diff = np.abs(end - start)
    
    _direction = np.sign(end - start)

    _c = start.copy()
    _c += _direction
    while not (_c == end).all():

        if np.min(_c) < 0 or np.max(_c) > 7: return False

        if board[_c[0], _c[1]] != EMPTY_FIELD: return False
        _c += _direction

    return True

def control_square(board: np.ndarray, piece_type: str, piece_pos: Tuple[int, int], piece_control: Tuple[int, int], color=0, capture=False) -> bool:

    ppos = np.array(piece_pos, dtype=np.int8)
    pcontrol = np.array(piece_control, dtype=np.int8)

    # check if move is possible
    if not free_path_board(board, ppos, pcontrol, piece_type): return False

    if piece_type == 'K':
        return np.max(np.abs(ppos - pcontrol)) == 1
    
    elif piece_type == 'Q':

        # row and columns
        if (ppos == pcontrol).any(): return True
        
        # diagonals
        if np.sum(ppos - pcontrol) == 0: return True
        if -ppos[0] + ppos[1] == -pcontrol[0] + pcontrol[1]: return True

        return False
    
    elif piece_type == 'R':

        if (ppos == pcontrol).any(): return True
        return False
    
    elif piece_type == 'B':

        if np.sum(ppos - pcontrol) == 0: return True
        if -ppos[0] + ppos[1] == -pcontrol[0] + pcontrol[1]: return True

        return False
    
    elif piece_type == 'N':
        return np.sum(np.abs(pcontrol - ppos)) == 3 and np.min(np.abs(pcontrol - ppos)) == 1
    
    else: # pawn

        dir = (-1) ** color
        if capture:
            if (ppos + np.array([-1 * dir, -1], dtype=np.int8) == pcontrol).all(): return True
            if (ppos + np.array([-1 * dir, 1], dtype=np.int8) == pcontrol).all(): return True
        else:
            if ppos[0] == 6 or ppos[0] == 1:
                if (ppos + np.array([-2 * dir, 0], dtype=np.int8) == pcontrol).all(): return True
            if (ppos + np.array([-1 * dir, 0], dtype=np.int8) == pcontrol).all(): return True
        
        return False

def field_to_string(coords: np.ndarray) -> str:

    return chr(ord('a') + coords[1]) + str(8 - coords[0])

def uci_to_coords(uci: str) -> np.ndarray:

    return np.array([8 - int(uci[1]), ord(uci[0]) - ord('a')], dtype=np.int32)

def pinned_piece_move(board: np.ndarray, piece_pos: np.ndarray, king_pos: np.ndarray, dest_pos: np.ndarray, turn: int):

    v = piece_pos - king_pos

    if v[0] == 0 and v[1] == 0: return False

    DIAGONAL = False
    ROW_COLUMN = False
    if np.min(v) == 0:
        ROW_COLUMN = True
    if abs(v[0]) == abs(v[1]):
        DIAGONAL = True
    if not DIAGONAL and not ROW_COLUMN:
        return False

    __dir = np.sign(v)
    u = dest_pos - piece_pos

    # move which does not uncover the king
    if (np.abs(__dir) == np.abs(np.sign(u))).all() and board[tuple(piece_pos)] != KNIGHT + turn * BLACK_OFF:
        return False

    # path from king to piece
    cur_pos = king_pos.copy() + __dir
    while np.max(cur_pos) < 8 and np.min(cur_pos) > -1 and board[tuple(cur_pos)] == EMPTY_FIELD:
        cur_pos += __dir

    if np.max(cur_pos) < 8 and np.min(cur_pos) > -1 and (cur_pos == piece_pos).all():
        
        cur_pos = piece_pos.copy() + __dir
        if np.max(cur_pos) > 7 or np.min(cur_pos) < 0: return False
        # find first piece that is on the same diagonal or row/column
        while board[cur_pos[0], cur_pos[1]] == EMPTY_FIELD:

            cur_pos += __dir
            if np.max(cur_pos) > 7 or np.min(cur_pos) < 0: return False

        if board[cur_pos[0], cur_pos[1]] == QUEEN + (1 - turn) * BLACK_OFF:
            return True

        if ROW_COLUMN:
            if board[cur_pos[0], cur_pos[1]] == ROOK + (1 - turn) * BLACK_OFF:
                return True

        else: # DIAGONAL
            
            if board[cur_pos[0], cur_pos[1]] == BISHOP + (1 - turn) * BLACK_OFF:
                return True
    
    return False

def pgn_to_uci_move(board: np.ndarray, pgn_move: str, piece_pos_table: Dict[int, List[np.ndarray]], turn: bool) -> str:

    _match = pgn_re.match(pgn_move)
    CASTLE = 'CASTLE_SHORT' if _match.group('CASTLE_SHORT') is not None else None
    CASTLE = 'CASTLE_LONG' if _match.group('CASTLE_LONG') is not None else CASTLE
    PIECE_TYPE = _match.group('PIECE_TYPE')
    DEST_FIELD = _match.group('DEST_FIELD')
    CAPTURE = _match.group('CAPTURE')
    START_COLUMN = _match.group('START_COLUMN')
    START_ROW = _match.group('START_ROW')
    PROMOTION = _match.group('PROMOTION')

    PIECE_OFF = turn * BLACK_OFF

    if CASTLE is not None:
        type_castle = CASTLES_MOVES[CASTLE]
        board[type_castle[turn][KING][0]] = EMPTY_FIELD
        board[type_castle[turn][ROOK][0]] = EMPTY_FIELD
        board[type_castle[turn][KING][1]] = KING + PIECE_OFF
        board[type_castle[turn][ROOK][1]] = ROOK + PIECE_OFF

        for i, p in enumerate(piece_pos_table[ROOK + PIECE_OFF]):
            if tuple(p) == type_castle[turn][ROOK][0]: 
                piece_pos_table[ROOK + PIECE_OFF][i] = np.array(type_castle[turn][ROOK][1])
        
        piece_pos_table[KING + PIECE_OFF][0] = np.array(type_castle[turn][KING][1])

        return type_castle[turn][KING][2]


    DEST_FIELD = uci_to_coords(DEST_FIELD)

    if CAPTURE:

        found = False        
        # en passant
        if board[tuple(DEST_FIELD)] == EMPTY_FIELD:
            taken_pos = DEST_FIELD + np.array([(-1) ** turn, 0], dtype=np.int8)
            for i, p in enumerate(piece_pos_table[PAWN + (1 - turn) * BLACK_OFF]):
                if (p == taken_pos).all():
                    piece_pos_table[PAWN + (1 - turn) * BLACK_OFF].pop(i)
                    board[taken_pos[0], taken_pos[1]] = EMPTY_FIELD
                    found = True
                    break

        else:
            # find captured piece and remove
            for i, p in enumerate(piece_pos_table[board[DEST_FIELD[0], DEST_FIELD[1]]]):
                if (p == DEST_FIELD).all(): 
                    found = True
                    piece_pos_table[board[DEST_FIELD[0], DEST_FIELD[1]]].pop(i)
                    break

        assert found == True, "NOT FOUND CAPTURED PIECE"

    if PIECE_TYPE != '':

        start_pos = np.array([-1, -1], dtype=np.int32)
        if START_COLUMN != '':
            start_pos[1] = ord(START_COLUMN) - ord('a')
        if START_ROW != '':
            start_pos[0] = 8 - int(START_ROW)

        for i, p in enumerate(piece_pos_table[PIECE_TO_INT[PIECE_TYPE] + PIECE_OFF]):
            
            eq = p == start_pos
            if (start_pos[eq == False] == -1).all() and control_square(board, PIECE_TYPE, p, DEST_FIELD) and \
                not pinned_piece_move(board, p, piece_pos_table[KING + PIECE_OFF][0], DEST_FIELD, turn):

                board[p[0], p[1]] = EMPTY_FIELD
                board[DEST_FIELD[0], DEST_FIELD[1]] = PIECE_TO_INT[PIECE_TYPE] + PIECE_OFF

                piece_pos_table[PIECE_TO_INT[PIECE_TYPE] + turn * BLACK_OFF][i] = DEST_FIELD
    
                return field_to_string(p) + field_to_string(DEST_FIELD)

    # pawn move
    start_pos = np.array([-1, -1], dtype=np.int32)
    
    if START_COLUMN != '':
        start_pos[1] = ord(START_COLUMN) - ord('a')
    
    for i, p in enumerate(piece_pos_table[WHITE_PAWN + PIECE_OFF]):

        eq  = p == start_pos
        if (start_pos[eq == False] == -1).all() and control_square(board, 'P', p, DEST_FIELD, turn, CAPTURE) and \
            not pinned_piece_move(board, p, piece_pos_table[KING + PIECE_OFF][0], DEST_FIELD, turn):

            board[p[0], p[1]] = EMPTY_FIELD
            board[DEST_FIELD[0], DEST_FIELD[1]] = PAWN + PIECE_OFF

            piece_pos_table[PAWN + PIECE_OFF][i] = DEST_FIELD

            if PROMOTION is not None:
                PROMOTED_TYPE = PIECE_TO_INT[PROMOTION[1]] + PIECE_OFF
                board[DEST_FIELD[0], DEST_FIELD[1]] = PROMOTED_TYPE
                # remove pawn
                piece_pos_table[PAWN + PIECE_OFF].pop(i)
                # add piece
                piece_pos_table[PROMOTED_TYPE].append(DEST_FIELD)

                return field_to_string(p) + field_to_string(DEST_FIELD)
            
            return field_to_string(p) + field_to_string(DEST_FIELD)


def pgn_to_uci_game(moves: List[str]) -> List[str]:

    board = CHESS_BOARD.copy()

    uci_moves = []
    piece_pos_table = set_starting_board(board=board)
    turn = 0

    for move in moves:

        if move in set(POSSIBLE_SCORES):

            uci_moves.append(move)
            return uci_moves

        uci = pgn_to_uci_move(board, move, piece_pos_table, turn)
        uci_moves.append(uci)

        turn = 1 - turn

    return uci_moves

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
        different representations+_
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

def compression_ratio(before, after) -> float:

    return 1 - os.path.getsize(after) / os.path.getsize(before)

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

def get_workspace_path() -> str:

    path = os.path.realpath(__file__)
    return path[: path.rfind('chesskurcz')]

def get_all_possible_uci_moves() -> Dict[str, int]:

    cnt = 0
    dictionary = {}
    for sy in range(8):
        for sx in range(8):
            for dy in range(8):
                for dx in range(8):
                    if sy == dy and sx == dx: continue

                    x_abs_diff = abs(sx - dx)
                    y_abs_diff = abs(dy - sy)                    
                    add = False
                    # columns
                    if sy == dy or sx == dx:
                        add = True
                    
                    # diagonals
                    elif sx + sy == dx + dy or -sx + sy == -dx + dy:
                        add = True
                    # knight moves
                    elif x_abs_diff + y_abs_diff == 3 and min(x_abs_diff, y_abs_diff) == 1:
                        add = True

                    if add:
                        dictionary[field_to_string(np.array([sy, sx])) + field_to_string(np.array([dy, dx]))] = cnt
                        cnt += 1
    
    # add promotions
    for i in range(8):
        for p in ['q', 'r', 'b', 'n']:
            uci_black = field_to_string(np.array([6, i])) + \
                    field_to_string(np.array([7, i])) + p
            uci_white = field_to_string(np.array([1, i])) + \
                    field_to_string(np.array([0, i])) + p

            dictionary[uci_black] = cnt
            cnt += 1
            dictionary[uci_white] = cnt
            cnt += 1 

            # right takes and promotes
            if i < 7:
                uci_black = field_to_string(np.array([6, i])) + \
                    field_to_string(np.array([7, i + 1])) + p
                uci_white = field_to_string(np.array([1, i])) + \
                    field_to_string(np.array([0, i + 1])) + p 

                dictionary[uci_black] = cnt
                cnt += 1
                dictionary[uci_white] = cnt 
                cnt += 1

            # left takes and promotes
            if i > 0:
                uci_black = field_to_string(np.array([6, i])) + \
                    field_to_string(np.array([7, i - 1])) + p
                uci_white = field_to_string(np.array([1, i])) + \
                    field_to_string(np.array([0, i - 1])) + p 

                dictionary[uci_black] = cnt
                cnt += 1
                dictionary[uci_white] = cnt 
                cnt += 1            

    for s in POSSIBLE_SCORES:
        dictionary[s] = cnt
        cnt += 1        

    return dictionary

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
                DEST_FIELD = chr(fa + i) + str(j + 1)
                # regular forward pawn move
                moves.append(DEST_FIELD)                
                # capture left pawn move
                moves.append(chr(fa + i + 1) + 'x' + DEST_FIELD)
                # capture right pawn move
                moves.append(chr(fa + i - 1) + 'x' + DEST_FIELD)

                # piece move
                for p in pieces:
                    # no capture
                    moves.append(p + DEST_FIELD)

                    #no capture with row or column index
                    for c in range(8):
                        moves.append(p + chr(fa + c) + DEST_FIELD)
                    for r in range(8):
                        moves.append(p + str(r + 1) + DEST_FIELD)

                    # capture
                    moves.append(p + 'x' + DEST_FIELD)

                    # capture with row or column index
                    for c in range(8):
                        moves.append(p + chr(fa + c) + 'x' + DEST_FIELD)
                    for r in range(8):
                        moves.append(p + str(r + 1) + 'x' + DEST_FIELD)

        # promotions
        pieces = ['Q', 'R', 'B', 'N']
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
    pass