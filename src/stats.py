import numpy as np
import chess
import chess.pgn
import threading
import json
import os
import time
import functools
import threading
from typing import List, Dict, Callable

from src.algorithms.utils import atomic_operation, sem_stats


WHITE_WINS = '1-0'
BLACK_WINS = '0-1'
DRAW = '1/2-1/2'

# Metrics
BITS_PER_MOVE = 'Number of bits per move'
BITS_PER_GAME = 'Number of bits per game'
COMPRESSION_RATIO = 'Compression ratio'
FILE_SIZE_UNCOMPRESSED = 'Size of source file'
FILE_SIZE_COMPRESSED = 'Size of compressed file'
MOVE_NO = 'Total number of moves'
GAME_NO = 'Total number of games'
RESULTS_DISTR = 'Distribution of results'
PIECE_MOVE_FREQ = 'Piece movement frequency'
COMPRESSION_TIME = 'Time to compress the file'
DECOMPRESSION_TIME = 'Time to decompress the file'
MOVE_DISTR_START = 'Start square move frequency'
MOVE_DISTR_END = 'End square move frequency'

class Stats:

    def __init__(self, path_data: str, path_encoded, sem: threading.Semaphore=None) -> None:

        self.instance_semaphore = threading.Semaphore()
        self.path_data = path_data
        self.path_encoded = path_encoded
        self.move_no = 0
        self.game_no = 0
        self.move_distr_start = np.zeros((8, 8), dtype=np.int32)
        self.move_distr_end = np.zeros((8, 8), dtype=np.int32)
        self.results_distr = {
            WHITE_WINS: 0,
            BLACK_WINS: 0,
            DRAW: 0
        }

        if sem is not None:
            self.sem = sem

        self.move_distr_piece = {
            'k': 0,
            'q': 0,
            'r': 0,
            'b': 0,
            'n': 0,
            'p': 0
        }

        self.__start = 0
        self.__end = 0
        self.compression_time = 0
        self.decompression_time = 0

        self.metrics = {
            BITS_PER_MOVE: 0,
            BITS_PER_GAME: 0,
            COMPRESSION_RATIO: 0,
            FILE_SIZE_UNCOMPRESSED: 0,
            FILE_SIZE_COMPRESSED: 0,
            MOVE_NO: 0,
            GAME_NO: 0,
            RESULTS_DISTR: None,
            PIECE_MOVE_FREQ: None,
            MOVE_DISTR_START: None,
            MOVE_DISTR_END: None,
            COMPRESSION_TIME: 0,
            DECOMPRESSION_TIME: 0         
        }

    @atomic_operation(sem=sem_stats)
    def add_game(self, game: chess.pgn.Game) -> None:

        self.game_no += 1
        self.results_distr[game.headers["Result"]] += 1

    @atomic_operation(sem=sem_stats)
    def add_move(self, move: chess.Move, board: chess.Board) -> None:

        uci = move.uci()
        start, end = uci[:2], uci[2:]
        self.move_no += 1
        self.move_distr_start[ord(start[0]) - ord('a')][int(start[1]) - 1] += 1
        self.move_distr_end[ord(end[0]) - ord('a')][int(end[1]) - 1] += 1

        self.move_distr_piece[board.piece_at(move.from_square).symbol().lower()] += 1

    def set_metrics(self,):

        self.metrics[MOVE_NO] = self.move_no
        self.metrics[GAME_NO] = self.game_no
        self.metrics[FILE_SIZE_COMPRESSED] = os.path.getsize(self.path_encoded)
        self.metrics[FILE_SIZE_UNCOMPRESSED] = os.path.getsize(self.path_data)
        self.metrics[BITS_PER_GAME] = 8 * self.metrics[FILE_SIZE_COMPRESSED] / self.game_no
        self.metrics[BITS_PER_MOVE] = 8 * self.metrics[FILE_SIZE_COMPRESSED] / self.move_no
        self.metrics[COMPRESSION_RATIO] = 1 - \
            self.metrics[FILE_SIZE_COMPRESSED] / self.metrics[FILE_SIZE_UNCOMPRESSED]
        
        self.metrics[COMPRESSION_TIME] = self.compression_time 
        self.metrics[DECOMPRESSION_TIME] = self.decompression_time 

        self.metrics[PIECE_MOVE_FREQ] = self.move_distr_piece
        self.metrics[RESULTS_DISTR] = self.results_distr

        k = []
        v = []
        for i in range(8):
            for j in range(8):
                k.append(chr(ord('a') + j) + str(i))
                v.append(int(self.move_distr_start[i][j]))
        self.metrics[MOVE_DISTR_START] = dict(zip(
            k, v
        ))

        k = []
        v = []
        for i in range(8):
            for j in range(8):
                k.append(chr(ord('a') + j) + str(i))
                v.append(int(self.move_distr_end[i][j]))
        self.metrics[MOVE_DISTR_END] = dict(zip(
            k, v
        ))   

    def start_timer(self,):

        self.__start = time.time()

    def stop_timer(self,):

        self.__end = time.time()
        self.compression_time = self.__end - self.__start

    def set_compression_time(self, time: int):

        self.compression_time = time

    def set_decompression_time(self, time: int):

        self.decompression_time = time

    def get_dict(self,):

        d = {}

        d['Metrics'] = self.metrics
        d['Source path'] = self.path_data
        d['Encoded path'] = self.path_encoded   

        return d     

    def get_json(self,):

        return json.dumps(self.get_dict())
    
    def include_games(self, games: List[chess.pgn.Game], verbose=False) -> None:

        for game in games:

            self.add_game(game)
            while game is not None:
                
                if game.next() is not None:
                    self.add_move(game.next().move, game.board())

                game = game.next()
