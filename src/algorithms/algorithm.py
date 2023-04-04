import io, os, sys

import src.logger
from src.algorithms.transform import Transform
import algorithms
import compressors, fast_naive, apm
import multiprocessing
from typing import Callable, List, Dict

import numpy as np

class Encoder:

    def __init__(self, alg='apm', thread_no=1, batch_size=1e4) -> None:
        
        self.alg = alg
        self.thread_no = thread_no
        self.batch_size = batch_size
        self.thresh_for_mult_threads = 1e6

        self.def_pgn_parser = Transform()# TODO add default move extractor
        self.def_out_format = Transform()# TODO add default output format transform

    def __process_encode(self, in_stream: io.TextIOWrapper,
                        out_stream: io.TextIOWrapper, in_tran: Transform=None,
                        verbose=False):

        parser = in_tran if in_tran is not None else self.def_pgn_parser

        moves = 0

        while moves is not None and len(moves) > 0:
            
            # read the data from in_stream
            moves = parser.transform()# TODO read data
            _encoder: Callable[[List[List[str]]], bytes] = getattr(compressors, 'encode_' + self.alg)

            enc_data = _encoder(moves)

            # write the data to the out_stream
                  

    def encode(self, in_stream: io.TextIOWrapper,
                out_stream: io.TextIOWrapper, in_tran: Transform=None, 
                max_games=None, verbose=False):

        file_path = in_stream.name
        file_size = os.path.getsize(file_path)

        ub_games = max_games
        if ub_games is None: ub_games = np.inf

        games_cnt = 0
        bytes_read_tot = 0

        # log start compressing

        for i in range(self.thread_no):
            # start each thread
            pass

        # log progress bar

        for i in range(self.thread_no):
            pass
            # join each thread

        # log compretion

    def decode(self, in_stream: io.TextIOWrapper,
                out_stream: io.TextIOWrapper, out_tran: Transform=None,
                max_games=None, verbose=False):

        pass

