import io, os, sys

from src.algorithms.transform import TransformIn, TransformOut
from src.algorithms.utils import read_binary, write_binary, standard_png_move_extractor, write_lines

import src.algorithms.compressors, src.algorithms.fast_naive
from src.algorithms import apm
import multiprocessing
from multiprocessing import Value, Array
from typing import Callable, List, Dict

import numpy as np
import re

import ctypes

class Encoder:

    def __init__(self, alg='apm', thread_no=1, batch_size=1e4) -> None:
        
        self.alg = alg
        self.thread_no = thread_no
        self.batch_size = batch_size
        self.thresh_for_mult_threads = 1e6

        self.__THRASH_REGEX = re.compile(r'(\?|\!|\{[^{}]*\}|\[.*\]|\n|\#|\+)')

        self.def_pgn_parser = TransformIn(standard_png_move_extractor, self.__THRASH_REGEX)# TODO add default move extractor
        self.def_out_format = TransformOut()# TODO add default output format transform

        self.__bytes_read_tot = Value(ctypes.c_int32, 0)
        self.__games_cnt = Value(ctypes.c_int32, 0)

        self.dest_lock = multiprocessing.Lock()
        self.source_lock = multiprocessing.Lock()

    def __process_encode(self, in_stream: io.TextIOWrapper,
                        out_stream: io.TextIOWrapper, in_tran: TransformIn=None,
                        verbose=False):

        parser = in_tran if in_tran is not None else self.def_pgn_parser

        moves = 0

        while moves is not None:
            
            with self.source_lock:
                d = '\n'.join(in_stream.readlines(self.batch_size))
            if not d: break

            with self.__bytes_read_tot.get_lock():
                self.__bytes_read_tot.value += len(d)

            moves = parser.transform(d)# TODO read data

            with self.__games_cnt.get_lock():
                self.__games_cnt.value += len(moves)

            _encoder: Callable[[List[List[str]]], bytes] = getattr(apm, 'encode_' + self.alg)

            enc_data = _encoder(moves)
            with self.source_lock:
                write_binary(out_stream, enc_data)      

    def __process_decode(self, in_stream: io.TextIOWrapper,
                        out_stream: io.TextIOWrapper, out_tran: TransformOut=None,
                        verbose=False, max_games=np.inf):

        parser = out_tran if out_tran is not None else self.def_out_format

        while self.__games_cnt.value < max_games:

            with self.__games_cnt.get_lock():
                read_games = getattr(apm, 'read_games_' + self.alg)
                enc_data, g_no = read_games(in_stream, self.batch_size, max_games)
                if not g_no: break
                self.__games_cnt.value += g_no
            with self.__bytes_read_tot.get_lock():
                self.__bytes_read_tot.value += len(enc_data)

            _decoder: Callable[[bytes], List[List[str]]] = getattr(apm, 'decode_' + self.alg)
            dec_data = _decoder(enc_data)

            # out = parser.transform(dec_data)

                # need to be resolved
            write_lines(out_stream, ['asdf', 'asdf'])


    def encode(self, in_stream: io.TextIOWrapper,
                out_stream: io.TextIOWrapper, in_tran: TransformIn=None, 
                max_games=None, verbose=False):

        file_path = in_stream.name
        file_size = os.path.getsize(file_path)

        ub_games = max_games
        if ub_games is None: ub_games = np.inf

        self.__games_cnt.value = 0
        self.__bytes_read_tot.value = 0

        processes = [
            multiprocessing.Process(target=self.__process_encode, args=(in_stream, out_stream, in_tran, verbose)) 
            for _ in range(self.thread_no)
        ]

        for i in range(self.thread_no):
            processes[i].start()

        # log progress bar

        for i in range(self.thread_no):
            processes[i].join()

        # log compretion

    def decode(self, in_stream: io.TextIOWrapper,
                out_stream: io.TextIOWrapper, out_tran: TransformOut=None,
                max_games=np.inf, verbose=False):

        self.__games_cnt.value = 0
        self.__bytes_read_tot.value = 0

        processes = [
            multiprocessing.Process(target=self.__process_decode, 
            args=(in_stream, out_stream, out_tran, verbose, max_games)) for _ in range(self.thread_no)
        ]

        for i in range(self.thread_no): processes[i].start()
        
        for i in range(self.thread_no): processes[i].join()

