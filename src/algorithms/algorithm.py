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

        self.__THRASH_REGEX = re.compile(r'(\?|\!|\{[^{}]*\}|\[[^\[\]]*\]|\n|\#|\+)')

        self.def_pgn_parser = TransformIn(standard_png_move_extractor, self.__THRASH_REGEX)# TODO add default move extractor
        self.def_out_format = TransformOut()# TODO add default output format transform

        self.__bytes_read_tot = Value(ctypes.c_int32, 0)
        self.__games_cnt = Value(ctypes.c_int32, 0)

        self.dest_lock = multiprocessing.Lock()
        self.source_lock = multiprocessing.Lock()

    def __reader(self, path: str, Q: multiprocessing.Queue, binary=False, max_games=np.inf):

        with open(path, 'r') as in_stream:

            while 1:

                if not Q.empty(): continue
                
                in_stream.flush()
                if binary:

                    read_games = getattr(apm, 'read_games_' + self.alg)
                    enc_data, g_no = read_games(in_stream, self.batch_size, max_games)

                    if not enc_data:
                        break

                    Q.put(tuple([enc_data, g_no]))
                    self.__bytes_read_tot.value += len(enc_data)

                else: 
                    d = ''.join(in_stream.readlines(self.batch_size))
                    if not d: break
                    Q.put(d)
                self.__bytes_read_tot.value += len(d)
        
        for _ in range(self.thread_no): Q.put('kill')

    def __writer(self, out_stream: io.TextIOWrapper, Q: multiprocessing.Queue, binary=False, max_games=np.inf):

        while 1:
            d = Q.get()
            if d == 'kill': break

            out_stream.flush()
            if binary:
                enc_data = d
                out_stream.write(enc_data)
            else: 
                out_stream.write(d)

    def __process_encode(self, q_read: multiprocessing.Queue,
                        q_write: multiprocessing.Queue, in_tran: TransformIn=None,
                        verbose=False):

        parser = in_tran if in_tran is not None else self.def_pgn_parser

        moves = 0

        while moves is not None:
            
            data = q_read.get()

            if data == 'kill': break

            moves = parser.transform(data)
            _encoder: Callable[[List[List[str]]], bytes] = getattr(apm, 'encode_' + self.alg)

            enc_data = _encoder(moves)
            q_write.put(enc_data)

    def __process_decode(self, Q_read: multiprocessing.Queue,
                        Q_write: multiprocessing.Queue, out_tran: TransformOut=None,
                        verbose=False, max_games=np.inf):

        parser = out_tran if out_tran is not None else self.def_out_format

        while 1:
            
            d = Q_read.get()
            if d == 'kill': break

            enc_data, g_no = d

            _decoder: Callable[[bytes], List[List[str]]] = getattr(apm, 'decode_' + self.alg)
            dec_data = _decoder(enc_data)

            Q_write.put(dec_data)


    def encode(self, in_stream: io.TextIOWrapper,
                out_stream: io.TextIOWrapper, in_tran: TransformIn=None, 
                max_games=None, verbose=False):

        # file_path = in_stream.name
        # file_size = os.path.getsize(file_path)

        ub_games = max_games
        if ub_games is None: ub_games = np.inf

        manager = multiprocessing.Manager()
        Q_read = manager.Queue()    
        Q_write = manager.Queue()
        pool = multiprocessing.Pool(multiprocessing.cpu_count() + 2)

        writer = pool.apply_async(self.__writer, (out_stream, Q_read, True, max_games))
        reader = pool.apply_async(self.__reader, (in_stream, Q_write, False, max_games))

        jobs = []
        for _ in range(self.thread_no):
            job = pool.apply_async(self.__process_encode, (Q_read, Q_write, in_tran, verbose))
            jobs.append(job)

        # for job in jobs: job.get()

        Q_write.put('kill')
        pool.close()
        pool.join()

    def decode(self, in_stream: io.TextIOWrapper,
                out_stream: io.TextIOWrapper, out_tran: TransformOut=None,
                max_games=np.inf, verbose=False):

        # file_path = in_stream.name
        # file_size = os.path.getsize(file_path)

        ub_games = max_games
        if ub_games is None: ub_games = np.inf

        manager = multiprocessing.Manager()
        Q_read = manager.Queue()    
        Q_write = manager.Queue()
        pool = multiprocessing.Pool(multiprocessing.cpu_count() + 2)

        writer = pool.apply_async(self.__writer, (out_stream, Q_read, False, ub_games))
        reader = pool.apply_async(self.__reader, (in_stream, Q_write, True, ub_games))

        jobs = []
        for _ in range(self.thread_no):
            job = pool.apply_async(self.__process_decode, (Q_read, Q_write, out_tran, verbose))
            jobs.append(job)

        Q_write.put('kill')
        pool.close()
        pool.join()


