import io, os, sys

from src.algorithms.transform import TransformIn, TransformOut
from src.algorithms.utils import standard_png_move_extractor
from src.stats import Stats

import src.algorithms.rank, src.algorithms.naive
from src.algorithms import apm
import multiprocessing
from multiprocessing import Value, Array
from typing import Callable, List, Dict

from src import logger

import importlib
import numpy as np
import re

import ctypes

class Encoder:

    def __init__(self, alg='apm', par_workers=1, batch_size=1e4) -> None:
        
        self.alg = alg
        self.par_workers = par_workers
        self.batch_size = batch_size
        self.thresh_for_mult_threads = 1e6

        self.__THRASH_REGEX = re.compile(r'(\?|\!|\{[^{}]*\}|\[[^\[\]]*\]|\n|\#|\+)')

        self.def_pgn_parser = TransformIn(standard_png_move_extractor, self.__THRASH_REGEX)
        self.def_out_format = TransformOut() # default output format transform

        self.__bytes_read_tot = Value(ctypes.c_int32, 0)
        self.__games_cnt = Value(ctypes.c_int32, 0)

        self.dest_lock = multiprocessing.Lock()
        self.source_lock = multiprocessing.Lock()

        self.module_alg = importlib.import_module('src.algorithms.' + alg)

        self.current_file = None

        self.curr_descriptor = None

    def __reader(self, path: str, Q: multiprocessing.Queue, binary=False, max_games=np.inf, verbose=False):
        _m = 'r'
        if binary: _m = 'rb'

        file_size = os.path.getsize(path)

        if verbose:
            # print progress bar
            logger.printProgressBar(0, file_size, prefix='Progress', suffix='Complete')

        with open(path, _m) as in_stream:

            while 1:

                if not Q.qsize() < 5: continue
                
                if binary:

                    read_games = getattr(self.module_alg, 'read_games_' + self.alg)
                    enc_data, g_no = read_games(in_stream, self.batch_size, max_games)

                    if not enc_data:
                        break

                    Q.put(tuple([enc_data, g_no]))
                    with self.__bytes_read_tot.get_lock():
                        self.__bytes_read_tot.value += len(enc_data)

                else: 
                    d = ''.join(in_stream.readlines(self.batch_size))
                    if not d: break
                    Q.put(d)
                
                    with self.__bytes_read_tot.get_lock():
                        self.__bytes_read_tot.value += len(d)

                if verbose:
                    with self.__bytes_read_tot.get_lock():
                        logger.printProgressBar(
                            self.__bytes_read_tot.value, file_size, prefix='Progress', suffix='Complete'
                        )
        
        for _ in range(self.par_workers): Q.put('kill')

    def __writer(self, path: str, Q: multiprocessing.Queue, binary=False, max_games=np.inf, verbose=False):
        
        _f = 'w'
        if binary: _f = 'wb'

        file_size = os.path.getsize(self.current_file)

        with open(path, _f) as f:
            while 1:
                d = Q.get()
                if d == 'kill': break

                f.write(d)            

    def __process_encode(self, Q_data: multiprocessing.Queue,
                        Q_enc: multiprocessing.Queue, in_tran: TransformIn=None,
                        verbose=False):

        parser = in_tran if in_tran is not None else self.def_pgn_parser

        moves = 0

        while moves is not None:
            
            data = Q_data.get()

            if data == 'kill': break

            moves = parser.transform(data)
            _encoder: Callable[[List[List[str]]], bytes] = getattr(self.module_alg, 'encode_' + self.alg)

            enc_data = _encoder(moves)
            Q_enc.put(enc_data)

    def __process_decode(self, Q_enc: multiprocessing.Queue,
                        Q_dec: multiprocessing.Queue, out_tran: TransformOut=None,
                        verbose=False, max_games=np.inf):

        concatenator = out_tran if out_tran is not None else self.def_out_format

        while 1:
            
            d = Q_enc.get()
            if d == 'kill': break

            enc_data, g_no = d

            _decoder: Callable[[bytes], List[List[str]]] = getattr(self.module_alg, 'decode_' + self.alg)
            dec_data = _decoder(enc_data)
            dec_data_str = concatenator.transform(dec_data)

            Q_dec.put(dec_data_str)

    def encode(self, in_stream: str,
                out_stream: str, in_tran: TransformIn=None, 
                max_games=None, verbose=False):

        self.current_file = in_stream

        self.__bytes_read_tot.value = 0
        if verbose:
            print('Compressing file', in_stream)

        ub_games = max_games
        if ub_games is None: ub_games = np.inf

        Q_data = multiprocessing.Queue()    
        Q_enc = multiprocessing.Queue()

        writer = multiprocessing.Process(target=self.__writer, args=(out_stream, Q_enc, True, max_games, verbose))
        reader = multiprocessing.Process(target=self.__reader, args=(in_stream, Q_data, False, max_games, verbose))

        reader.start()
        writer.start()

        workers: List[multiprocessing.Process] = []
        for _ in range(self.par_workers):
            workers.append(multiprocessing.Process(target=self.__process_encode, args=(Q_data, Q_enc, in_tran, verbose)))
            workers[-1].start()

        reader.join()

        for worker in workers: worker.join()
        
        Q_enc.put('kill')

        writer.join()

        self.current_file = None

    def decode(self, in_stream: str,
                out_stream: str, out_tran: TransformOut=None,
                max_games=np.inf, verbose=False):

        self.current_file = in_stream

        self.__bytes_read_tot.value = 0
        ub_games = max_games
        if ub_games is None: ub_games = np.inf

        Q_dec = multiprocessing.Queue()    
        Q_enc = multiprocessing.Queue()

        writer = multiprocessing.Process(target=self.__writer, args=(out_stream, Q_dec, False, ub_games, verbose))
        reader = multiprocessing.Process(target=self.__reader, args=(in_stream, Q_enc, True, ub_games, verbose))

        reader.start()
        writer.start()

        if verbose:
            print('Decompressing file', in_stream)
        workers: List[multiprocessing.Process] = []
        for _ in range(self.par_workers):
            workers.append(multiprocessing.Process(target=self.__process_decode, args=(Q_enc, Q_dec, out_tran, verbose)))
            workers[-1].start()

        reader.join()

        for w in workers: w.join()

        Q_dec.put('kill')

        writer.join()

        self.current_file = None

    def decode_batch_of_games(self, path: str, output_path: str, N: int, out_tran: TransformOut, verbose: bool):

        if self.curr_descriptor is not None:

            if path != self.curr_descriptor.name:

                if not self.curr_descriptor.closed: self.curr_descriptor.close()

            else: self.curr_descriptor = open(path, 'rb')

        else: self.curr_descriptor = open(path, 'rb')

        read_games = getattr(self.module_alg, 'read_games_' + self.alg)
        enc_data, g_no = read_games(self.curr_descriptor, self.batch_size, N)

        Q_dec = multiprocessing.Queue()    
        Q_enc = multiprocessing.Queue()

        writer = multiprocessing.Process(target=self.__writer, args=(output_path, Q_dec, False, np.inf, verbose))

        writer.start()

        if verbose:
            print('Decoding', g_no , 'of games from file', self.curr_descriptor.name)
        workers: List[multiprocessing.Process] = []
        for _ in range(self.par_workers):
            workers.append(multiprocessing.Process(target=self.__process_decode, args=(Q_enc, Q_dec, out_tran, verbose)))
            workers[-1].start()

        for w in workers: w.join()

        Q_dec.put('kill')

        writer.join()