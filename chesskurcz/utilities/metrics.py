import chess
import numpy as np
import chess.pgn
from abc import abstractmethod
from typing import Dict

class Metric:

    def __init__(self, name: str) -> None:
       self.name = name 

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return self.name

    @abstractmethod
    def update(self) -> None:

        pass

    @abstractmethod
    def calculate(self):
        pass

    def result(self):
        pass

class AvgMoveNumberInPosition(Metric):

    def __init__(self, name: str = "avg_moves_in_position") -> None:
        super().__init__(name)

        self.sum = 0
        self.count = 0
    
    def update(self, board: chess.Board, move: chess.Move) -> None:

        self.sum += self.calculate(board, move)
        self.count += 1

    def result(self) -> float:
        if self.count == 0: 
            return 0

        return self.sum / self.count
    
    def calculate(self, board: chess.Board, move: chess.Move) -> float:
        return board.legal_moves.count()

class MaxMoveNumberInPosition(Metric):

    def __init__(self, name: str = 'max_moves_in_position') -> None:
        super().__init__(name)

        self.max_moves_in_position = -1

    def update(self, board: chess.Board, move: chess.Move) -> None:

        self.max_moves_in_position = max(self.max_moves_in_position, MaxMoveNumberInPosition.calculate(board, move))
    
    def result(self):
        return self.max_moves_in_position

    @staticmethod    
    def calculate(board: chess.Board, move: chess.Move):
        return board.legal_moves.count()


class FenExtractor(Metric):

    def __init__(self, name: str = 'fen') -> None:
        super().__init__(name)

        self.fens = []

    def update(self, board: chess.Board, move: chess.Move) -> None:
        self.fens.append(FenExtractor.calculate(board, move))

    @staticmethod    
    def calculate(board: chess.Board, move: chess.Move) -> str:
        return board.fen()
    
    def result(self):
        return self.fens

    
class UciExtractor(Metric):

    def __init__(self, name: str = 'uci') -> None:
        super().__init__(name)
        self.ucis = []
    
    def update(self, board: chess.Board, move: chess.Move) -> None:
        self.ucis.append(UciExtractor.calculate(board, move))

    @staticmethod
    def calculate(board: chess.Board, move: chess.Move) -> str:
        return str(move)
    
    def result(self):
        return self.ucis

class EmptySquaresNumMoves(Metric):

    def __init__(self, name: str = 'empty_square_num_moves') -> None:
        super().__init__(name)

        self.num_moves = []
        self.num_pieces = []
        self.count = 0
        self.stats = {'max': None, 'min': None, 'mean': None, 'median': None}
    
    def update(self, board: chess.Board, move: chess.Move) -> None:

        self.calculate(board, move)

    def calculate(self, board: chess.Board, move: chess.Move) -> None:

        self.num_moves.append(board.legal_moves.count())
        self.num_pieces.append(bin(board.occupied).count("1"))
    
    def result(self) -> Dict[str, float]:

        if not self.count:
            return self.stats

        diffs = (64 - np.array(self.num_pieces)) - np.array(self.num_moves)
        self.stats['max'] = np.max(diffs)
        self.stats['min'] = np.min(diffs)
        self.stats['mean'] = np.mean(diffs)
        self.stats['median'] = np.median(diffs)

        return self.stats


class PieceTypeProbability(Metric):

    def __init__(self, name: str = 'piece_type_probability') -> None:
        super().__init__(name)

        self.piece_sums = {piece_name: 0 for piece_name in chess.PIECE_NAMES if piece_name is not None}
        self.count = 0
    
    def update(self, board: chess.Board, move: chess.Move) -> None:
        
        piece_name = chess.piece_name(board.piece_at(move.from_square).piece_type)
        self.count += 1 
        self.calculate(piece_name)

    def calculate(self, piece_name) -> None:
        self.piece_sums[piece_name] += 1

    def result(self) -> Dict[str, float]:
        if not self.count:       return {
            k: v / self.count for k, v in self.piece_sums.items()
        } 