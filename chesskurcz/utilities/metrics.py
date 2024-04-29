import chess
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

        self.max_moves_in_position = max(self.max_moves_in_position, self.calculate(board, move))
    
    def result(self):
        return self.max_moves_in_position
    
    def calculate(self, board: chess.Board, move: chess.Move):
        return board.legal_moves.count()


class FenExtractor(Metric):

    def __init__(self, name: str = 'fen') -> None:
        super().__init__(name)

        self.fens = []

    def update(self, board: chess.Board, move: chess.Move) -> None:
        self.fens.append(self.calculate(board, move))
    
    def calculate(self, board: chess.Board, move: chess.Move) -> str:
        return board.fen()
    
    def result(self):
        return self.fens

    
class UciExtractor(Metric):

    def __init__(self, name: str = 'uci') -> None:
        super().__init__(name)
        self.ucis = []
    
    def update(self, board: chess.Board, move: chess.Move) -> None:
        self.ucis.append(self.calculate(board, move))
    
    def calculate(self, board: chess.Board, move: chess.Move) -> str:
        return str(move)
    
    def result(self):
        return self.ucis


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
        return {
            k: v / self.count for k, v in self.piece_sums.items()
        } 