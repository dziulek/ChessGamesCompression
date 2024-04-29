import chess
import chess.pgn
from abc import abstractmethod

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
