import chess
import chess.pgn
from abc import abstractmethod

class Metric:

    def __init__(self) -> None:
       pass 

    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def update(self) -> None:

        pass

    @abstractmethod
    def calculate(self):
        pass

    @abstractmethod
    def result(self):
        pass

class MaxMoveNumberInPosition(Metric):

    def __init__(self) -> None:
        super().__init__()

        self.sum = 0
        self.count = 0
    
    def __repr__(self) -> str:
        return 'max_move_number_in_position'
    
    def update(self, board: chess.Board, move: chess.Move) -> None:

        self.sum += self.calculate(board, move)
        self.count += 1

    def result(self) -> float:

        return self.sum / self.count
    
    def calculate(self, board: chess.Board, move: chess.Move) -> float:

        return len(board.legal_moves)