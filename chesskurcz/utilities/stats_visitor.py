import chess
import chess.pgn
from typing import Any, Dict, List, Tuple, NamedTuple, Union

from chesskurcz.utilities.metrics import Metric, MaxMoveNumberInPosition 

class StatsVisitor(chess.pgn.BaseVisitor):

    def __init__(self, Game: Any = chess.pgn.Game) -> None:
        super().__init__(Game)

    def begin_game(self) -> None:
        super().begin_game()

        self.statistics: Dict[str, Any] = {} 
        self.metrics: List[Metric] = []

    def add_metric(self, metrics: Union[Metric, List[Metric]]):

        if not isinstance(metrics, list):
            metrics = [metrics]

        self.metrics += metrics
    
    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        super().visit_move(board, move)

        for metric in self.metrics:
            metric.update(board, move)