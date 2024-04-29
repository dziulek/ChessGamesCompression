import chess
import chess.pgn
from typing import Any, Dict, List, Tuple, NamedTuple, Union

from chesskurcz.utilities.metrics import Metric, AvgMoveNumberInPosition, FenExtractor, UciExtractor, \
    MaxMoveNumberInPosition
                    

class StatsVisitor(chess.pgn.GameBuilder):

    METRICS = set()

    def begin_game(self) -> None:
        super().begin_game()

        self.statistics: Dict[str, Any] = {} 
        self.metrics: List[Metric] = [m() for m in StatsVisitor.METRICS]

    @staticmethod
    def add_metric(metrics: Union[Metric, List[Metric]]):

        if not isinstance(metrics, list):
            metrics = [metrics]

        for metric in metrics:
            StatsVisitor.METRICS.add(metric)
    
    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        super().visit_move(board, move)

        for metric in self.metrics:
            metric.update(board, move)

    def end_game(self) -> None:
        super().end_game()

        for metric in self.metrics:
            self.statistics[str(metric)] = metric.result()

    def result(self) -> Tuple[chess.pgn.Game, Dict[str, Any]]:
        
        return self.game, self.statistics
