import io, os, sys
import re
from typing import Callable, List, Dict
import chess.pgn

def game_from_uci_to_pgn(game) -> List[str]:

    out = []

    if type(game) == list:

        _g = ' '.join(game)
        g = chess.pgn.read_game(io.StringIO(_g))

    if type(game) == str:
        
        g = chess.pgn.read_game(io.StringIO(game))

    if type(game) == chess.pgn.Game:
        
        g = game

    while g is not None:
        g = g.next()
        out.append(str(g.move))

    return out + [g.game().headers["Result"]]
    
def game_from_pgn_to_uci(game) -> List[str]:

    out = []

    if type(game) == list:

        _g = ' '.join(game)
        g = chess.pgn.read_game(io.StringIO(_g))

    if type(game) == str:

        _g = game
        g = chess.pgn.read_game(io.StringIO(_g))
        
    if type(game) == chess.pgn.Game:
        g = game


    result = g.game().headers["Result"]
    g = g.next()

    while g is not None:

        out.append(g.move.uci())        
        g = g.next()
    
    return out + [result]

class TransformIn:

    def __init__(self, move_extractor: Callable[[str], List[List[str]]],
                 drop_pattern: re.Pattern=None) -> None:
        
        self.drop_patterns = drop_pattern
        self.move_extractor = move_extractor

    def transform(self, _in: str) -> List[List[str]]:

        if self.drop_patterns is not None:
            out = re.sub(self.drop_patterns, '', _in)
        
        return self.move_extractor(out)

class TransformOut:

    def __init__(self, move_repr: Callable=None, sep_games='\n', sep_moves=' ') -> None:
        
        self.id_fun = lambda m: m

        if move_repr is not None:
            self.move_representation = move_repr
        else: self.move_representation = self.id_fun
        self.sep_moves = sep_moves
        self.sep_games = sep_games

    def transform(self, _in: List[List[str]]) -> str:

        _out = []
        for g in _in:
            _out.append(self.move_representation(g))
        
        return self.sep_games.join([self.sep_moves.join(g) for g in _out]) + self.sep_games