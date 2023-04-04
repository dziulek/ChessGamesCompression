import io, os, sys
import re
from typing import Callable, List, Dict

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

    def __init__(self, move_extractor: Callable[[str], List[List[str]]]=None,
                 move_repr: Callable[[List[List[str]]], str]=None) -> None:
        
        self.move_representation = move_repr
        self.move_extractor = move_extractor

    def transform(self, _in: str) -> List[List[str]]:
        
        return _in