import io, os, sys
import re
from typing import Callable, List, Dict

class Transform:

    def __init__(self, move_extractor: Callable[[str], List[List[str]]],
                 drop_pattern: re.Pattern=None) -> None:
        
        self.drop_patterns = drop_pattern
        self.move_extractor = move_extractor

    def transform(self, _in: str) -> List[List[str]]:

        if self.drop_patterns is not None:
            out = re.sub(self.drop_patterns, '', _in)
        
        return self.move_extractor(out)