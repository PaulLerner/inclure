# -*- coding: utf-8 -*-
from jsonargparse import CLI
from jsonargparse.typing import register_type
from pathlib import Path
import re

register_type(Path)


extra_spaces = re.compile(r'\s+')
SEP = r"\."
# same as "\w" but does not match digits
LETTER = r"[^\d\W]"
sep_sub = rf"\1.\2"
extra_seps = re.compile(rf'({LETTER})[·•]({LETTER})')


def preproc(text):
    # common preproc: strip extra spaces
    text = extra_spaces.sub(' ', text.strip())
    # keep a single separator : "."
    text = extra_seps.sub(sep_sub, text)
    # hack : run twice in case of things like "tou.s.tes" where "u.s" matching will prevent "s.t" matching
    text = extra_seps.sub(sep_sub, text)
    
    return text