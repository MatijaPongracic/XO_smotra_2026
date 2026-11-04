"""
src package initialization.
Automatski postavlja PROJECT_ROOT i SRC_DIR u sys.path
kako bi svi podmoduli i paketi mogli neometano uvoziti module međusobno i config.py iz korijena.
"""

import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

_PATHS_TO_ENSURE = [
    _PROJECT_ROOT,
    _SRC_DIR,
    os.path.join(_SRC_DIR, "gui"),
    os.path.join(_SRC_DIR, "vision"),
    os.path.join(_SRC_DIR, "robot"),
    os.path.join(_SRC_DIR, "game"),
    os.path.join(_SRC_DIR, "web"),
]

for p in _PATHS_TO_ENSURE:
    if p not in sys.path:
        sys.path.insert(0, p)
