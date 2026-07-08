"""
Entry point: python main.py [preset]

Presets are RunConfig factories in config.PRESETS ("main" by default);
edit config.py to define new ones or tweak parameters.
"""

import sys

from config import PRESETS
from core.loop import run

if __name__ == "__main__":
    preset = sys.argv[1] if len(sys.argv) > 1 else "main"
    run(PRESETS[preset]())
