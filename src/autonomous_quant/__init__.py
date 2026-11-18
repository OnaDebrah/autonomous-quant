"""The seed code of *The Autonomous Quant*: a minimal research loop to read, run and extend.

Each module follows a chapter of the book (see README.md). Nothing here is a trading system, and nothing in it claims
to find an edge: its value is that its controls can be shown to work, on worlds where the truth is planted.
"""

__version__ = "0.1.0"

import logging as _logging

_logging.getLogger(__name__).addHandler(_logging.NullHandler())  # the application decides where logs go
