"""Pytest bootstrap configuration.

Ensures the repository root is on ``sys.path`` so the test suite can
import the top-level packages (``quoter`` and ``src``) regardless of how
pytest is invoked (``pytest``, ``python -m pytest``, or from a subdirectory)
and from a clean clone.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
