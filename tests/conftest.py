"""Root pytest configuration ensuring local project root takes precedence in sys.path."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path or sys.path[0] != PROJECT_ROOT:
    if PROJECT_ROOT in sys.path:
        sys.path.remove(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)
