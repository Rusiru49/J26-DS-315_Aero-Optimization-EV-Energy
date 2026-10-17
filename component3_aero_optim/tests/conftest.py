"""
Makes the component's modules importable from the flat tests/ folder without
turning component3_aero_optim into a package (matching the repo's existing
flat-import style rather than introducing __init__.py/relative imports).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
