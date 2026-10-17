"""
Makes the component3 package (in ../src) importable for the tests without
installing it. (Alternatively: pip install -e . from the component folder.)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
