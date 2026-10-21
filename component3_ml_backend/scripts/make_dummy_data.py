"""
Make the dummy sedan dataset (Member 2's format, invented physics) in data/placeholder_sedan/.

Usage: python scripts/make_dummy_data.py
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

runpy.run_module("component3.data.dummy_generator", run_name="__main__")
