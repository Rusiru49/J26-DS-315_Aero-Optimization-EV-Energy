"""
Check a data folder against Member 2's contract.

Usage:
    python scripts/check_dataset.py                 # the dummy dataset
    python scripts/check_dataset.py <folder>        # e.g. Member 2's real delivery
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_checker import check_dataset, print_report


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else config.PLACEHOLDER_DATA_DIR
    print(f"Checking {data_dir}\n")
    ok = print_report(check_dataset(data_dir))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
