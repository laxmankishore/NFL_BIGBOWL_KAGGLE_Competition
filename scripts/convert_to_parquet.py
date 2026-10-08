"""One-time conversion of the 9 official CSVs to data/parquet/.

Usage: python scripts/convert_to_parquet.py [--overwrite]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ALL_TABLES  # noqa: E402
from src.io import check_raw_files, convert_one  # noqa: E402

if __name__ == "__main__":
    overwrite = "--overwrite" in sys.argv
    check_raw_files()
    for name in ALL_TABLES:
        t0 = time.time()
        n = convert_one(name, overwrite=overwrite)
        print(f"{name:28s} {n:>10,d} rows  ({time.time() - t0:5.1f}s)")
