import pytest

from src.config import ALL_TABLES
from src.io import parquet_path

HAVE_PARQUET = all(parquet_path(n).exists() for n in ALL_TABLES)


def pytest_collection_modifyitems(config, items):
    if HAVE_PARQUET:
        return
    skip = pytest.mark.skip(reason="Parquet not built; run scripts/convert_to_parquet.py")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
