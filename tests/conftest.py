import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import generator  # noqa: E402
from utils.cleaning import clean_all  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return generator.generate(seed=42, write=False)


@pytest.fixture(scope="session")
def T(raw):
    cleaned, _ = clean_all({k: v.copy() for k, v in raw.items()})
    return cleaned


@pytest.fixture(scope="session")
def ref(T):
    from utils.loader import data_window
    return data_window(T)[1]


@pytest.fixture(scope="session")
def monthly(T):
    from analytics.kpis import monthly_table
    return monthly_table(T)
