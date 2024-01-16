"""Benchmarking Tests

pytest -s --benchmark tests/benchmark.py

"""

import pytest
import numpy as np
from plnn.model import PLNN
from plnn.dataset import get_dataloaders

TRAINDIR = "data/benchmark_data/benchmark_data_train"
VALIDDIR = "data/benchmark_data/benchmark_data_valid"
NSIMS_TRAIN = 20
NSIMS_VALID = 5

OUTDIR = "tests/benchmark/tmp_out"

@pytest.mark.benchmark
class TestBenchmarkModel:

    def test1(self):
        pass
