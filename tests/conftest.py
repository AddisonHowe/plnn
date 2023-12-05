import pytest

#####################
##  Configuration  ##
#####################

def pytest_addoption(parser):
    parser.addoption(
        "--benchmark", action="store_true", default=False, 
        help="run benchmarking tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: mark test as benchmarking")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--benchmark"):
        # --benchmark given in cli: do not skip benchmarking tests
        return
    skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
            