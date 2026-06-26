import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip = pytest.mark.skip(reason="Pass --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
