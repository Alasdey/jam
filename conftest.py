# Root conftest so pytest puts the project root on sys.path
# (modules are imported as top-level packages: config, creation, rewards, ...).

from functools import cache

import pytest


@cache
def _cuda_available() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs CuPy and a CUDA device")


def pytest_collection_modifyitems(items):
    # CUDA starts driver threads, and forking a threaded process is unsafe:
    # run GPU tests after every test that forks payoff worker processes.
    items.sort(key=lambda item: item.get_closest_marker("cuda") is not None)


def pytest_runtest_setup(item):
    if item.get_closest_marker("cuda") and not _cuda_available():
        pytest.skip("needs CuPy and a CUDA device")
