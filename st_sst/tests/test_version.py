"""Smoke test: the package imports and declares a version."""

import st_sst


def test_version():
    assert isinstance(st_sst.__version__, str)
    assert st_sst.__version__
