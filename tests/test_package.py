"""The package imports and carries its version."""

import tomllib
from pathlib import Path

import froude


def test_the_package_version_matches_pyproject():
    meta = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert froude.__version__ == meta["project"]["version"]
