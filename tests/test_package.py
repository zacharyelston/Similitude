"""The package imports and carries its version."""

import tomllib
from pathlib import Path

import similitude


def test_the_package_version_matches_pyproject():
    meta = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert similitude.__version__ == meta["project"]["version"]
