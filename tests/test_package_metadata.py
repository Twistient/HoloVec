from importlib.metadata import version

import holovec


def test_runtime_version_matches_package_metadata() -> None:
    assert holovec.__version__ == version("holovec")
