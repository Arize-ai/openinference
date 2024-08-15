"""
This is a dummy test to ensure that every package has one test.
"""

from openinference.semconv.version import __version__ as semconv_version


def test_version() -> None:
    print(semconv_version)
