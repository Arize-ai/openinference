import os
import subprocess
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent.parent

_SCRIPT = """
import sys

import pytest
import wrapt

# The metaclass conflict needs ObjectProxy to carry its own metaclass, as pure-Python wrapt does.
assert type(wrapt.ObjectProxy) is not type, "wrapt's C extension is still in use"
sys.exit(pytest.main(
    ["-q", "-p", "no:cacheprovider", "tests/test_gapic_client.py::test_instrumentor"],
))
"""


def test_instrumentor_without_wrapt_c_extension() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=_PACKAGE_DIR,
        env={**os.environ, "WRAPT_DISABLE_EXTENSIONS": "true"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
