import os
import subprocess
import sys
from typing import Dict

import pytest

_PROXY_SCRIPT = """
import asyncio

import wrapt

# Confirm the pure-Python ObjectProxy is the one in use, so the test can't pass vacuously.
assert type(wrapt.ObjectProxy).__name__ == "_ObjectProxyMetaType", type(wrapt.ObjectProxy)

from openinference.instrumentation.bedrock import BedrockInstrumentor  # noqa: F401
from openinference.instrumentation.bedrock._proxy import _proxy

seen = []


def callback(v):
    seen.append(v)
    return v


# The callback sees the iterator from `__iter__`, each item, then the closing StopIteration.
iterator = iter([1, 2, 3])
assert list(_proxy(iterator, callback=callback)) == [1, 2, 3]
assert seen[0] is iterator, seen
assert seen[1:4] == [1, 2, 3], seen
assert isinstance(seen[4], StopIteration), seen


async def coro():
    return 42


async def main():
    return await _proxy(coro(), callback=callback)


seen.clear()
assert asyncio.run(main()) == 42
assert seen == [42], seen
"""


@pytest.fixture
def env_without_wrapt_extension() -> Dict[str, str]:
    env = dict(os.environ)
    env["WRAPT_DISABLE_EXTENSIONS"] = "true"
    return env


def test_proxy_works_when_wrapt_c_extension_is_disabled(
    env_without_wrapt_extension: Dict[str, str],
) -> None:
    """Importing the instrumentor and proxying must work with pure-Python wrapt."""
    result = subprocess.run(
        [sys.executable, "-c", _PROXY_SCRIPT],
        env=env_without_wrapt_extension,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
