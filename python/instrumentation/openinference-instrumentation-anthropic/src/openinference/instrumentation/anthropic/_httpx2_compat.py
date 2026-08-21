"""Pytest plugin that aliases ``httpx``/``httpcore`` to ``httpx2``/``httpcore2``.

anthropic>=1 migrated its HTTP stack from ``httpx`` to the ``httpx2`` fork. The test
harness mocks HTTP with ``respx`` and records cassettes with ``vcrpy``, both of which
patch the real ``httpx``. Aliasing ``httpx`` to ``httpx2`` before either is imported
makes them operate on the module anthropic actually uses, so requests are mocked and
recorded as before.

The alias must be installed before ``respx``/``httpx`` are imported, so this is
wired up as a ``-p`` plugin (loaded ahead of entry-point plugins such as respx)
via ``addopts`` in ``pyproject.toml`` rather than as a ``conftest``. With
anthropic<1 (no ``httpx2`` installed) it is a no-op.
"""

try:
    import httpx2

    httpx2.alias_httpx()
except Exception:  # pragma: no cover - anthropic<1 ships no httpx2 to alias
    pass
