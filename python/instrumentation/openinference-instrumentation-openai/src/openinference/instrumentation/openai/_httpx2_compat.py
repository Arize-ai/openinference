"""Pytest plugin that aliases ``httpx``/``httpcore`` to ``httpx2``/``httpcore2``.

openai>=3 migrated its HTTP stack from ``httpx`` to the ``httpx2`` fork. The test
harness mocks HTTP with ``respx`` and instruments outgoing requests with
``opentelemetry-instrumentation-httpx``, both of which patch the real ``httpx``.
Aliasing ``httpx`` to ``httpx2`` before either is imported makes them operate on
the module openai actually uses, so requests are mocked and instrumented as before.

The alias must be installed before ``respx``/``httpx`` are imported, so this is
wired up as a ``-p`` plugin (loaded ahead of entry-point plugins such as respx)
via ``addopts`` in ``pyproject.toml`` rather than as a ``conftest``. With
openai<3 (no ``httpx2`` installed) it is a no-op.
"""

try:
    import httpx2

    httpx2.alias_httpx()
except Exception:  # pragma: no cover - openai<3 ships no httpx2 to alias
    pass
