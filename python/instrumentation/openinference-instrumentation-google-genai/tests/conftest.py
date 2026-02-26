import os
from typing import Any, Dict

import pytest


def _normalize_request(request: Any) -> Any:
    request.headers.clear()
    request.method = request.method.upper()
    return request


def _strip_response_headers(response: Any) -> Any:
    return {**response, "headers": {}}


def _match_method_case_insensitive(r1: Any, r2: Any) -> bool:
    return (r1.method or "").upper() == (r2.method or "").upper()


@pytest.fixture(scope="session")
def vcr_config() -> Dict[str, Any]:
    return {
        "before_record_request": _normalize_request,
        "before_record_response": _strip_response_headers,
        "decode_compressed_response": True,
        "record_mode": "once",
        "match_on": ["method_case_insensitive", "scheme", "host", "port", "path", "query"],
        "custom_matchers": {"method_case_insensitive": _match_method_case_insensitive},
    }


def pytest_recording_configure(config: Any, vcr: Any) -> None:
    vcr.register_matcher("method_case_insensitive", _match_method_case_insensitive)


@pytest.fixture(scope="session", autouse=True)
def _bridge_google_api_key() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if google_api_key:
            os.environ["GEMINI_API_KEY"] = google_api_key


# Three patches for vcrpy's aiohttp stubs (upstream bug: https://github.com/kevin1024/vcrpy/issues/927).
#
# Fix #1 — cache the stream per response instance.
#   google-genai's SSE reader calls `response.content.readline()` inside a
#   `while True` loop.  The default `content` property creates a brand-new
#   MockStream (asyncio.StreamReader) on every access, so the loop always reads
#   the first SSE line and spins forever.  Caching returns the same reader and
#   lets the loop drain the buffer and exit on EOF.
#
# Fix #2 — no-op MockStream.set_exception.
#   aiohttp calls stream.set_exception(ClientConnectionError) when closing a
#   response.  That poisons the cached stream and makes the next readline()
#   raise instead of returning data.
#
# Fix #3 — normalise old-format cassettes before build_response sees them.
#   Cassettes recorded with older vcrpy versions use a flat layout:
#     { content: "...", status_code: 200, http_version: "HTTP/1.1" }
#   Current vcrpy expects:
#     { body: { string: "..." }, status: { code: 200, message: "OK" } }
#   We patch build_response to silently upgrade old cassettes on the fly so
#   they never need to be manually edited or re-recorded.
#
# Fix #4 — gzip decoding for aiohttp stubs.
#   VCR's httpcore stubs handle gzip, but aiohttp stubs return raw gzip bytes.
#   Decompress gzip in text(), read(), and the content stream.
try:
    import gzip

    import vcr.stubs.aiohttp_stubs as _aiohttp_stubs  # type: ignore[import-untyped]
    from vcr.stubs.aiohttp_stubs import (
        MockClientResponse,
        MockStream,
    )

    def _decompress_body(body: bytes) -> bytes:
        if body and body[:2] == b"\x1f\x8b":
            return gzip.decompress(body)
        return body

    async def _patched_text(self: Any, encoding: str = "utf-8", errors: str = "strict") -> str:
        return _decompress_body(self._body).decode(encoding, errors=errors)

    async def _patched_read(self: Any) -> bytes:
        return _decompress_body(self._body)

    MockClientResponse.text = _patched_text
    MockClientResponse.read = _patched_read

    def _cached_content(self: Any) -> Any:
        if not hasattr(self, "_content_stream_cache"):
            s = MockStream()
            body = _decompress_body(self._body)
            if body:
                s.feed_data(body)
            s.feed_eof()
            self._content_stream_cache = s
        return self._content_stream_cache

    MockClientResponse.content = property(_cached_content)

    def _noop_set_exception(self: Any, exc: Any) -> None:
        pass

    MockStream.set_exception = _noop_set_exception

    _original_build_response = _aiohttp_stubs.build_response

    def _build_response_compat(vcr_request: Any, vcr_response: Any, history: Any) -> Any:
        # Upgrade old-style cassette response dict to the format current vcrpy expects.
        if "status_code" in vcr_response and "status" not in vcr_response:
            vcr_response = dict(vcr_response)
            vcr_response["status"] = {
                "code": vcr_response.pop("status_code"),
                "message": vcr_response.pop("http_version", "OK"),
            }
            body = vcr_response.pop("content", b"")
            if isinstance(body, str):
                body = body.encode()
            vcr_response["body"] = {"string": body}
        return _original_build_response(vcr_request, vcr_response, history)

    _aiohttp_stubs.build_response = _build_response_compat
    # play_responses imports build_response by name at module level, so patch
    # the module attribute that play_responses calls through.
    import vcr.stubs.aiohttp_stubs as _m

    _m.build_response = _build_response_compat
except ImportError:
    pass
