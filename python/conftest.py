import asyncio
from asyncio.events import BaseDefaultEventLoopPolicy
from typing import Any, Dict

import pytest


@pytest.fixture(scope="session")
def event_loop_policy() -> BaseDefaultEventLoopPolicy:
    try:
        import uvloop
    except ImportError:
        return asyncio.DefaultEventLoopPolicy()
    return uvloop.EventLoopPolicy()


@pytest.fixture
def vcr_config() -> Dict[str, Any]:
    return dict(
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
        decode_compressed_response=True,
        ignore_localhost=True,
    )
