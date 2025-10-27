"""Simple test to verify basic functionality"""
import pytest

def test_basic():
    """Just check that tests run"""
    assert True

@pytest.mark.asyncio
async def test_async_basic():
    """Check async tests work"""
    import asyncio
    await asyncio.sleep(0.001)
    assert True
