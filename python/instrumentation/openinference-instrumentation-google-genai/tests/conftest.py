import pytest


def method_case_insensitive(r1, r2):
    return r1.method.lower() == r2.method.lower()


@pytest.fixture(scope="session")
def vcr_config():
    return {
        "record_mode": "once",
        "match_on": ["scheme", "host", "port", "path", "query", "method"],
        "custom_matchers": {
            "method": method_case_insensitive,
        },
    }
