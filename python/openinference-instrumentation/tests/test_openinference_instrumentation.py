import pytest
from openinference.instrumentation import end_session, start_session, suppress_tracing
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    Context,
    attach,
    get_value,
    set_value,
)


class TestSuppressTracing:
    def test_instrumentation_is_suppressed_inside_context(self) -> None:
        with suppress_tracing():
            assert get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True

    def test_previous_attribute_value_is_restored_after_exiting_context(self) -> None:
        previous_token = object()
        attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, previous_token))
        assert get_value(_SUPPRESS_INSTRUMENTATION_KEY) is previous_token
        with suppress_tracing():
            pass
        assert get_value(_SUPPRESS_INSTRUMENTATION_KEY) is previous_token


class TestStartSession:
    def test_session_id_is_added_to_context(self) -> None:
        assert get_value("session.id") is None
        token = start_session("session-id")
        assert token is not None
        assert get_value("session.id") == "session-id"

    def test_no_effect_and_logs_warning_if_session_exists(self) -> None:
        start_session("existing-session-id")
        assert get_value("session.id") == "existing-session-id"
        with pytest.warns(UserWarning):
            start_session("new-session-id")
        assert get_value("session.id") == "existing-session-id"


class TestEndSession:
    def test_active_session_id_is_removed_from_context(self) -> None:
        token = start_session("session-id")
        assert get_value("session.id") == "session-id"
        end_session(token)
        assert get_value("session.id") is None

    def test_no_effect_when_no_session_is_active(self) -> None:
        assert get_value("session.id") is None
        end_session(object())
        assert get_value("session.id") is None


@pytest.fixture(autouse=True)
def reset_context() -> None:
    attach(Context())
