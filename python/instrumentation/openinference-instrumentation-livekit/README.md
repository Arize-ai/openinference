# OpenInference LiveKit Instrumentation

Traces the [LiveKit Python SDK](https://github.com/livekit/python-sdks). `Room.connect`, `Room.disconnect`, and `LocalParticipant.perform_rpc` come from the `livekit` package. `TwirpClient.request` comes from `livekit-api`. Install either package, or both. Calls from a package that is not installed are left alone.

`Room.connect` records the WebSocket URL. The access token is not recorded. `TwirpClient.request` records the service and method. Request headers are not recorded.

## Install

```shell
pip install openinference-instrumentation-livekit livekit livekit-api
```

## Instrument

```python
from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.livekit import LiveKitInstrumentor

LiveKitInstrumentor().instrument(tracer_provider=TracerProvider())
```

## Examples

`examples/in_memory.py` replays the four calls against stand-ins and prints the span attributes. It does not connect to LiveKit.

`examples/phoenix_local.py` prints the same attributes and also exports them to `http://localhost:6006/v1/traces`. Start Phoenix first if you want the export to land.
