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

The examples call the real SDK. They follow the [python-sdks RPC example](https://github.com/livekit/python-sdks/blob/main/examples/rpc.py) and the server API sample in that repository's README: mint a token, create and list a room, connect two participants, and `perform_rpc` a `greet` method.

Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`. A local `livekit-server --dev --node-ip 127.0.0.1` accepts `ws://localhost:7880`, `devkey`, and `secret`. `--node-ip 127.0.0.1` keeps the peer connection on localhost.

`examples/in_memory.py` prints the span attributes. `examples/phoenix_local.py` prints the same attributes and also exports them to `http://localhost:6006/v1/traces`. Start Phoenix first if you want the export to land.
