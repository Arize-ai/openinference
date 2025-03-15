# OpenInference AWS Bedrock Agent Instrumentation

Python autoinstrumentation library for AWS Bedrock Agent calls made using `boto3`.

This package implements OpenInference tracing for `invoke_agent` calls made using a `boto3` `bedrock-agent-runtime` client. These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize `phoenix`](https://github.com/Arize-ai/phoenix).


## Installation

```shell
pip install openinference-instrumentation-bedrock-agent
```


In a notebook environment (`jupyter`, `colab`, etc.) install `openinference-instrumentation-bedrock`, `arize-phoenix` and `boto3`.

[You can test out this quickstart guide in Google Colab!](https://colab.research.google.com/github/Arize-ai/phoenix/blob/main/tutorials/integrations/bedrock_tracing_tutorial.ipynb)

```shell
pip install openinference-instrumentation-bedrock-agent arize-phoenix boto3
```

Ensure that `boto3` is [configured with AWS credentials](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

First, import dependencies required to autoinstrument AWS Bedrock and set up `phoenix` as an collector for OpenInference traces.

```python
from urllib.parse import urljoin

import boto3
import phoenix as px

from openinference.instrumentation.bedrock_agent import BedrockAgentInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
```

Next, we'll start a `phoenix` server and set it as a collector.

```python
px.launch_app()
session_url = px.active_session().url
phoenix_otlp_endpoint = urljoin(session_url, "v1/traces")
phoenix_exporter = OTLPSpanExporter(endpoint=phoenix_otlp_endpoint)
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=phoenix_exporter))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)
```

Instrumenting `boto3` is simple:

```python
BedrockAgentInstrumentor().instrument()
```

Now, all calls to `invoke_model` are instrumented and can be viewed in the `phoenix` UI.

```python
session = boto3.session.Session()
client = session.client("bedrock-agent-runtime")
agent_id = '1CF333B9DE'
agent_alias_id = 'CHNHBWEEGD'
session_id = ''
input_text="What is the good time to visit the Taj Mahal?",
response = client.invoke_agent(
    inputText="What is the good time to visit the Taj Mahal?",
    agentId=agent_id,
    agentAliasId=agent_alias_id,
    sessionId=session_id,
    enableTrace=True
)
for event in response['completion']:
    if 'chunk' in event:
        # Output will be retunred in chunk
        chunk_data = event['chunk']
        if 'bytes' in chunk_data:
            output_text = chunk_data['bytes'].decode('utf8')
            print(output_text)
    elif 'trace' in event:
        # Traces will be retunred in event if traces enabled.
        print(event['trace'])
```

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
* [More info on Bedrock Agents](https://aws.amazon.com/bedrock/agents/)
* [More info on invoke_agent boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html)