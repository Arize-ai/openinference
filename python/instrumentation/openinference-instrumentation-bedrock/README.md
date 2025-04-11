# OpenInference AWS Bedrock Instrumentation

Python autoinstrumentation library for AWS Bedrock calls made using `boto3`.

This package implements OpenInference tracing for `invoke_model`, `invoke_agent` and `converse` calls made using the `boto3` `bedrock-runtime` and `bedrock-agent-runtime` clients. These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize `phoenix`](https://github.com/Arize-ai/phoenix).

[![pypi](https://badge.fury.io/py/openinference-instrumentation-bedrock.svg)](https://pypi.org/project/openinference-instrumentation-bedrock/)

> [!NOTE]\
> The Converse API was introduced in botocore [v1.34.116](https://github.com/boto/botocore/blob/develop/CHANGELOG.rst). Please use v1.34.116 or above to utilize converse.

## Supported Models

Find the list of Bedrock-supported models and their IDs [here](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns). Future testing is planned for additional models.

| Model                               | Supported Methods    |
| ----------------------------------- | -------------------- |
| `Anthropic Claude 2.0`              | converse, invoke     |
| `Anthropic Claude 2.1`              | converse, invoke     |
| `Anthropic Claude 3 Sonnet 1.0`     | converse             |
| `Anthropic Claude 3.5 Sonnet`       | converse             |
| `Anthropic Claude 3 Haiku`          | converse             |
| `Meta Llama 3 8b Instruct`          | converse             |
| `Meta Llama 3 70b Instruct`         | converse             |
| `Mistral AI Mistral 7B Instruct`    | converse             |
| `Mistral AI Mixtral 8X7B Instruct`  | converse             |
| `Mistral AI Mistral Large`          | converse             |
| `Mistral AI Mistral Small`          | converse             |

## Installation

```shell
pip install openinference-instrumentation-bedrock
```

## Quickstart

> [!IMPORTANT]\
> OpenInference for AWS Bedrock supports both [`invoke_model`](https://botocore.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html#BedrockRuntime.Client.invoke_model) and [`converse`](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html#). For models that use the Messages API, such as Anthropic Claude 3 and Anthropic Claude 3.5, use the [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html) instead.

In a notebook environment (`jupyter`, `colab`, etc.) install `openinference-instrumentation-bedrock`, `arize-phoenix` and `boto3`.

[You can test out this quickstart guide in Google Colab!](https://colab.research.google.com/github/Arize-ai/phoenix/blob/main/tutorials/integrations/bedrock_tracing_tutorial.ipynb)

```shell
pip install openinference-instrumentation-bedrock arize-phoenix boto3
```

Ensure that `boto3` is [configured with AWS credentials](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

First, import dependencies required to autoinstrument AWS Bedrock and set up `phoenix` as an collector for OpenInference traces.

```python
from urllib.parse import urljoin

import boto3
import phoenix as px

from openinference.instrumentation.bedrock import BedrockInstrumentor
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
BedrockInstrumentor().instrument()
```

Now, all calls to `invoke_model` are instrumented and can be viewed in the `phoenix` UI.

```python
session = boto3.session.Session()
client = session.client("bedrock-runtime")
prompt = b'{"prompt": "Human: Hello there, how are you? Assistant:", "max_tokens_to_sample": 1024}'
response = client.invoke_model(modelId="anthropic.claude-v2", body=prompt)
response_body = json.loads(response.get("body").read())
print(response_body["completion"])
```

Alternatively, all calls to `converse` are instrumented and can be viewed in the `phoenix` UI.

```python
session = boto3.session.Session()
client = session.client("bedrock-runtime")

message1 = {
            "role": "user",
            "content": [{"text": "Create a list of 3 pop songs."}]
}
message2 = {
        "role": "user",
        "content": [{"text": "Make sure the songs are by artists from the United Kingdom."}]
}
messages = []

messages.append(message1)
response = client.converse(
    modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=messages
)
out = response["output"]["message"]
messages.append(out)
print(out.get("content")[-1].get("text"))

messages.append(message2)
response = client.converse(
    modelId="anthropic.claude-v2:1",
    messages=messages
)
out = response['output']['message']
print(out.get("content")[-1].get("text"))
```
All calls to `invoke_agent` are instrumented and can be viewed in the `phoenix` UI. You can enable the agent traces by passing `enableTrace=True` argument.

```python
session = boto3.session.Session()
client = session.client("bedrock-agent-runtime")
agent_id = '<AgentId>'
agent_alias_id = '<AgentAliasId>'
session_id = f"default-session1_{int(time.time())}"

attributes = dict(
    inputText="When is a good time to visit the Taj Mahal?",
    agentId=agent_id,
    agentAliasId=agent_alias_id,
    sessionId=session_id,
    enableTrace=True
)
response = client.invoke_agent(**attributes)

for idx, event in enumerate(response['completion']):
    if 'chunk' in event:
        chunk_data = event['chunk']
        if 'bytes' in chunk_data:
            output_text = chunk_data['bytes'].decode('utf8')
            print(output_text)
    elif 'trace' in event:
        print(event['trace'])
```

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)