# OpenInference crewAI Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-crewai.svg)](https://pypi.org/project/openinference-instrumentation-crewai/)

Python auto-instrumentation library for LLM agents implemented with CrewAI

Crews are fully OpenTelemetry-compatible and can be sent to an OpenTelemetry collector for monitoring, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

## Installation

```shell
pip install openinference-instrumentation-crewai
```

## Quickstart

This quickstart shows you how to instrument your guardrailed LLM application 

Install required packages.

```shell
pip install crewai crewai-tools  arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start Phoenix in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address. (Phoenix does not send data over the internet. It only operates locally on your machine.)

```shell
python -m phoenix.server.main serve
```

Set up `CrewAIInstrumentor` to trace your crew and send the traces to Phoenix at the endpoint defined below.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from openinference.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

CrewAIInstrumentor().instrument(tracer_provider=trace_provider)
```

Set up a simple crew to do research
```python
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY" 
search_tool = SerperDevTool()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  # You can pass an optional llm attribute specifying what model you wanna use.
  # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
  tools=[search_tool]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
  expected_output="Full analysis report in bullet points",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
  expected_output="Full blog post of at least 4 paragraphs",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=True,
  process=Process.sequential
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
```

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
