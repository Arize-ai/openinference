"""
Based on the Haystack tutorial at https://haystack.deepset.ai/tutorials/36_building_fallbacks_with_conditional_routing.

Requires the following environment variables:

- OPENAI_API_KEY
- SERPERDEV_API_KEY

To get the latter, create an account at https://serper.dev/.
"""

from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.routers import ConditionalRouter
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.dataclasses import Document
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.haystack import HaystackInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

HaystackInstrumentor().instrument(tracer_provider=tracer_provider)


document_content = """
Munich, the vibrant capital of Bavaria in southern Germany, exudes a perfect
blend of rich cultural heritage and modern urban sophistication. Nestled along
the banks of the Isar River, Munich is renowned for its splendid architecture,
including the iconic Neues Rathaus (New Town Hall) at Marienplatz and the
grandeur of Nymphenburg Palace. The city is a haven for art enthusiasts, with
world-class museums like the Alte Pinakothek housing masterpieces by renowned
artists. Munich is also famous for its lively beer gardens, where locals and
tourists gather to enjoy the city's famed beers and traditional Bavarian
cuisine. The city's annual Oktoberfest celebration, the world's largest beer
festival, attracts millions of visitors from around the globe. Beyond its
cultural and culinary delights, Munich offers picturesque parks like the English
Garden, providing a serene escape within the heart of the bustling metropolis.
Visitors are charmed by Munich's warm hospitality, making it a must-visit
destination for travelers seeking a taste of both old-world charm and
contemporary allure.
"""

documents = [Document(content=document_content)]
prompt_template = """
Answer the following query given the documents.
If the answer is not contained within the documents reply with 'no_answer'
Query: {{query}}
Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}
"""

prompt_builder = PromptBuilder(template=prompt_template)
llm = OpenAIGenerator(model="gpt-3.5-turbo")


prompt_for_websearch = """
Answer the following query given the documents retrieved from the web.
Your answer shoud indicate that your answer was generated from websearch.

Query: {{query}}
Documents:
{% for document in documents %}
  {{document.content}}
{% endfor %}
"""

websearch = SerperDevWebSearch()
prompt_builder_for_websearch = PromptBuilder(template=prompt_for_websearch)
llm_for_websearch = OpenAIGenerator(model="gpt-3.5-turbo")


routes = [
    {
        "condition": "{{'no_answer' in replies[0]}}",
        "output": "{{query}}",
        "output_name": "go_to_websearch",
        "output_type": str,
    },
    {
        "condition": "{{'no_answer' not in replies[0]}}",
        "output": "{{replies[0]}}",
        "output_name": "answer",
        "output_type": str,
    },
]

router = ConditionalRouter(routes)


pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.add_component("router", router)
pipe.add_component("websearch", websearch)
pipe.add_component("prompt_builder_for_websearch", prompt_builder_for_websearch)
pipe.add_component("llm_for_websearch", llm_for_websearch)

pipe.connect("prompt_builder", "llm")
pipe.connect("llm.replies", "router.replies")
pipe.connect("router.go_to_websearch", "websearch.query")
pipe.connect("router.go_to_websearch", "prompt_builder_for_websearch.query")
pipe.connect("websearch.documents", "prompt_builder_for_websearch.documents")
pipe.connect("prompt_builder_for_websearch", "llm_for_websearch")


query = "Where is Munich?"
result = pipe.run(
    {
        "prompt_builder": {"query": query, "documents": documents},
        "router": {"query": query},
    }
)


query = "How many people live in Munich?"
result = pipe.run(
    {
        "prompt_builder": {"query": query, "documents": documents},
        "router": {"query": query},
    }
)
