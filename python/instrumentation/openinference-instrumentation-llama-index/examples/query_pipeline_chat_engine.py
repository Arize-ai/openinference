from typing import Any, Dict, List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_pipeline import (
    ArgPackComponent,
    CustomQueryComponent,
    InputComponent,
    QueryPipeline,
)
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import BeautifulSoupWebReader
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:4317"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

reader = BeautifulSoupWebReader()
documents = reader.load_data(["https://docs.anthropic.com/claude/docs/tool-use"])
lines = documents[0].text.split("\n")

# remove sections with more than two empty lines in a row
fixed_lines = [lines[0]]
for idx in range(1, len(lines)):
    if lines[idx].strip() == "" and lines[idx - 1].strip() == "":
        continue
    fixed_lines.append(lines[idx])

documents[0].text = "\n".join(fixed_lines)


index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model="text-embedding-3-large", embed_batch_size=256),
)


# First, we create an input component to capture the user query
input_component = InputComponent()

# Next, we use the LLM to rewrite a user query
rewrite = (
    "Please write a query to a semantic search engine using the current conversation.\n"
    "\n"
    "\n"
    "{chat_history_str}"
    "\n"
    "\n"
    "Latest message: {query_str}\n"
    'Query:"""\n'
)
rewrite_template = PromptTemplate(rewrite)
llm = OpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.2,
)

# we will retrieve two times, so we need to pack the retrieved nodes into a single list
argpack_component = ArgPackComponent()

# using that, we will retrieve...
retriever = index.as_retriever(similarity_top_k=6)

# then postprocess/rerank with Colbert
reranker = LLMRerank(top_n=3)

# then lastly, we need to create a response using the nodes AND chat history

DEFAULT_CONTEXT_PROMPT = (
    "Here is some context that may be relevant:\n"
    "-----\n"
    "{node_context}\n"
    "-----\n"
    "Please write a response to the following question, using the above context:\n"
    "{query_str}\n"
)


class ResponseWithChatHistory(CustomQueryComponent):
    llm: OpenAI = Field(..., description="OpenAI LLM")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use for the LLM"
    )
    context_prompt: str = Field(
        default=DEFAULT_CONTEXT_PROMPT,
        description="Context prompt to use for the LLM",
    )

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        # NOTE: this is OPTIONAL but we show you where to do validation as an example
        return input

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        # NOTE: These are required inputs. If you have optional inputs please override
        # `optional_input_keys_dict`
        return {"chat_history", "nodes", "query_str"}

    @property
    def _output_keys(self) -> set:
        return {"response"}

    def _prepare_context(
        self,
        chat_history: List[ChatMessage],
        nodes: List[NodeWithScore],
        query_str: str,
    ) -> List[ChatMessage]:
        node_context = ""
        for idx, node in enumerate(nodes):
            node_text = node.get_content(metadata_mode="llm")
            node_context += f"Context Chunk {idx}:\n{node_text}\n\n"

        formatted_context = self.context_prompt.format(
            node_context=node_context, query_str=query_str
        )
        user_message = ChatMessage(role="user", content=formatted_context)

        chat_history.append(user_message)

        if self.system_prompt is not None:
            chat_history = [ChatMessage(role="system", content=self.system_prompt)] + chat_history

        return chat_history

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        chat_history = kwargs["chat_history"]
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context = self._prepare_context(chat_history, nodes, query_str)

        response = llm.chat(prepared_context)

        return {"response": response}

    async def _arun_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the component asynchronously."""
        # NOTE: Optional, but async LLM calls are easy to implement
        chat_history = kwargs["chat_history"]
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context = self._prepare_context(chat_history, nodes, query_str)

        response = await llm.achat(prepared_context)

        return {"response": response}


response_component = ResponseWithChatHistory(
    llm=llm,
    system_prompt=(
        "You are a Q&A system. You will be provided with the previous chat history, "
        "as well as possibly relevant context, to assist in answering a user message."
    ),
)

pipeline = QueryPipeline(
    modules={
        "input": input_component,
        "rewrite_template": rewrite_template,
        "llm": llm,
        "rewrite_retriever": retriever,
        "query_retriever": retriever,
        "join": argpack_component,
        "reranker": reranker,
        "response_component": response_component,
    },
    verbose=False,
)

# run both retrievers -- once with the hallucinated query, once with the real query
pipeline.add_link("input", "rewrite_template", src_key="query_str", dest_key="query_str")
pipeline.add_link(
    "input",
    "rewrite_template",
    src_key="chat_history_str",
    dest_key="chat_history_str",
)
pipeline.add_link("rewrite_template", "llm")
pipeline.add_link("llm", "rewrite_retriever")
pipeline.add_link("input", "query_retriever", src_key="query_str")

# each input to the argpack component needs a dest key -- it can be anything
# then, the argpack component will pack all the inputs into a single list
pipeline.add_link("rewrite_retriever", "join", dest_key="rewrite_nodes")
pipeline.add_link("query_retriever", "join", dest_key="query_nodes")

# reranker needs the packed nodes and the query string
pipeline.add_link("join", "reranker", dest_key="nodes")
pipeline.add_link("input", "reranker", src_key="query_str", dest_key="query_str")

# synthesizer needs the reranked nodes and query str
pipeline.add_link("reranker", "response_component", dest_key="nodes")
pipeline.add_link("input", "response_component", src_key="query_str", dest_key="query_str")
pipeline.add_link(
    "input",
    "response_component",
    src_key="chat_history",
    dest_key="chat_history",
)


pipeline_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)


user_inputs = [
    "Hello!",
    "How does tool-use work with Claude-3 work?",
    "What models support it?",
    "Thanks, that what I needed to know!",
]

if __name__ == "__main__":
    for msg in user_inputs:
        # get memory
        chat_history = pipeline_memory.get()

        # prepare inputs
        chat_history_str = "\n".join([str(x) for x in chat_history])

        # run pipeline
        response = pipeline.run(
            query_str=msg,
            chat_history=chat_history,
            chat_history_str=chat_history_str,
        )

        # update memory
        user_msg = ChatMessage(role="user", content=msg)
        pipeline_memory.put(user_msg)
        print(str(user_msg))

        pipeline_memory.put(response.message)
        print(str(response.message))
        print()
