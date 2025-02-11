import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, OpenAIServerModel, Tool

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(
    lambda row: row["source"].startswith("huggingface/transformers")
)

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Uses semantic search to retrieve the parts of transformers documentation "
        "that could be most relevant to answer your query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The query to perform. "
                "This should be semantically close to your target documents. "
                "Use the affirmative form rather than a question."
            ),
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=10)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )


retriever_tool = RetrieverTool(docs_processed)
agent = CodeAgent(
    tools=[retriever_tool],
    model=OpenAIServerModel("gpt-4o"),
    max_steps=4,
    verbosity_level=2,
)

agent_output = agent.run(
    "For a transformers model training, which is slower, the forward or the backward pass?"
)

print("Final output:")
print(agent_output)
