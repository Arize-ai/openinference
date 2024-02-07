# LangChain specific span attributes

LC_METADATA_PREFIX = "langchain.metadata"
"""
In LangChain you can attach metadata to a chain. Each of the metadata keys will be prefixed with
`langchain.metadata` and will be added to the span attributes.
https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html#langchain.chains.llm.LLMChain.metadata
"""
