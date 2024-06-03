from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

if __name__ == "__main__":
    response_gen = llm.stream_chat([ChatMessage(content="hello")])
    for response in response_gen:
        print(response, end="")
