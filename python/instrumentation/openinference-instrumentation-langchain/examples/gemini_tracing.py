import os

from langchain_google_genai import ChatGoogleGenerativeAI
from phoenix.otel import register

# Set Environment Variables
os.environ["GEMINI_API_KEY"] = ""
os.environ["PHOENIX_CLIENT_HEADERS"] = "api_key="
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

# Register Phoenix Tracer
# (be sure the openinference-instrumentation-langchain package is installed)
tracer_provider = register(project_name="gemini-responses-demo", auto_instrument=True)
tracer = tracer_provider.get_tracer(__name__)

# Streaming Call
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GEMINI_API_KEY"])
result = llm.stream("What are the usecases of LLMs?")
for x in result:
    print(x.content, end="", flush=True)

llm.invoke("What are the usecases of LLMs?")
