import streamlit as st
import pandas as pd
import requests
import json

import os

from dotenv import load_dotenv

load_dotenv()

fastapi_base_url = os.getenv("FASTAPI_BACKEND_URL", "localhost")

# interact with FastAPI endpoint
backend = f"{fastapi_base_url}/api/rag/compile-program"
openai_models_url = f"{fastapi_base_url}/api/rag/list-models"


# Assume this function is available to convert CSV to JSON
def convert_csv_to_json(csv_file):
    df = pd.read_csv(csv_file)
    return df.to_dict(orient="records")


def get_list_models(server_url: str):
    r = requests.get(url=server_url)

    raw_json = r.json()

    return raw_json["models"]


def compile_rag(
    server_url: str,
    trainset_data: dict,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    payload = {
        "items": trainset_data,
        # "chat_history": chat_history,
        "openai_model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    r = requests.post(url=backend, json=payload)

    return r


st.set_page_config(
    page_title="RAG powered by DSPy",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("DSPy Optimizer")
st.markdown(
    """
## Using DSPy Optimizers with Small Training Sets

DSPy, a powerful framework developed by StanfordNLP, provides a suite of tools and functionalities for natural language processing tasks. One of its capabilities includes optimizing models based on a variety of input data. This page focuses on how to use DSPy optimizers with a small dataset, specifically a CSV file containing pairs of questions and answers.

### Preparing Your Data

Your dataset should be in a CSV file with two columns: `question` and `answer`. Here's an example of what your data might look like:

```csv
question,answer
"What is the capital of France?","Paris"
"Who wrote 'To Kill a Mockingbird'?","Harper Lee"
"What is the chemical symbol for water?","H2O"
"Who painted the Mona Lisa?","Leonardo da Vinci"
"What year did the Titanic sink?","1912"

"""
)

# Sidebar config
with st.sidebar:
    st.subheader("Models and parameters")
    selected_model = st.sidebar.selectbox(
        "Choose an Ollama model available on your system",
        get_list_models(server_url=openai_models_url),
        key="selected_model",
    )
    llm = selected_model
    temperature = st.sidebar.slider(
        "temperature", min_value=0.01, max_value=5.0, value=0.1, step=0.01
    )
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_tokens = st.sidebar.slider("max_tokens", min_value=32, max_value=150, value=120, step=8)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Convert the uploaded file to JSON
    json_data = convert_csv_to_json(uploaded_file)

    # Adjust the data structure to match FastAPI's expected input
    json_data_to_send = {"items": json_data}

    # Display the JSON data (optional)
    st.json(json_data_to_send)

    # Button to send data to FastAPI
    if st.button("Compile your program."):
        # Here you need to match the endpoint and data format expected by your FastAPI backend
        response = compile_rag(
            server_url=backend,
            trainset_data=json_data,
            model_name=llm,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if response.status_code == 200:
            st.success("Successfully compiled RAG program!")
            # Parse the response JSON using Pydantic models (if the response contains data to be displayed)
            # For example, if your FastAPI returns the processed data back
            # processed_data = [QAItem(**item) for item in response.json()['processed']]
            # for item in processed_data:
            #     st.write(f"Question: {item.question}, Answer: {item.answer}")
        else:
            st.error("Failed to upload data.")
