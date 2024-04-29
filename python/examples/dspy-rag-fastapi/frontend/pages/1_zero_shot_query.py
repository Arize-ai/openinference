import os
import requests

import streamlit as st

from typing import List
from dotenv import load_dotenv

load_dotenv()

fastapi_base_url = os.getenv("FASTAPI_BACKEND_URL", "localhost")

# interact with FastAPI endpoint
backend = f"{fastapi_base_url}/api/rag/zero-shot-query"
openai_models_url = f"{fastapi_base_url}/api/rag/list-models"


def zero_shot_query(
    server_url: str,
    query: str,
    # chat_history: List[dict],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    payload = {
        "query": query,
        # "chat_history": chat_history,
        "vendor_model": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    r = requests.post(url=server_url, json=payload)

    return r.json()

    # return payload


def get_list_models(server_url: str):
    r = requests.get(url=server_url)

    raw_json = r.json()

    return raw_json["models"]


st.set_page_config(
    page_title="RAG powered by DSPy",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Sidebar config
with st.sidebar:
    st.title("Zero Shot Query")
    st.info(
        "Perform zero-shot query on Paul Graham’s essay “What I Worked On”. ",
        icon="📃",
    )

    st.subheader("Models and parameters")
    selected_model = st.sidebar.selectbox(
        "Choose an OpenAI model available on your system",
        get_list_models(server_url=openai_models_url),
        key="selected_model",
    )
    llm = selected_model
    temperature = st.sidebar.slider(
        "temperature", min_value=0.01, max_value=5.0, value=0.1, step=0.01
    )
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_tokens = st.sidebar.slider("max_tokens", min_value=32, max_value=150, value=120, step=8)


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Paul Graham’s essay!",
        }
    ]


if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = zero_shot_query(
                server_url=backend,
                query=prompt,
                # chat_history=st.session_state.messages,
                model_name=llm,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            st.write(response["answer"])
            message = {"role": "assistant", "content": response["answer"]}
            st.session_state.messages.append(message)  # Add response to message history
