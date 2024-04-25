import streamlit as st

st.set_page_config(
    page_title="RAG powered by DSPy",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("RAG powered by DSPy")

st.sidebar.info("Select a demo above.")

st.markdown(
    """
    ## Description This project introduces a [Streamlit](https://streamlit.io)
    application designed to interface seamlessly with the
    [DSPy](https://github.com/stanfordnlp/dspy) framework by StanfordNLP,
    encapsulated within a [FastAPI](https://github.com/tiangolo/fastapi)
    backend. It offers an intuitive and interactive frontend solution,
    showcasing the capabilities of DSPy through a user-friendly web interface.
    This application's traces are captured by [Arize
    Phoenix](https://github.com/Arize-ai/phoenix) for observability.

    In a notable application of this system, the Retrieval-Augmented Generation
    (RAG) process is performed on the insightful essay "What I Worked On" by
    Paul Graham. 

    ### Info [GitHub](https://github.com/Arize-ai/openinference)
"""
)
