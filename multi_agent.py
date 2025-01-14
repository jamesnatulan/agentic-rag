from smolagents import CodeAgent, ManagedAgent

from src.rag import (
    init_rag_agent,
    init_chroma_vector_store,
    load_pdf,
    load_dataset,
)
from src.web_search import init_web_search_agent
import streamlit as st

from src.common import load_model


@st.cache_resource
def st_init_chroma_vector_store():
    # Load the vector store
    vector_store, splitter = init_chroma_vector_store()
    return vector_store, splitter


@st.cache_resource
def init_agentic_rag(provider=None, model_id=None, api_key=None, api_base=None):
    # Load the model
    model = load_model(
        provider=provider,
        model_id=model_id,
        api_key=api_key,
        api_base=api_base,
    )
    # Initialize retriever Agent
    rag_agent = init_rag_agent(model)
    rag_agent = ManagedAgent(
        rag_agent,
        name="retriever_agent",
        description="""Uses a retriever tool to retrieve information from a vector database of related documents
        using semantic search. 
        """,
        additional_prompting="Use the retriever tool to find the information you need.",
    )

    # Initialize web search Agent
    web_search_agent = init_web_search_agent(model)
    web_search_agent = ManagedAgent(
        web_search_agent,
        name="web_search_agent",
        description="""Runs web searches only to append, verify, or fill in missing information from
        a generated response from the retriever_agent. Only run the whole input query to this agent if the
        retriever_agent does not return the desired information.
        """,
    )

    # Create the manager agent
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[rag_agent, web_search_agent],
        max_steps=5,
        verbose=False,
    )

    return manager_agent


def main():
    # Start streamlit app
    st.title("Multi Agent RAG Demo")

    # Initialize the vector store
    vector_store, splitter = st_init_chroma_vector_store()

    # Documents here
    st.sidebar.header("Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF", type=["pdf"], accept_multiple_files=True
    )
    if len(uploaded_files) > 0:
        docs = load_pdf(uploaded_files, splitter)
        progress_text = "Loading PDF into vector store..."
        progress_bar = st.sidebar.progress(0, text=progress_text)

        for i, doc in enumerate(docs):
            vector_store.add_documents([doc])
            progress_bar.progress(i / len(docs))
        progress_bar.empty()
        st.sidebar.success("PDFs loaded successfully.")

    dataset = st.sidebar.text_input(
        "Enter dataset name", "jamesnatulan/transformers-docs"
    )
    if st.sidebar.button("Load dataset"):
        docs = load_dataset(dataset, splitter)
        progress_text = "Loading dataset into vector store..."
        progress_bar = st.sidebar.progress(0, text=progress_text)

        for i, doc in enumerate(docs):
            vector_store.add_documents([doc])
            progress_bar.progress(i / len(docs), text=progress_text)
        progress_bar.empty()
        st.sidebar.success("Dataset loaded successfully.")

    # Initialize the Agentic-RAG agent
    st.sidebar.header("LLM Model Configuration")
    provider = st.sidebar.selectbox(
        "Select model provider",
        ["huggingface", "ollama", "openai"],
        index=0,
        help="Choose the model provider you want to use. HuggingFace uses the HuggingFace API, while ollama uses local models through Ollama",
    )

    st.sidebar.divider()
    if provider == "ollama":
        model_id = st.sidebar.text_input(
            "Enter model ID",
            "qwen2.5-coder:7b",
            help="The model name listed in ollama. Run 'ollama list' to see available models.",
        )
        api_base = st.sidebar.text_input(
            "Enter API base",
            "http://localhost:11434",
            help="The base URL of the ollama API.",
        )
        api_key = None
    elif provider == "huggingface":
        model_id = st.sidebar.text_input(
            "Enter model ID",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            help="The model ID from HuggingFace.",
        )
        api_base = None
        api_key = st.sidebar.text_input(
            "Enter token", "", help="Your HuggingFace token."
        )
    elif provider == "openai":
        model_id = st.sidebar.text_input(
            "Enter model ID", "gpt-4", help="The model to use from OpenAI."
        )
        api_base = "https://api.openai.com/v1"
        api_key = st.sidebar.text_input(
            "Enter API key", "", help="Your OpenAI API key."
        )

    agentic_rag = init_agentic_rag(provider, model_id, api_key, api_base)
    st.sidebar.divider()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    prompt = st.chat_input("Enter query: ")
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Run the manager agent with the user input
            prompt_template = """
                Chat history:
                {history}

                User input: 
                {query}
            """
            response = agentic_rag.run(
                prompt_template.format(history=st.session_state.messages, query=prompt),
                stream=False,
            )
            st.write(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
