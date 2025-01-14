from smolagents import CodeAgent, ManagedAgent, LiteLLMModel

from rag import init_rag_agent
from web_search import init_web_search_agent

import streamlit as st

@st.cache_resource
def init_agentic_rag():
    # Initialize retriever Agent
    _, rag_agent = init_rag_agent()
    rag_agent = ManagedAgent(
        rag_agent,
        name="retriever_agent",
        description="""Use this agent first to check and retrieve information from the knowledge base. If you have
        missing information, you can use the web search agent to fill in the gaps.""",
    )

    # Initialize web search Agent
    web_search_agent = init_web_search_agent()
    web_search_agent = ManagedAgent(
        web_search_agent,
        name="web_search",
        description="""Runs web searches only to append, verify, or fill in missing information from
        a generated response from the retriever agent""",
    )

    # Initialize the model
    model = LiteLLMModel(
        model_id="ollama/qwen2.5-coder:1.5b",
        api_base="http://localhost:11434",
        api_key=None,
    )

    # Create the manager agent
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[rag_agent, web_search_agent],
        max_steps=5,
        additional_authorized_imports=["time", "numpy", "pandas"],
        verbose=False,
    )

    return manager_agent


def main():
    print("Hello from agentic-rag!")

    agentic_rag = init_agentic_rag()

    # Start streamlit app
    st.title("Agentic-RAG Demo")

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
