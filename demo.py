from smolagents import CodeAgent, ManagedAgent, LiteLLMModel

from rag import init_rag_agent
from web_search import init_web_search_agent

def main():
    print("Hello from agentic-rag!")

    # Initialize retriever Agent
    vector_store, rag_agent = init_rag_agent()
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
        # verbose=True,
    )

    # Run sample question
    query = input("Ask me anything!: ")
    agent_output = manager_agent.run(query)

    print("Final output:")
    print("==================================")
    print(f"Query: {query}")
    print("Answer: ")
    print(agent_output)


if __name__ == "__main__":
    main()
