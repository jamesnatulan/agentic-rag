from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    Model,
)
from src.common import load_model

def init_web_search_agent(model: Model):
    # Create web search Agent
    web_search_agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model,
        max_steps=10,
        verbose=True,
    )

    return web_search_agent


def main():

    print("Web Search Agent")

    # Initialize model
    model = load_model(
        provider="ollama",
        model_id="qwen2.5-coder:7b",
        api_key=None,
        api_base="http://localhost:11434",
    )

    # Initialize the Web Search Agent
    print("Initializing the Web Search Agent...")
    web_search_agent = init_web_search_agent(model)

    question = input("Enter query!: ")
    response = web_search_agent.run(question)
    print(response)


if __name__ == "__main__":
    main()
