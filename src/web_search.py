from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    Model,
)

def init_web_search_agent(model: Model):
    # Create web search Agent
    web_search_agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model,
        max_steps=10,
        # verbose=False,
    )

    return web_search_agent


def main():

    print("Web Search Agent")

    # Initialize the Web Search Agent
    print("Initializing the Web Search Agent...")
    web_search_agent = init_web_search_agent()

    question = input("Enter query!: ")
    response = web_search_agent.run(question)
    print(response)


if __name__ == "__main__":
    main()
