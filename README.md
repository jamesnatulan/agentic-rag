# Agentic RAG

In this project, I built an Agentic RAG app using Huggingface's [smolagents](https://huggingface.co/docs/smolagents/index) library. The main advantage of this library is that its very easy to use, and it utilizes `CodeAgents` which are agents that use code to express and execute its actions instead of the usual dictionary-like outputs such as JSON. Advantages of using `CodeAgents` are summarized in [this post by Huggingface](https://huggingface.co/blog/beating-gaia#code-agent-%F0%9F%A7%91%F0%9F%92%BB).

There are two available RAG configurations: single-agent and multi-agent. The single-agent one utilizes a single agent that has access to a Retrieval tool. The multi-agent one utilizes 3 agents in total; one is a RAG agent, which has access to a Retrieval tool (this is pretty much the same with the single-agent one), another is a Web Search agent, which handles searching the web for information if needed, and lastly, a Manager agent, which is responsible for figuring out what to do for the given task by the user, and delegates subtasks to the other two agents. 

![architecture diagram](assets/docs/architecture.png)

The rule of thumb is: always go for simplicity! If a single agent one performs well enough, then it probably is the best to use! Multiagent is advantageous on more complex tasks, but for a simple RAG system, such as this one, multiagent is probably overkill. 

## Exploring the App

The app has a simple UI: a sidebar that contains configuration options for the RAG system and documents processing, and a main section that contains the chat history with the agent. The app uses Huggingface's Inference API by default, and the LLM behind the endpoint is the Qwen2.5-32B model, which is free! You can also use OpenAI, just make sure to input your API key. You can also use Ollama but this feature is only for local deployment of the app, which you can know more about in this [section](#deploy-locally).

To upload documents on the vector store, you can use a huggingface dataset, or PDFs. Just select in the Documents section in the sidebar the documents you wish to store. 

A demo of the app is available here: https://agentic-rag-demo.streamlit.app/ but it is advisable to explore the app locally with a machine that has a GPU in it (NVIDIA with CUDA cores). This is because on local, inference and generating embeddings from the documents is a lot faster. Also, you can check out the model's cognitive function step by step on the console when you run locally! 

## Deploy locally

### Python dependencies

This repository uses [UV](https://astral.sh/blog/uv) as its python dependency management tool. Install UV by:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Initialize virtual env and activate
```bash
uv venv
source .venv/bin/activate
```

Install dependencies with:
```bash
uv sync
```

### Optional: ollama
You have the option to utilize models running on your own machine using Ollama. 

To install [ollama](https://ollama.com/), run:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

You can run models with ollama using the commands:
```bash
ollama run <model name>
```

For example, if you'd like to use Meta's Llama3.2 model, run:
```bash
ollama run llama3.2
```
This command will fetch the model (in the case above, it will fetch Llama3.2 3B model) from Ollama's model hub, then run and serve the model. The default endpoint is `http://localhost:11434`

### Launch the App

Run `streamlit run app.py` to launch the app. 