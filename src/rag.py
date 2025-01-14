import os

from smolagents import CodeAgent, Model

from smolagents import Tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def init_chroma_vector_store():
    # Create persist directory at project root
    dir = ".db"
    os.makedirs(dir, exist_ok=True)

    # Initialize embedding model
    emb_model = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda", "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceEmbeddings(
        model_name=emb_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    # Initialize the vector store
    vector_store = Chroma(
        collection_name="rag_vector_store",
        embedding_function=embedding_model,
        persist_directory=".db",
    )

    return vector_store


class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vector_store: Chroma, **kwargs):
        super().__init__(**kwargs)
        self.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 2, "lambda_mult": 0.5},
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )


def init_rag_agent(
    model: Model
):
    # Initialize the vector store
    vector_store = init_chroma_vector_store()

    # Initialize the retriever tool and Agent
    retriever_tool = RetrieverTool(vector_store)
    retriever_agent = CodeAgent(
        tools=[retriever_tool],
        model=model,
        max_steps=4,
        # verbose=False,
    )

    return retriever_agent


def main():
    print("RAG Agent")

    # Initialize the RAG Agent
    print("Initializing the RAG Agent...")
    retriever_agent = init_rag_agent()

    query = input("Enter your query: ")
    response = retriever_agent.run(query)
    print(response)


if __name__ == "__main__":
    main()
