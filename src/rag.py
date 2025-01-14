import os
from smolagents import CodeAgent, Model

from smolagents import Tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.common import load_model


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

    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return vector_store, text_splitter


class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of the documents stored in a vector database that could be most relevant to answer your query."
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
            search_type="similarity",
            search_kwargs={"k": 10},
        )

    def update_description(self, description: str):
        self.description = self.description.replace("{documents}", description)

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


def init_rag_agent(model: Model):
    # Initialize the vector store
    vector_store, _ = init_chroma_vector_store()

    # Initialize the retriever tool and Agent
    retriever_tool = RetrieverTool(vector_store)
    retriever_agent = CodeAgent(
        tools=[retriever_tool],
        model=model,
        max_steps=4,
        # verbose=False,
    )

    return retriever_agent


def load_pdf(files, text_splitter):
    pages = []
    for file in files:
        # Load PDF as a document
        loader = PyPDFLoader(file)
        for page in loader.lazy_load():
            pages.append(page)

    # load into Documents
    docs_processed = text_splitter.split_documents(pages)

    return docs_processed


def load_dataset(dataset, text_splitter, content_field="text"):
    # Load the dataset
    knowledge_base = datasets.load_dataset(dataset, split="train")

    # load into Documents
    source_docs = [Document(page_content=doc[content_field]) for doc in knowledge_base]
    docs_processed = text_splitter.split_documents(source_docs)

    return docs_processed


def main():
    print("RAG Agent")

    # Initialize model
    model = load_model(
        provider="ollama",
        model_id="qwen2.5-coder:7b",
        api_key=None,
        api_base="http://localhost:11434",
    )

    # Initialize the RAG Agent
    print("Initializing the RAG Agent...")
    retriever_agent = init_rag_agent(model)

    query = input("Enter your query: ")
    response = retriever_agent.run(query)
    print(response)


if __name__ == "__main__":
    main()
