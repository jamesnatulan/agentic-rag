import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from smolagents import HfApiModel, CodeAgent

from retriever import RetrieverTool

def main():
    print("Hello from agentic-rag!")

    # Load the knowledge-base
    # TODO: Uses transformers library of markdown files, can be changed with other scraped docs
    knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
    knowledge_base = knowledge_base.filter(
        lambda row: row["source"].startswith("huggingface/transformers")
    )

    # load into Documents
    source_docs = [
        Document(
            page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}
        )
        for doc in knowledge_base
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    docs_processed = text_splitter.split_documents(source_docs)
    retriever_tool = RetrieverTool(docs_processed)

    # Initialize the Agent
    agent = CodeAgent(
        tools=[retriever_tool],
        model=HfApiModel(),
        max_steps=4,
        verbose=True,
    )

    # Run sample
    agent_output = agent.run("For a transformers model training, which is slower, the forward or the backward pass?")

    print("Final output:")
    print(agent_output)


if __name__ == "__main__":
    main()
