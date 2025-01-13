import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag import init_chroma_vector_store


def main():
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

    # Initialize the vector store
    vector_store = init_chroma_vector_store()
    vector_store.add_documents(docs_processed)


if __name__ == "__main__":
    main()