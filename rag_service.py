from pathlib import Path

from pinecone import Pinecone, ServerlessSpec

from langchain_ollama import OllamaEmbeddings

from langchain_pinecone import PineconeVectorStore

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.documents import Document

from config import (
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE
)


# =========================================================
# CREATE PINECONE INDEX
# =========================================================

def create_index():

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    indexes = [
        index["name"]
        for index in pc.list_indexes()
    ]

    if PINECONE_INDEX_NAME not in indexes:

        pc.create_index(

            name=PINECONE_INDEX_NAME,

            # nomic-embed-text = 768 dimensions
            dimension=768,

            metric="cosine",

            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print(
            "Pinecone index created."
        )

    else:

        print(
            "Pinecone index already exists."
        )


# =========================================================
# LOAD DOCUMENTS
# =========================================================

def load_documents():

    documents = []

    folder = Path("documents")

    for file in folder.glob("*.txt"):

        text = file.read_text(
            encoding="utf-8"
        )

        documents.append(

            Document(

                page_content=text,

                metadata={
                    "source": file.name
                }
            )
        )

    return documents


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

def get_embeddings():

    embeddings = OllamaEmbeddings(

        model=OLLAMA_EMBEDDING_MODEL,

        base_url=OLLAMA_BASE_URL
    )

    return embeddings


# =========================================================
# CREATE VECTOR STORE
# =========================================================

def get_vector_store():

    embeddings = get_embeddings()

    vector_store = PineconeVectorStore(

        index_name=PINECONE_INDEX_NAME,

        embedding=embeddings,

        namespace=PINECONE_NAMESPACE
    )

    return vector_store


# =========================================================
# INGEST DOCUMENTS
# =========================================================

def ingest_documents():

    create_index()

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    vector_store = get_vector_store()

    vector_store.add_documents(
        chunks
    )

    print(
        "Documents added to Pinecone."
    )


# =========================================================
# RETRIEVE DOCUMENTS
# =========================================================

def retrieve_documents(question):

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(

        search_kwargs={
            "k": 4
        }
    )

    documents = retriever.invoke(
        question
    )

    return documents

