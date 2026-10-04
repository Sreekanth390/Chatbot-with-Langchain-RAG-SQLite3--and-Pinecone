import os

from dotenv import load_dotenv


load_dotenv()


# =========================================================
# PINECONE
# =========================================================

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "student-chatbot"
)

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "student-documents"
)


# =========================================================
# OLLAMA
# =========================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_CHAT_MODEL = os.getenv(
    "OLLAMA_CHAT_MODEL",
    "llama3.2"
)

OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text"
)
