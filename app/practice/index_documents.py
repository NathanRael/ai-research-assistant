import requests

from app.practice.doc_manager import DocManager
from app.practice.vector_store import VectorStore

OLLAMA_BASE_URL = "http://localhost:11434"


def check_ollama() -> None:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.ConnectionError:
        raise SystemExit(
            "Cannot connect to Ollama at http://localhost:11434. "
            "Make sure Ollama is installed and running (`ollama serve`)."
        )
    except requests.Timeout:
        raise SystemExit(
            "Ollama did not respond in time. "
            "Check that Ollama is running and not overloaded."
        )


def index_documents():
    check_ollama()
    print("Loading documents...")
    docs = DocManager.load_documents()
    print(f"Loaded {len(docs)} documents")
    if not docs:
        raise SystemExit("No documents found. Add PDF/TXT files to the data directory.")
    print("Indexing documents...")
    vs = VectorStore()
    vs.clear_database()
    chunks = DocManager().split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    if chunks:
        vs.add_documents(chunks)
        print(f"{len(chunks)} chunks saved...")
    else:
        print("No chunks to save.")


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    print("Starting indexing...")

    index_documents()

    print("Indexing completed!")
