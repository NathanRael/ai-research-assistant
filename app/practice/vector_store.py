from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.practice.embedding import Embeder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "storage"

class VectorStore:
    def __init__(self, directory: Optional[str] = DEFAULT_DIR):
        self.db = Chroma(persist_directory=directory, embedding_function=Embeder.get_embedding_function())

    def clear_database(self):
        self.db.reset_collection()

    def add_documents(self, chunks: list[Document]):
        db = self.db
        db.add_documents(
            documents=chunks
        )

    def search(self, query: str, top_k: Optional[int] = 5):
        db = self.db
        return db.similarity_search(query=query, k=top_k)
