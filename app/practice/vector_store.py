from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.practice.embedding import Embeder
from app.user_data import storage_dir


class VectorStore:
    def __init__(self, directory: Optional[str] = None):
        persist_dir = directory or str(storage_dir())
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=Embeder.get_embedding_function(),
        )

    def clear_database(self):
        self.db.reset_collection()

    def add_documents(self, chunks: list[Document]) -> list[str]:
        return self.db.add_documents(documents=chunks)

    def search(self, query: str, top_k: Optional[int] = 5, where: Optional[dict] = None):
        return self.db.similarity_search(query=query, k=top_k, filter=where)

    def get_ids_and_metadata(self) -> dict:
        """Return all chunk ids with their metadata."""
        return self.db.get(include=["metadatas"])

    def delete_ids(self, ids: list[str]) -> None:
        if ids:
            self.db.delete(ids=ids)
