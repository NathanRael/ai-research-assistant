from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.user_data import storage_dir


class MemoryService:
    """Persistent user memory backed by a dedicated Chroma collection."""

    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "user_memory",
    ) -> None:
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=str(storage_dir()),
        )

    def save(self, content: str, memory_type: str = "fact") -> str:
        """Store a single memory and return its id."""
        document = Document(page_content=content, metadata={"type": memory_type})
        ids = self.db.add_documents([document])
        return ids[0]

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Return the stored memories most relevant to the query."""
        return self.db.similarity_search(query, k=k)
