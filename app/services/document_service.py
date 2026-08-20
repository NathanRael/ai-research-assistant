from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

from app.practice.doc_manager import DocManager
from app.practice.vector_store import VectorStore


@dataclass
class DocumentInfo:
    """Summary of an indexed document."""

    name: str
    source: str
    chunks: int
    added_at: str


class DocumentService:
    """Index and search the user's uploaded documents.

    Documents live in the vector store collection, which is separate from the
    user-memory collection managed by MemoryService.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def add_document(self, path: str | Path) -> DocumentInfo:
        """Validate, chunk and index a document file.

        Re-adding an existing file replaces its previous chunks.
        """
        file = Path(path).expanduser().resolve()
        if not file.is_file():
            raise FileNotFoundError(f"File not found: {file}")

        documents = DocManager.load_file(file)
        chunks = DocManager.split_documents(documents)
        if not chunks:
            raise ValueError(f"No extractable text found in '{file.name}'.")

        self.remove_document(file.name, source=str(file))

        added_at = datetime.now().isoformat(timespec="seconds")
        for chunk in chunks:
            chunk.metadata.update(
                {"document_name": file.name, "added_at": added_at}
            )

        self.vector_store.add_documents(chunks)
        return DocumentInfo(
            name=file.name,
            source=str(file),
            chunks=len(chunks),
            added_at=added_at,
        )

    def remove_document(self, name: str, source: str = "") -> int:
        """Remove all chunks belonging to a document. Returns chunks removed."""
        ids = self._chunk_ids_for(name, source)
        self.vector_store.delete_ids(ids)
        return len(ids)

    def list_documents(self) -> list[DocumentInfo]:
        """Return one entry per indexed document, sorted by name."""
        records = self.vector_store.get_ids_and_metadata()
        grouped: dict[str, DocumentInfo] = {}
        for metadata in records["metadatas"]:
            metadata = metadata or {}
            source = metadata.get("source", "")
            name = metadata.get("document_name") or Path(source).name or "unknown"
            info = grouped.get(name)
            if info is None:
                grouped[name] = DocumentInfo(
                    name=name,
                    source=source,
                    chunks=1,
                    added_at=metadata.get("added_at", ""),
                )
            else:
                info.chunks += 1
                if not info.source and source:
                    info.source = source
        return sorted(grouped.values(), key=lambda info: info.name.lower())

    def search(
        self, query: str, k: int = 5, document_name: str = ""
    ) -> list[Document]:
        """Return the document chunks most relevant to the query.

        When document_name is given, the search is restricted to that document.
        """
        where = None
        if document_name:
            info = next(
                (
                    doc
                    for doc in self.list_documents()
                    if doc.name.lower() == document_name.lower()
                ),
                None,
            )
            if info is None:
                return []
            where = (
                {"source": info.source}
                if info.source
                else {"document_name": info.name}
            )
        return self.vector_store.search(query, top_k=k, where=where)

    def _chunk_ids_for(self, name: str, source: str = "") -> list[str]:
        records = self.vector_store.get_ids_and_metadata()
        ids = []
        for chunk_id, metadata in zip(records["ids"], records["metadatas"]):
            metadata = metadata or {}
            matches_name = metadata.get("document_name") == name or (
                Path(metadata.get("source", "")).name == name
            )
            matches_source = bool(source) and metadata.get("source") == source
            if matches_name or matches_source:
                ids.append(chunk_id)
        return ids
