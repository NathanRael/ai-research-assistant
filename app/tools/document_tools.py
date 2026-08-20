from langchain_core.tools import BaseTool, tool

from app.services.document_service import DocumentService


def create_document_tools(document_service: DocumentService) -> list[BaseTool]:
    """Build the document tools on top of the document service."""

    @tool
    def search_documents(query: str, document_name: str = "") -> str:
        """Search the user's uploaded documents for content relevant to the query.

        Args:
            query: What to look for in the documents.
            document_name: Optional document name to restrict the search to a single document.
        """
        documents = document_service.search(query, document_name=document_name)
        if not documents:
            if document_name:
                return f"No matching content found in document '{document_name}'."
            return "No matching content found in the user's documents."
        blocks = []
        for idx, doc in enumerate(documents, start=1):
            name = doc.metadata.get("document_name", "unknown")
            page = doc.metadata.get("page")
            location = f"{name} (page {page})" if page else name
            blocks.append(f"{idx}. {doc.page_content}\nSource: {location}")
        return "\n\n".join(blocks)

    @tool
    def list_documents() -> str:
        """List the documents the user has uploaded, with their names and chunk counts."""
        infos = document_service.list_documents()
        if not infos:
            return "No documents have been uploaded yet."
        return "\n".join(f"- {info.name} ({info.chunks} chunks)" for info in infos)

    return [search_documents, list_documents]
