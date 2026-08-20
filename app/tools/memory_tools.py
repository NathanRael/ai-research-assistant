from langchain_core.tools import BaseTool, tool

from app.services.memory_service import MemoryService


def create_memory_tools(memory_service: MemoryService) -> list[BaseTool]:
    """Build the user-memory tools on top of the memory service."""

    @tool
    def save_memory(content: str, memory_type: str = "fact") -> str:
        """Permanently save a piece of information about the user.

        Args:
            content: The information to remember, written as a clear statement about the user.
            memory_type: Category of the memory, e.g. "preference", "fact", "work".
        """
        memory_id = memory_service.save(content, memory_type=memory_type)
        return f"Memory saved (id={memory_id})."

    @tool
    def retrieve_memory(query: str) -> str:
        """Retrieve previously saved information about the user that matches the query."""
        memories = memory_service.search(query)
        if not memories:
            return "No saved memories match this query."
        lines = [
            f"- [{doc.metadata.get('type', 'fact')}] {doc.page_content}"
            for doc in memories
        ]
        return "\n".join(lines)

    return [save_memory, retrieve_memory]
