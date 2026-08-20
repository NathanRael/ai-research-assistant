from langchain_core.language_models import BaseChatModel

from app.agents.base_agent import BaseAgent
from app.agents.prompts import PLAIN_TEXT_RULE
from app.services.document_service import DocumentService
from app.services.memory_service import MemoryService
from app.tools.document_tools import create_document_tools
from app.tools.memory_tools import create_memory_tools

PROMPT = f"""You handle everything about the user: their personal information, memories and documents.

Guidelines:
- When the user shares personal information (preferences, work, projects, facts),
  save it with save_memory as a clear statement about the user, e.g. "User works with React".
- When asked anything about the user personally, call retrieve_memory first and answer based on what you find.
- For questions about the user's uploaded documents, use search_documents.
  If you need to know which documents exist, or which one is relevant, call list_documents first.
  If the user asks about a specific document, pass its name to search_documents.
- Answer only from retrieved content and mention which document it comes from.
  If nothing relevant is found, say so honestly instead of inventing information.

{PLAIN_TEXT_RULE}"""


class UserContextAgent(BaseAgent):
    """Stores and recalls personal information, and answers questions about user documents."""

    def __init__(
        self,
        llm: BaseChatModel,
        memory_service: MemoryService,
        document_service: DocumentService,
    ) -> None:
        super().__init__(
            name="user_context",
            description=(
                "Stores and recalls personal information about the user "
                "and answers questions about the user's documents."
            ),
            llm=llm,
            tools=create_memory_tools(memory_service) + create_document_tools(document_service),
            prompt=PROMPT,
        )
