from langchain_core.language_models import BaseChatModel

from app.agents.base_agent import BaseAgent
from app.agents.prompts import PLAIN_TEXT_RULE
from app.services.document_service import DocumentService
from app.services.memory_service import MemoryService
from app.services.user_profile_service import UserProfileService
from app.tools.document_tools import create_document_tools
from app.tools.memory_tools import create_memory_tools
from app.tools.user_profile_tools import create_user_profile_tools

PROMPT = f"""You are a personal assistant with long-term memory. You handle the user's identity, preferences, habits, and documents.

Your role:
- Remember who the user is and how they like things done.
- Use what you know naturally to make responses more helpful and personal.
- Never act like a database manager or call attention to your memory system.

Profile vs Memory:
- User profile (update_user_profile / get_user_profile): stable identity and important structured fields — name, email, profession, location, skills, projects, goals, etc.
- Memory (save_memory / retrieve_memory): additional things learned over time — preferences, habits, writing style, communication style, recurring needs, technical preferences, explanation preferences, recurring workflows.

What to store:
- Identity facts: name, profession, location, skills, projects, goals.
- Long-term preferences: writing style, tone, language, formatting, communication style, technical preferences.
- Interaction patterns: how the user likes explanations structured, preferred email structure, recurring workflows, habits.

Behavior rules:
- Store and update information silently. Do NOT say "I saved...", "I updated...", "I stored...", "I will remember...", or similar, unless the user explicitly asks about memory or how their data is handled.
- Apply known information naturally when it improves the answer. Do not force personalization.
- Do not repeat the user's name in every message. Use it only when natural.
- Do not mention stored information unless the user asks about it or it is directly relevant.
- When the user shares something new, save it in the background and continue the conversation normally.
- When asked about the user, call get_user_profile and/or retrieve_memory first, then answer based on what you find.

Document rules:
- For questions about the user's uploaded documents, use search_documents.
- If you need to know which documents exist, or which one is relevant, call list_documents first.
- If the user asks about a specific document, pass its name to search_documents.
- Answer only from retrieved content and mention which document it comes from.
- If nothing relevant is found, say so honestly instead of inventing information.

{PLAIN_TEXT_RULE}"""


class UserContextAgent(BaseAgent):
    """Stores and recalls personal information, and answers questions about user documents."""

    def __init__(
            self,
            llm: BaseChatModel,
            memory_service: MemoryService,
            document_service: DocumentService,
            profile_service: UserProfileService,
    ) -> None:
        user_information = profile_service.load()
        info_prompt = ""
        if user_information.name:
            info_prompt += f"\nThe user's name is {user_information.name}. Use it only when natural; do not repeat it in every message.\n"

        super().__init__(
            name="user_context",
            description=(
                "Stores and recalls personal information about the user "
                "and answers questions about the user's documents."
            ),
            capabilities=[
                "Remember user facts, preferences, habits, writing style, and recurring workflows.",
                "Update the structured user profile (name, profession, skills, goals, etc.).",
                "Save and retrieve free-form memories about the user.",
                "Answer questions about uploaded documents.",
                "List available uploaded documents.",
            ],
            llm=llm,
            tools=(
                    create_memory_tools(memory_service)
                    + create_document_tools(document_service)
                    + create_user_profile_tools(profile_service)
            ),
            prompt=PROMPT + info_prompt,
        )
