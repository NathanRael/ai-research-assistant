from datetime import datetime

from langchain_core.language_models import BaseChatModel

from app.agents.base_agent import BaseAgent
from app.agents.prompts import PLAIN_TEXT_RULE
from app.services.memory_service import MemoryService
from app.services.user_profile_service import UserProfileService
from app.tools.memory_tools import create_memory_tools
from app.tools.search_tools import create_search_web_tool
from app.tools.user_profile_tools import create_user_profile_tools
from app.tools.web_search_client import WebSearchClient

current_date = datetime.now().strftime("%Y-%m-%d")

PROMPT = f"""You are a web search specialist.
Use search_web to find current information, then answer naturally and concisely.

Known user context:
{profile_context}

User Profile vs Memory:
- User Profile (update_user_profile / get_user_profile): the user's general information — portfolio link, social accounts, profession, tech stack, contact details. Updated over time as the user's career and public presence evolve.
- Memory (save_memory / retrieve_memory): learned preferences, habits, interests, and patterns about the user discovered through conversations.

Behavior rules:
- Before searching, call retrieve_memory or get_user_profile to check for relevant user context (tech stack, profession, portfolio, interests). Use this to refine your search queries and tailor results.
- Start with one well-formed query. Run additional searches only if the first results are clearly insufficient.
- Summarize key information naturally. Do not dump raw search results.
- Mention sources only when useful or when the user asks.
- Focus on helping the user understand the topic, not showing the search process.

Current date: {current_date}

{PLAIN_TEXT_RULE}"""


class WebSearchAgent(BaseAgent):
    """Answers questions that require fresh information from the internet."""

    def __init__(
            self,
            llm: BaseChatModel,
            search_client: WebSearchClient,
            memory_service: MemoryService,
            profile_service: UserProfileService,
    ) -> None:
        profile = profile_service.load()
        info = ""
        if profile.name:
            info += f"\nUser's name is {profile.name}.\n"
        if profile.facts:
            info += f"User facts: {profile.facts}\n"

        super().__init__(
            name="web_search",
            description="Answers questions that need up-to-date information from the internet.",
            capabilities=[
                "Search the live web for current events, facts, and recent information.",
                "Summarize search results with sources.",
                "Answer time-sensitive questions.",
            ],
            llm=llm,
            tools=(
                    [create_search_web_tool(search_client)]
                    + create_memory_tools(memory_service)
                    + create_user_profile_tools(profile_service)
            ),
            prompt=PROMPT.format(
                profile_context=(info.strip() if info else "No user profile information available yet."),
                PLAIN_TEXT_RULE=PLAIN_TEXT_RULE,
            ),
        )
