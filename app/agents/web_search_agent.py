from datetime import datetime

from langchain_core.language_models import BaseChatModel

from app.agents.base_agent import BaseAgent
from app.agents.prompts import PLAIN_TEXT_RULE
from app.tools.search_tools import create_search_web_tool
from app.tools.web_search_client import WebSearchClient

current_date = datetime.now().strftime("%Y-%m-%d")

PROMPT = f"""You are a web search specialist.
Use the search_web tool to find up-to-date information, then answer the user's
question with a concise summary of the most relevant results.

Search strategy:
- Start with one well-formed, optimized query.
- Run additional searches only if the first results are clearly insufficient.
- Avoid parallel or redundant searches.

Always mention the sources you used.

Current date : {current_date}

{PLAIN_TEXT_RULE}"""


class WebSearchAgent(BaseAgent):
    """Answers questions that require fresh information from the internet."""

    def __init__(self, llm: BaseChatModel, search_client: WebSearchClient) -> None:
        super().__init__(
            name="web_search",
            description="Answers questions that need up-to-date information from the internet.",
            capabilities=[
                "Search the live web for current events, facts, and recent information.",
                "Summarize search results with sources.",
                "Answer time-sensitive questions.",
            ],
            llm=llm,
            tools=[create_search_web_tool(search_client)],
            prompt=PROMPT,
        )
