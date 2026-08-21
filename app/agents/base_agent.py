from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel


class BaseAgent:
    """Base for all agents: an LLM equipped with tools and a system prompt."""

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: list[str],
        llm: BaseChatModel,
        tools: list[Any],
        prompt: str,
    ) -> None:
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.llm = llm
        self.tools = tools
        self.prompt = prompt
        self._agent = None

    def build(self):
        """Compile the underlying LangChain agent (cached)."""
        if self._agent is None:
            self._agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=self.prompt,
                name=self.name,
            )
        return self._agent
