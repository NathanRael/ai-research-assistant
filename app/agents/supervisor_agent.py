from enum import Enum
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.base_agent import BaseAgent
from app.agents.prompts import PLAIN_TEXT_RULE

GENERAL_CHOICE = "general"

SUPERVISOR_PROMPT = """You are the supervisor of a personal AI assistant.
Your only job is to route each user request to the most suitable specialist agent.

Available agents:
{agents}

Rules:
- Pick exactly one agent for each request.
- If the user shares personal information about themselves (facts, preferences, work, projects), route to the agent that handles user context so it can be remembered.
- Use "{general}" only for greetings, small talk, or questions that can be answered directly without any tools.
"""

GENERAL_PROMPT = f"""You are a friendly personal assistant.
Answer the user's message directly and concisely. Do not invent personal facts about the user.

{PLAIN_TEXT_RULE}"""


class SupervisorAgent:
    """Decides which specialist agent should handle the user's request."""

    def __init__(self, llm: BaseChatModel, agents: Sequence[BaseAgent]) -> None:
        self.name = "supervisor"
        self.llm = llm
        self.agents: dict[str, BaseAgent] = {agent.name: agent for agent in agents}
        self._router = self._build_router()

    def route(self, messages: Sequence[BaseMessage]) -> str:
        """Return the name of the agent that should handle the conversation."""
        decision = self._router.invoke(
            [SystemMessage(content=self._system_prompt()), *messages]
        )
        choice = decision.agent.value
        return choice if choice in self.agents else GENERAL_CHOICE

    def answer_general(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Answer directly when no specialist is needed."""
        return self.llm.invoke([SystemMessage(content=GENERAL_PROMPT), *messages])

    def _build_router(self):
        choices = Enum(
            "AgentChoice",
            {name.upper(): name for name in [*self.agents, GENERAL_CHOICE]},
        )

        class RouteDecision(BaseModel):
            """Which specialist should handle the user's request."""

            agent: choices = Field(
                description="Name of the agent that should handle the request"
            )

        return self.llm.with_structured_output(RouteDecision, method="function_calling")

    def _system_prompt(self) -> str:
        agent_lines = "\n".join(
            f"- {name}: {agent.description}" for name, agent in self.agents.items()
        )
        return SUPERVISOR_PROMPT.format(agents=agent_lines, general=GENERAL_CHOICE)
