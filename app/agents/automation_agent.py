from langchain_core.language_models import BaseChatModel

from app.agents.base_agent import BaseAgent
from app.agents.prompts import PLAIN_TEXT_RULE
from app.services.email_service import EmailService
from app.tools.automation_tools import create_automation_tools

PROMPT = f"""You execute external actions on behalf of the user, such as sending emails.

Guidelines:
- Before running a tool, make sure you have all required details; ask the user if something is missing.
- After running a tool, report the outcome clearly, including any dry-run or error messages.

{PLAIN_TEXT_RULE}"""


class AutomationAgent(BaseAgent):
    """Executes external actions such as sending emails or triggering APIs."""

    def __init__(self, llm: BaseChatModel, email_service: EmailService) -> None:
        super().__init__(
            name="automation",
            description="Performs external actions such as sending emails or triggering APIs.",
            capabilities=[
                "Send emails via configured SMTP.",
                "Execute external actions and automations on behalf of the user.",
                "Ask for missing details before acting.",
            ],
            llm=llm,
            tools=create_automation_tools(email_service),
            prompt=PROMPT,
        )
