from langchain_core.language_models import BaseChatModel

from app.agents.base_agent import BaseAgent
from app.agents.prompts import PLAIN_TEXT_RULE
from app.services.email_service import EmailService
from app.tools.automation_tools import create_automation_tools

PROMPT = f"""You execute external actions on behalf of the user, such as sending and reading emails.

Authorization and review rules:
- Before sending any email, use preview_email to show the draft and ask the user to confirm.
- Only call send_email with confirmed=true after the user has explicitly approved the draft.
- Before reading the user's inbox, use preview_read_emails to explain what will be accessed and ask for authorization.
- Only call read_emails with authorized=true after the user has explicitly authorized it.
- If details are missing (recipient, subject, etc.), ask the user before previewing.
- After running a tool, report the outcome clearly, including any dry-run or error messages.

{PLAIN_TEXT_RULE}"""


class AutomationAgent(BaseAgent):
    """Executes external actions such as sending emails or triggering APIs."""

    def __init__(self, llm: BaseChatModel, email_service: EmailService) -> None:
        super().__init__(
            name="automation",
            description="Performs external actions such as sending emails or triggering APIs.",
            capabilities=[
                "Preview and send emails with user confirmation.",
                "Preview and read emails with user authorization.",
                "Execute external actions and automations on behalf of the user.",
                "Ask for missing details before acting.",
            ],
            llm=llm,
            tools=create_automation_tools(email_service),
            prompt=PROMPT,
        )
