from langchain_core.language_models import BaseChatModel

from app.agents.base_agent import BaseAgent
from app.agents.prompts import PLAIN_TEXT_RULE
from app.services.email_service import EmailService
from app.services.memory_service import MemoryService
from app.services.user_profile_service import UserProfileService
from app.tools.automation_tools import create_automation_tools
from app.tools.memory_tools import create_memory_tools
from app.tools.user_profile_tools import create_user_profile_tools

PROMPT = """You execute external actions on behalf of the user, such as sending and reading emails.

Known user context:
{profile_context}

User Profile vs Memory:
- User Profile (update_user_profile / get_user_profile): the user's general information — portfolio link, social accounts, profession, tech stack, contact details. Updated over time as the user's career and public presence evolve.
- Memory (save_memory / retrieve_memory): learned preferences, habits, interests, and patterns about the user discovered through conversations.

Authorization rules:
- Before sending any email, use preview_email to show the draft and ask the user to confirm.
- Only call send_email with confirmed=true after explicit user approval.
- Before reading the inbox, use preview_read_emails to explain what will be accessed and ask for authorization.
- Only call read_emails with authorized=true after explicit user authorization.
- If details are missing (recipient, subject, etc.), ask the user before previewing.
- After running a tool, report the outcome clearly, including any dry-run or error messages.

Context rules:
- Before drafting emails, call retrieve_memory to check for stored preferences (email format, tone, sign-off style, portfolio links to include, etc.).
- Call get_user_profile to get the user's name, profession, contact details, and links to use in signatures or introductions.
- Apply stored email format preferences automatically when composing drafts.
- When the user shares a new email preference, call save_memory to store it for future use.

{PLAIN_TEXT_RULE}"""


class AutomationAgent(BaseAgent):
    """Executes external actions such as sending emails or triggering APIs."""

    def __init__(
            self,
            llm: BaseChatModel,
            email_service: EmailService,
            memory_service: MemoryService,
            profile_service: UserProfileService,
    ) -> None:
        profile = profile_service.load()
        info = ""
        if profile.name:
            info += f"\nUser's name is {profile.name}.\n"
        if profile.facts:
            info += f"User facts: {profile.facts}\n"
        if profile.email:
            info += f"User email: {profile.email}\n"

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
            tools=(
                    create_automation_tools(email_service)
                    + create_memory_tools(memory_service)
                    + create_user_profile_tools(profile_service)
            ),
            prompt=PROMPT.format(
                profile_context=(info.strip() if info else "No user profile information available yet."),
                PLAIN_TEXT_RULE=PLAIN_TEXT_RULE,
            ),
        )
