from langchain_core.tools import BaseTool, tool

from app.services.email_service import EmailService


def create_automation_tools(email_service: EmailService) -> list[BaseTool]:
    """Build the automation tools on top of their services."""

    @tool
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email on behalf of the user.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Email body text.
        """
        return email_service.send(to=to, subject=subject, body=body)

    return [send_email]
