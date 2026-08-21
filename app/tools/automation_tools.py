from langchain_core.tools import BaseTool, tool

from app.services.email_service import EmailService


def _format_email(email: dict[str, str]) -> str:
    if "error" in email:
        return email["error"]
    return (
        f"From: {email.get('from', 'Unknown')}\n"
        f"Subject: {email.get('subject', 'No subject')}\n"
        f"Date: {email.get('date', 'Unknown')}\n"
        f"{email.get('body', '')}"
    )


def create_automation_tools(email_service: EmailService) -> list[BaseTool]:
    """Build the automation tools on top of their services."""

    @tool
    def preview_email(to: str, subject: str, body: str) -> str:
        """Show the user a preview of the email before sending."""
        return email_service.preview_send(to=to, subject=subject, body=body)

    @tool
    def send_email(to: str, subject: str, body: str, confirmed: bool = False) -> str:
        """Send an email on behalf of the user.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Email body text.
            confirmed: Set to True only after the user has reviewed and approved the email.
        """
        if not confirmed:
            return (
                email_service.preview_send(to=to, subject=subject, body=body)
                + "\n\nSet confirmed=true to send after the user approves."
            )
        return email_service.send(to=to, subject=subject, body=body)

    @tool
    def preview_read_emails(limit: int = 5) -> str:
        """Show what emails will be read before accessing the inbox."""
        return email_service.preview_read(limit=limit)

    @tool
    def read_emails(limit: int = 5, authorized: bool = False) -> str:
        """Read recent emails from the user's inbox.

        Args:
            limit: Maximum number of recent emails to read.
            authorized: Set to True only after the user has authorized reading their emails.
        """
        if not authorized:
            return (
                email_service.preview_read(limit=limit)
                + "\n\nSet authorized=true to proceed after the user approves."
            )
        emails = email_service.read_emails(limit=limit)
        if not emails:
            return "No emails found or IMAP is not configured."
        return "\n\n---\n\n".join(_format_email(e) for e in emails)

    return [preview_email, send_email, preview_read_emails, read_emails]
