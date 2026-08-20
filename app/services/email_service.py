import smtplib
from dataclasses import dataclass
from email.message import EmailMessage


@dataclass
class SmtpConfig:
    """SMTP connection settings. Empty host means email is not configured."""

    host: str = ""
    port: int = 587
    username: str = ""
    password: str = ""
    sender: str = ""


class EmailService:
    """Sends emails over SMTP. Falls back to a dry run when not configured."""

    def __init__(self, config: SmtpConfig | None = None) -> None:
        self.config = config or SmtpConfig()

    @property
    def configured(self) -> bool:
        return bool(self.config.host and self.config.username)

    def send(self, to: str, subject: str, body: str) -> str:
        """Send an email and return a human-readable result message."""
        if not self.configured:
            return (
                f"[dry-run] Email to '{to}' with subject '{subject}' was not sent: "
                "SMTP is not configured (set SMTP_HOST / SMTP_USERNAME)."
            )

        message = EmailMessage()
        message["From"] = self.config.sender or self.config.username
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)

        if self.config.port == 465:
            with smtplib.SMTP_SSL(self.config.host, self.config.port) as server:
                server.login(self.config.username, self.config.password)
                server.send_message(message)
        else:
            with smtplib.SMTP(self.config.host, self.config.port) as server:
                server.starttls()
                server.login(self.config.username, self.config.password)
                server.send_message(message)

        return f"Email sent to {to}."
