import email
import imaplib
import smtplib
from dataclasses import dataclass
from email.header import decode_header
from email.message import EmailMessage
from typing import Optional


@dataclass
class SmtpConfig:
    """SMTP/IMAP connection settings. Empty host means email is not configured."""

    host: str = ""
    port: int = 587
    username: str = ""
    password: str = ""
    sender: str = ""
    imap_host: str = ""
    imap_port: int = 993
    imap_username: str = ""
    imap_password: str = ""
    imap_folder: str = "INBOX"


class EmailService:
    """Sends and reads emails. Falls back to a dry run when not configured."""

    def __init__(self, config: SmtpConfig | None = None) -> None:
        self.config = config or SmtpConfig()

    @property
    def configured(self) -> bool:
        return bool(self.config.host and self.config.username)

    @property
    def imap_configured(self) -> bool:
        return bool(
            (self.config.imap_host and self.config.imap_username)
            or (self.config.host and self.config.username)
        )

    def preview_send(self, to: str, subject: str, body: str) -> str:
        """Return a human-readable preview of the email that would be sent."""
        sender = self.config.sender or self.config.username or "<not configured>"
        return (
            f"Ready to send email:\n"
            f"  From: {sender}\n"
            f"  To: {to}\n"
            f"  Subject: {subject}\n"
            f"  Body:\n{body}\n\n"
            f"Confirm to proceed with sending."
        )

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

    def preview_read(self, limit: int = 5) -> str:
        """Return a preview of what will be read from the inbox."""
        if not self.imap_configured:
            return "IMAP is not configured (set IMAP_HOST / IMAP_USERNAME)."
        return (
            f"Ready to read the latest {limit} emails from "
            f"'{self.config.imap_folder or 'INBOX'}'. Confirm to proceed."
        )

    def read_emails(
        self, limit: int = 5, folder: Optional[str] = None
    ) -> list[dict[str, str]]:
        """Read recent emails from the configured IMAP inbox."""
        if not self.imap_configured:
            return []

        host = self.config.imap_host or self.config.host
        username = self.config.imap_username or self.config.username
        password = self.config.imap_password or self.config.password
        mailbox = folder or self.config.imap_folder or "INBOX"

        try:
            with imaplib.IMAP4_SSL(host, self.config.imap_port) as mail:
                mail.login(username, password)
                mail.select(mailbox)
                _, data = mail.search(None, "ALL")
                ids = data[0].split()
                recent_ids = ids[-limit:] if len(ids) > limit else ids
                emails: list[dict[str, str]] = []
                for email_id in reversed(recent_ids):
                    _, msg_data = mail.fetch(email_id, "(RFC822)")
                    raw_email = msg_data[0][1]
                    parsed = email.message_from_bytes(raw_email)
                    subject = self._decode_header(parsed.get("Subject", "No subject"))
                    from_ = self._decode_header(parsed.get("From", "Unknown sender"))
                    date = parsed.get("Date", "Unknown date")
                    body = ""
                    if parsed.is_multipart():
                        for part in parsed.walk():
                            content_type = part.get_content_type()
                            if content_type == "text/plain":
                                payload = part.get_payload(decode=True)
                                if payload:
                                    body = payload.decode("utf-8", errors="replace")
                                    break
                    else:
                        payload = parsed.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="replace")
                    emails.append(
                        {
                            "subject": subject,
                            "from": from_,
                            "date": date,
                            "body": body.strip(),
                        }
                    )
                return emails
        except Exception as exc:
            return [{"error": f"Failed to read emails: {exc}"}]

    @staticmethod
    def _decode_header(value: Optional[str]) -> str:
        if not value:
            return ""
        parts = decode_header(value)
        decoded = []
        for part, charset in parts:
            if isinstance(part, bytes):
                decoded.append(part.decode(charset or "utf-8", errors="replace"))
            else:
                decoded.append(part)
        return "".join(decoded)
