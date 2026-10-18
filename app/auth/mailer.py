import logging
import smtplib
from email.message import EmailMessage

from fastapi import HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)


def send_password_reset_email(to_email: str, otp: str) -> None:
    """Send a 6-digit password reset code via SMTP with STARTTLS.

    If SMTP is unconfigured:
      - In development with DEV_PRINT_OTP=true, prints OTP to console for local debugging.
      - Otherwise, raises HTTP 503 'Email service not configured'.
    """
    smtp_host = settings.smtp_host
    smtp_port = settings.smtp_port
    smtp_user = settings.smtp_user
    smtp_password = settings.smtp_password
    smtp_from = settings.smtp_from or smtp_user or "noreply@varman.ai"

    # Check if SMTP credentials are provided
    if not smtp_user or not smtp_password:
        if settings.app_env.lower() == "development" and settings.dev_print_otp:
            logger.info("[DEV ONLY] Password reset OTP for %s: %s", to_email, otp)
            return
        logger.warning("[Mailer] SMTP credentials not configured. Cannot send password reset email.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email service not configured",
        )

    msg = EmailMessage()
    msg["Subject"] = "Varman Password Reset Code"
    msg["From"] = smtp_from
    msg["To"] = to_email

    plain_text = (
        f"Your Varman password reset code is {otp}. "
        "It expires in 10 minutes. If you did not request it, ignore this email."
    )
    html_content = (
        f"<!DOCTYPE html><html><body>"
        f"<p>Your Varman password reset code is <strong>{otp}</strong>. "
        "It expires in 10 minutes. If you did not request it, ignore this email.</p>"
        f"</body></html>"
    )

    msg.set_content(plain_text)
    msg.add_alternative(html_content, subtype="html")

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        logger.info("[Mailer] Password reset email sent successfully to %s", to_email)
    except Exception as e:
        logger.error("[Mailer] Failed to send email to %s: %s", to_email, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to send email. Please try again later.",
        ) from e
