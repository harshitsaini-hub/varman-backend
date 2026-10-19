import logging
import smtplib
from email.message import EmailMessage

import httpx
from fastapi import HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"


def is_brevo_configured() -> bool:
    """Check if Brevo HTTPS provider is selected and credentials are configured."""
    return (
        settings.email_provider.strip().lower() == "brevo"
        and bool(settings.brevo_api_key.strip())
        and bool(settings.email_from.strip())
    )


def is_smtp_configured() -> bool:
    """Check if SMTP credentials and host are fully configured."""
    return bool(settings.smtp_host and settings.smtp_user and settings.smtp_password)


def _send_brevo(to_email: str, otp: str) -> None:
    """Send OTP password reset email via Brevo transactional email API over HTTPS.

    Never logs the API key, the OTP, or the recipient email.
    """
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "api-key": settings.brevo_api_key.strip(),
    }
    payload = {
        "sender": {
            "name": "Varman",
            "email": settings.email_from.strip(),
        },
        "to": [{"email": to_email}],
        "subject": "Varman Password Reset Code",
        "htmlContent": (
            "<!DOCTYPE html><html><body>"
            f"<p>Your Varman password reset code is <strong>{otp}</strong>. "
            "It expires in 10 minutes. If you did not request it, ignore this email.</p>"
            "</body></html>"
        ),
        "textContent": (
            f"Your Varman password reset code is {otp}. "
            "It expires in 10 minutes. If you did not request it, ignore this email."
        ),
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(BREVO_API_URL, headers=headers, json=payload)
        if resp.is_success:
            logger.info("[Mailer] Password reset email sent successfully via Brevo")
            return
        logger.info("[Mailer] Failed to send password reset email via Brevo: status=%d", resp.status_code)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to send email",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.info("[Mailer] Failed to send password reset email via Brevo: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to send email",
        ) from exc


def _send_smtp(to_email: str, otp: str) -> None:
    """Send OTP password reset email via SMTP with STARTTLS."""
    smtp_host = settings.smtp_host
    smtp_port = settings.smtp_port
    smtp_user = settings.smtp_user
    smtp_password = settings.smtp_password
    smtp_from = settings.smtp_from or settings.email_from or smtp_user or "noreply@varman.ai"

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
        logger.info("[Mailer] Password reset email sent successfully via SMTP")
    except Exception as e:
        logger.info("[Mailer] Failed to send email via SMTP: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to send email",
        ) from e


def send_password_reset_email(to_email: str, otp: str) -> None:
    """Send a 6-digit password reset code.

    Provider choice precedence:
      1. Brevo HTTPS API if EMAIL_PROVIDER=brevo and BREVO_API_KEY and EMAIL_FROM are configured.
      2. SMTP relay if SMTP is configured (host, user, password).
      3. Otherwise:
         - If APP_ENV is production OR DEV_PRINT_OTP is not explicitly True:
           raises HTTP 503 'Email service not configured'. Never prints OTP in production.
         - If APP_ENV is not production AND DEV_PRINT_OTP is explicitly True:
           prints OTP to console for local debugging.
    """
    if is_brevo_configured():
        _send_brevo(to_email, otp)
        return

    if is_smtp_configured():
        _send_smtp(to_email, otp)
        return

    is_prod = settings.app_env.lower() == "production"
    if is_prod or not settings.dev_print_otp:
        logger.warning("[Mailer] No email provider configured. Cannot send password reset email.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email service not configured",
        )

    # Non-production with DEV_PRINT_OTP explicitly True
    logger.info("[DEV ONLY] Password reset OTP for %s: %s", to_email, otp)
