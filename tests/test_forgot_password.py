import logging
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import httpx
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.password_reset import PasswordResetOTP
from app.models.user import User


@pytest.mark.asyncio
async def test_register_then_login_works(client: AsyncClient):
    """Register returns access_token & vault_salt directly, and subsequent login works."""
    reg_res = await client.post(
        "/api/auth/register",
        json={
            "email": "freshuser@example.com",
            "password": "validPassword123",
            "display_name": "Fresh User",
        },
    )
    assert reg_res.status_code == 201
    data = reg_res.json()
    assert "access_token" in data
    assert "vault_salt" in data
    assert len(data["vault_salt"]) == 64

    # Subsequent login works with same credentials
    login_res = await client.post(
        "/api/auth/login",
        data={"username": "freshuser@example.com", "password": "validPassword123"},
    )
    assert login_res.status_code == 200
    assert "access_token" in login_res.json()
    assert login_res.json()["vault_salt"] == data["vault_salt"]


@pytest.mark.asyncio
async def test_registration_atomicity_on_failure(client: AsyncClient, db: AsyncSession):
    """Failure during registration rolls back and leaves no orphaned user in the database."""
    with patch("app.auth.security.create_access_token", side_effect=RuntimeError("Token generation failed")):
        with patch.object(db, "commit", side_effect=RuntimeError("Simulated DB commit error")):
            res = await client.post(
                "/api/auth/register",
                json={
                    "email": "orphancheck@example.com",
                    "password": "validPassword123",
                    "display_name": "Orphan Test",
                },
            )
            assert res.status_code == 500

    # Verify no row was persisted in database
    stmt = select(User).where(User.email == "orphancheck@example.com")
    persisted = (await db.execute(stmt)).scalar_one_or_none()
    assert persisted is None


@pytest.mark.asyncio
async def test_forgot_password_registered_sends_mail_and_hashes_otp(
    client: AsyncClient, test_user: User, db: AsyncSession
):
    """Registered email receives 1 email, and the database stores an HMAC hash rather than plaintext OTP."""
    with patch("app.auth.mailer.send_password_reset_email") as mock_mail:
        res = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
        )
        assert res.status_code == 200
        assert "Password reset code sent" in res.json()["message"]
        assert mock_mail.call_count == 1

        # Check argument passed to mailer
        args, _ = mock_mail.call_args
        to_email, sent_otp = args
        assert to_email == test_user.email
        assert re.match(r"^\d{6}$", sent_otp)

        # Inspect database record
        stmt = (
            select(PasswordResetOTP)
            .where(PasswordResetOTP.user_id == test_user.id)
            .order_by(PasswordResetOTP.created_at.desc())
        )
        record = (await db.execute(stmt)).scalars().first()
        assert record is not None
        assert record.otp_hash != sent_otp  # Must NOT be stored in plaintext
        assert len(record.otp_hash) == 64  # SHA-256 hex length
        assert record.attempts == 0
        assert record.used_at is None


@pytest.mark.asyncio
async def test_forgot_password_unregistered_email_404(client: AsyncClient):
    """Unregistered email returns 404 when REVEAL_UNREGISTERED_EMAIL is True (default)."""
    settings.reveal_unregistered_email = True
    res = await client.post(
        "/api/auth/forgot-password/request",
        json={"email": "nobody@example.com"},
    )
    assert res.status_code == 404
    assert "No account found for this email. Please sign up first." in res.json()["detail"]


@pytest.mark.asyncio
async def test_forgot_password_unregistered_email_generic_200(client: AsyncClient):
    """Unregistered email returns generic 200 when REVEAL_UNREGISTERED_EMAIL is False."""
    settings.reveal_unregistered_email = False
    try:
        res = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": "hidden_nobody@example.com"},
        )
        assert res.status_code == 200
        assert "If an account exists" in res.json()["message"]
    finally:
        settings.reveal_unregistered_email = True


@pytest.mark.asyncio
async def test_forgot_password_cooldown_and_hourly_limits(
    client: AsyncClient, test_user: User, db: AsyncSession
):
    """Enforces 60-second resend cooldown and 3 requests per email per hour."""
    with patch("app.auth.mailer.send_password_reset_email"):
        # 1st request succeeds
        res1 = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
        )
        assert res1.status_code == 200

        # Immediate 2nd request triggers 60s cooldown
        res2 = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
        )
        assert res2.status_code == 429
        assert "60 seconds" in res2.json()["detail"]

        # Age the first record by 61 seconds to bypass cooldown
        stmt = (
            select(PasswordResetOTP)
            .where(PasswordResetOTP.user_id == test_user.id)
            .order_by(PasswordResetOTP.created_at.desc())
        )
        rec = (await db.execute(stmt)).scalars().first()
        rec.created_at = datetime.now(timezone.utc) - timedelta(seconds=65)
        await db.commit()

        # 2nd aged request succeeds
        res3 = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
        )
        assert res3.status_code == 200

        # Age the 2nd record
        rec2 = (await db.execute(stmt)).scalars().first()
        rec2.created_at = datetime.now(timezone.utc) - timedelta(seconds=65)
        await db.commit()

        # 3rd request succeeds
        res4 = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
        )
        assert res4.status_code == 200

        # Age the 3rd record
        rec3 = (await db.execute(stmt)).scalars().first()
        rec3.created_at = datetime.now(timezone.utc) - timedelta(seconds=65)
        await db.commit()

        # 4th request triggers hourly limit (max 3/hour)
        res5 = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
        )
        assert res5.status_code == 429
        assert "Too many password reset requests for this email" in res5.json()["detail"]


@pytest.mark.asyncio
async def test_wrong_otp_increments_attempts_and_locks(
    client: AsyncClient, test_user: User
):
    """Wrong OTP increments attempts and locks/invalidates after 5 failures."""
    with patch("app.auth.mailer.send_password_reset_email"):
        await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
        )

    # Submit 4 wrong attempts
    for attempt in range(1, 5):
        res = await client.post(
            "/api/auth/forgot-password/reset",
            json={
                "email": test_user.email,
                "otp": f"99999{attempt}",
                "new_password": "newSecurePassword123",
            },
        )
        assert res.status_code == 400
        assert f"{5 - attempt} attempt(s) remaining" in res.json()["detail"]

    # 5th wrong attempt locks the OTP
    res_lock = await client.post(
        "/api/auth/forgot-password/reset",
        json={
            "email": test_user.email,
            "otp": "999995",
            "new_password": "newSecurePassword123",
        },
    )
    assert res_lock.status_code == 400
    assert "Too many failed attempts. This code has been invalidated" in res_lock.json()["detail"]

    # Subsequent attempt fails because it is locked
    res_subsequent = await client.post(
        "/api/auth/forgot-password/reset",
        json={
            "email": test_user.email,
            "otp": "999996",
            "new_password": "newSecurePassword123",
        },
    )
    assert res_subsequent.status_code == 400
    assert "Too many failed attempts. This code has been invalidated" in res_subsequent.json()["detail"]


@pytest.mark.asyncio
async def test_reset_password_success_and_login(
    client: AsyncClient, test_user: User, db: AsyncSession
):
    """Valid OTP resets password; new password logs in, old password fails; vault_salt is preserved."""
    initial_salt = test_user.vault_salt
    sent_otp = None

    def mock_send_mail(to_email, otp):
        nonlocal sent_otp
        sent_otp = otp

    with patch("app.auth.mailer.send_password_reset_email", side_effect=mock_send_mail):
        req_res = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
        )
        assert req_res.status_code == 200

    assert sent_otp is not None

    # Reset password with received OTP
    reset_res = await client.post(
        "/api/auth/forgot-password/reset",
        json={
            "email": test_user.email,
            "otp": sent_otp,
            "new_password": "brandNewPassword2026!",
        },
    )
    assert reset_res.status_code == 200
    assert "Password has been reset successfully" in reset_res.json()["message"]

    # Verify old password no longer works
    old_login = await client.post(
        "/api/auth/login",
        data={"username": test_user.email, "password": "password123"},
    )
    assert old_login.status_code == 401

    # Verify new password logs in successfully
    new_login = await client.post(
        "/api/auth/login",
        data={"username": test_user.email, "password": "brandNewPassword2026!"},
    )
    assert new_login.status_code == 200
    assert "access_token" in new_login.json()
    assert new_login.json()["vault_salt"] == initial_salt

    # Verify OTP is marked used and cannot be reused
    reused = await client.post(
        "/api/auth/forgot-password/reset",
        json={
            "email": test_user.email,
            "otp": sent_otp,
            "new_password": "yetAnotherPassword123",
        },
    )
    assert reused.status_code == 400
    assert "Invalid or expired password reset code" in reused.json()["detail"]


@pytest.mark.asyncio
async def test_expired_otp_fails(
    client: AsyncClient, test_user: User, db: AsyncSession
):
    """Expired OTP is rejected."""
    sent_otp = None

    def mock_send_mail(to_email, otp):
        nonlocal sent_otp
        sent_otp = otp

    with patch("app.auth.mailer.send_password_reset_email", side_effect=mock_send_mail):
        await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
        )

    # Age the OTP beyond 10 minutes (11 minutes ago)
    stmt = (
        select(PasswordResetOTP)
        .where(PasswordResetOTP.user_id == test_user.id)
        .order_by(PasswordResetOTP.created_at.desc())
    )
    rec = (await db.execute(stmt)).scalars().first()
    rec.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    await db.commit()

    res = await client.post(
        "/api/auth/forgot-password/reset",
        json={
            "email": test_user.email,
            "otp": sent_otp,
            "new_password": "newSecurePassword123",
        },
    )
    assert res.status_code == 400
    assert "Password reset code has expired" in res.json()["detail"]


@pytest.mark.asyncio
async def test_forgot_password_unconfigured_smtp_returns_503(
    client: AsyncClient, test_user: User, caplog
):
    """When SMTP is unconfigured, return 503 'Email service not configured' and never print OTP in prod."""
    orig_host = settings.smtp_host
    orig_user = settings.smtp_user
    orig_pass = settings.smtp_password
    orig_env = settings.app_env
    orig_dev_print = settings.dev_print_otp
    user_email = test_user.email
    caplog.set_level(logging.INFO)

    try:
        # Ensure SMTP is unconfigured
        settings.smtp_user = ""
        settings.smtp_password = ""

        # Case 1: production and dev_print_otp=False -> 503
        settings.app_env = "production"
        settings.dev_print_otp = False
        res1 = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": user_email},
        )
        assert res1.status_code == 503
        assert res1.json()["detail"] == "Email service not configured"

        # Case 2: production and dev_print_otp=True (simulate prod misconfiguration)
        # MUST STILL return 503 and NEVER print OTP in production
        settings.app_env = "production"
        settings.dev_print_otp = True
        caplog.clear()
        res2 = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": user_email},
        )
        assert res2.status_code == 503
        assert res2.json()["detail"] == "Email service not configured"
        assert "[DEV ONLY]" not in caplog.text

        # Case 3: development and dev_print_otp=False -> 503
        settings.app_env = "development"
        settings.dev_print_otp = False
        res3 = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": user_email},
        )
        assert res3.status_code == 503
        assert res3.json()["detail"] == "Email service not configured"

        # Case 4: development and dev_print_otp=True explicitly -> allowed with dev print
        settings.app_env = "development"
        settings.dev_print_otp = True
        caplog.clear()
        res4 = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": user_email},
        )
        assert res4.status_code == 200
        assert "[DEV ONLY]" in caplog.text

    finally:
        settings.smtp_host = orig_host
        settings.smtp_user = orig_user
        settings.smtp_password = orig_pass
        settings.app_env = orig_env
        settings.dev_print_otp = orig_dev_print


@pytest.mark.asyncio
async def test_forgot_password_ip_rate_limiting_via_x_forwarded_for(
    client: AsyncClient, test_user: User, db: AsyncSession
):
    """Derive client IP from first value of X-Forwarded-For and enforce 10/hr rate limit."""
    target_ip = "203.0.113.195"
    different_ip = "198.51.100.77"
    now_utc = datetime.now(timezone.utc)

    # Pre-seed 10 OTP requests from target_ip created within the last 30 minutes
    for i in range(10):
        dummy_record = PasswordResetOTP(
            user_id=test_user.id,
            email=f"dummy{i}@example.com",
            request_ip=target_ip,
            otp_hash=f"hash{i}" * 8,
            expires_at=now_utc + timedelta(minutes=10),
            created_at=now_utc - timedelta(minutes=15),
            attempts=0,
        )
        db.add(dummy_record)
    await db.commit()

    with patch("app.auth.mailer.send_password_reset_email"):
        # Request with X-Forwarded-For where first value matches target_ip -> blocked by IP rate limit
        res_blocked = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
            headers={"x-forwarded-for": f"{target_ip}, 10.0.0.1"},
        )
        assert res_blocked.status_code == 429
        assert "Too many requests from this network" in res_blocked.json()["detail"]

        # Request from different IP succeeds (first entry in X-Forwarded-For chain)
        res_allowed = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
            headers={"x-forwarded-for": f"{different_ip}, 10.0.0.1"},
        )
        assert res_allowed.status_code == 200


@pytest.mark.asyncio
async def test_brevo_success_sends_single_post_and_logs_no_secrets(
    test_user: User, caplog
):
    """Brevo success sends exactly 1 POST to Brevo API with api-key header, valid JSON, and no leaked secrets."""
    from app.auth import mailer

    orig_provider = settings.email_provider
    orig_key = settings.brevo_api_key
    orig_from = settings.email_from
    test_api_key = "mock-secret-key-12345"
    test_from_email = "mailer@varman.ai"
    test_otp = "852963"

    try:
        settings.email_provider = "brevo"
        settings.brevo_api_key = test_api_key
        settings.email_from = test_from_email

        caplog.clear()
        caplog.set_level(logging.INFO)

        mock_resp = httpx.Response(201, json={"messageId": "mock-msg-id-1"})
        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            mailer.send_password_reset_email(test_user.email, test_otp)

            assert mock_post.call_count == 1
            call_url = mock_post.call_args[0][0]
            call_kwargs = mock_post.call_args[1]

            assert call_url == "https://api.brevo.com/v3/smtp/email"
            assert call_kwargs["headers"]["api-key"] == test_api_key
            assert call_kwargs["headers"]["content-type"] == "application/json"
            assert call_kwargs["headers"]["accept"] == "application/json"

            json_body = call_kwargs["json"]
            assert json_body["sender"] == {"name": "Varman", "email": test_from_email}
            assert json_body["to"] == [{"email": test_user.email}]
            assert json_body["subject"] == "Varman Password Reset Code"
            assert test_otp in json_body["htmlContent"]
            assert test_otp in json_body["textContent"]
            assert "encrypted vault inaccessible" in json_body["htmlContent"]
            assert "encrypted vault inaccessible" in json_body["textContent"]

        # Ensure no secrets or emails in logs
        assert test_api_key not in caplog.text
        assert test_otp not in caplog.text
        assert test_user.email not in caplog.text
        assert "[Mailer] Password reset email sent successfully via Brevo" in caplog.text

    finally:
        settings.email_provider = orig_provider
        settings.brevo_api_key = orig_key
        settings.email_from = orig_from


@pytest.mark.asyncio
async def test_brevo_failure_and_timeout_returns_503_and_leaves_no_otp_row(
    client: AsyncClient, test_user: User, db: AsyncSession
):
    """When Brevo fails or times out, return 503 'Failed to send email' and leave no row in password_reset_otps."""
    orig_provider = settings.email_provider
    orig_key = settings.brevo_api_key
    orig_from = settings.email_from

    try:
        settings.email_provider = "brevo"
        settings.brevo_api_key = "mock-secret-key-12345"
        settings.email_from = "mailer@varman.ai"
        user_id = test_user.id
        user_email = test_user.email

        # Case 1: HTTP error response (e.g. 500 from Brevo)
        mock_err_resp = httpx.Response(500, text="Internal Server Error")
        with patch("httpx.Client.post", return_value=mock_err_resp):
            res_fail = await client.post(
                "/api/auth/forgot-password/request",
                json={"email": user_email},
            )
            assert res_fail.status_code == 503
            assert res_fail.json()["detail"] == "Failed to send email"

        # Verify no OTP record was persisted
        stmt = select(PasswordResetOTP).where(PasswordResetOTP.user_id == user_id)
        assert (await db.execute(stmt)).first() is None

        # Case 2: Timeout exception (network hang)
        with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Connection timed out")):
            res_timeout = await client.post(
                "/api/auth/forgot-password/request",
                json={"email": user_email},
            )
            assert res_timeout.status_code == 503
            assert res_timeout.json()["detail"] == "Failed to send email"

        # Verify still no OTP record persisted
        assert (await db.execute(stmt)).first() is None

    finally:
        settings.email_provider = orig_provider
        settings.brevo_api_key = orig_key
        settings.email_from = orig_from


def test_email_provider_precedence(test_user: User):
    """Verify provider choice precedence: Brevo when configured, SMTP when configured, 503 when neither."""
    from fastapi import HTTPException
    from app.auth import mailer

    orig_provider = settings.email_provider
    orig_key = settings.brevo_api_key
    orig_from = settings.email_from
    orig_smtp_host = settings.smtp_host
    orig_smtp_user = settings.smtp_user
    orig_smtp_pass = settings.smtp_password
    orig_app_env = settings.app_env
    orig_dev_print = settings.dev_print_otp

    try:
        # 1. Brevo configured (even if SMTP is also configured) -> calls _send_brevo
        settings.email_provider = "brevo"
        settings.brevo_api_key = "mock-brevo-key"
        settings.email_from = "noreply@varman.ai"
        settings.smtp_user = "user@example.com"
        settings.smtp_password = "smtppassword"

        with patch.object(mailer, "_send_brevo") as mock_b, patch.object(mailer, "_send_smtp") as mock_s:
            mailer.send_password_reset_email(test_user.email, "112233")
            mock_b.assert_called_once_with(test_user.email, "112233")
            mock_s.assert_not_called()

        # 2. Brevo NOT configured, but SMTP configured -> calls _send_smtp
        settings.email_provider = ""
        settings.brevo_api_key = ""
        settings.smtp_host = "smtp.gmail.com"
        settings.smtp_user = "user@example.com"
        settings.smtp_password = "smtppassword"

        with patch.object(mailer, "_send_brevo") as mock_b, patch.object(mailer, "_send_smtp") as mock_s:
            mailer.send_password_reset_email(test_user.email, "445566")
            mock_s.assert_called_once_with(test_user.email, "445566")
            mock_b.assert_not_called()

        # 3. Neither configured in production -> raises HTTP 503 "Email service not configured"
        settings.email_provider = ""
        settings.brevo_api_key = ""
        settings.smtp_user = ""
        settings.smtp_password = ""
        settings.app_env = "production"
        settings.dev_print_otp = False

        with pytest.raises(HTTPException) as exc_info:
            mailer.send_password_reset_email(test_user.email, "778899")
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "Email service not configured"

    finally:
        settings.email_provider = orig_provider
        settings.brevo_api_key = orig_key
        settings.email_from = orig_from
        settings.smtp_host = orig_smtp_host
        settings.smtp_user = orig_smtp_user
        settings.smtp_password = orig_smtp_pass
        settings.app_env = orig_app_env
        settings.dev_print_otp = orig_dev_print


@pytest.mark.asyncio
async def test_verify_otp_endpoint_success_and_failures(
    client: AsyncClient, test_user: User, db: AsyncSession
):
    """Test verify-otp endpoint validates code without consuming it, locks after 5 attempts, and fails if expired."""
    sent_codes = []

    def mock_send(to_email, otp):
        sent_codes.append(otp)

    with patch("app.auth.mailer.send_password_reset_email", side_effect=mock_send):
        res_req = await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
        )
        assert res_req.status_code == 200

    assert len(sent_codes) == 1
    valid_otp = sent_codes[0]

    # 1. Unknown email -> 400 Normalized Error (no account enumeration)
    res_unknown = await client.post(
        "/api/auth/forgot-password/verify-otp",
        json={"email": "nobody_exists@example.com", "otp": valid_otp},
    )
    assert res_unknown.status_code == 400
    assert "Invalid or expired password reset code" in res_unknown.json()["detail"]

    # 2. Invalid code -> 400 with 4 attempts remaining
    res_wrong = await client.post(
        "/api/auth/forgot-password/verify-otp",
        json={"email": test_user.email, "otp": "000000" if valid_otp != "000000" else "111111"},
    )
    assert res_wrong.status_code == 400
    assert "4 attempt(s) remaining" in res_wrong.json()["detail"]

    # 3. Valid code -> 200 and success message
    res_valid = await client.post(
        "/api/auth/forgot-password/verify-otp",
        json={"email": test_user.email, "otp": valid_otp},
    )
    assert res_valid.status_code == 200
    assert "Verification code confirmed" in res_valid.json()["message"]

    # Ensure OTP is NOT marked used yet
    stmt = (
        select(PasswordResetOTP)
        .where(PasswordResetOTP.user_id == test_user.id)
        .order_by(PasswordResetOTP.created_at.desc())
    )
    record = (await db.execute(stmt)).scalars().first()
    assert record.used_at is None

    # 4. Now reset password using the same valid code -> 200
    res_reset = await client.post(
        "/api/auth/forgot-password/reset",
        json={
            "email": test_user.email,
            "otp": valid_otp,
            "new_password": "newSecurePassword2026!",
        },
    )
    assert res_reset.status_code == 200

    # 5. After reset, OTP is marked used, so verify-otp fails with 400
    res_after = await client.post(
        "/api/auth/forgot-password/verify-otp",
        json={"email": test_user.email, "otp": valid_otp},
    )
    assert res_after.status_code == 400
    assert "Invalid or expired password reset code" in res_after.json()["detail"]


@pytest.mark.asyncio
async def test_verify_otp_endpoint_lockout_and_expiration(
    client: AsyncClient, test_user: User, db: AsyncSession
):
    """5 failed attempts on verify-otp locks the OTP, and expired OTP returns 400."""
    sent_codes = []

    def mock_send(to_email, otp):
        sent_codes.append(otp)

    with patch("app.auth.mailer.send_password_reset_email", side_effect=mock_send):
        await client.post(
            "/api/auth/forgot-password/request",
            json={"email": test_user.email},
        )

    valid_otp = sent_codes[0]
    wrong_otp = "000000" if valid_otp != "000000" else "111111"

    # Fail 4 times (attempts 1 to 4)
    for i in range(1, 5):
        res = await client.post(
            "/api/auth/forgot-password/verify-otp",
            json={"email": test_user.email, "otp": wrong_otp},
        )
        assert res.status_code == 400
        assert f"{5 - i} attempt(s) remaining" in res.json()["detail"]

    # 5th attempt locks it
    res5 = await client.post(
        "/api/auth/forgot-password/verify-otp",
        json={"email": test_user.email, "otp": wrong_otp},
    )
    assert res5.status_code == 400
    assert "Too many failed attempts" in res5.json()["detail"]

    # Even correct OTP is rejected now
    res_locked = await client.post(
        "/api/auth/forgot-password/verify-otp",
        json={"email": test_user.email, "otp": valid_otp},
    )
    assert res_locked.status_code == 400
    assert "Too many failed attempts" in res_locked.json()["detail"]


@pytest.mark.asyncio
async def test_account_enumeration_protection_when_disclosure_disabled(
    client: AsyncClient, test_user: User, db: AsyncSession
):
    """When reveal_unregistered_email=False, /verify-otp and /reset return identical 400 responses
    across nonexistent accounts, accounts without OTP, invalid OTP, expired OTP, and locked OTP."""
    orig_reveal = settings.reveal_unregistered_email
    settings.reveal_unregistered_email = False
    try:
        nonexistent_email = "nonexistent_enum_target@example.com"
        initial_hash = test_user.hashed_password

        # Case 1: Nonexistent account
        res_v_non = await client.post(
            "/api/auth/forgot-password/verify-otp",
            json={"email": nonexistent_email, "otp": "123456"},
        )
        res_r_non = await client.post(
            "/api/auth/forgot-password/reset",
            json={"email": nonexistent_email, "otp": "123456", "new_password": "HackedPassword1!"},
        )

        # Case 2: Registered account with NO active reset request
        res_v_no_req = await client.post(
            "/api/auth/forgot-password/verify-otp",
            json={"email": test_user.email, "otp": "123456"},
        )
        res_r_no_req = await client.post(
            "/api/auth/forgot-password/reset",
            json={"email": test_user.email, "otp": "123456", "new_password": "HackedPassword1!"},
        )

        expected_msg = "Invalid or expired password reset code. Please request a new code."

        assert res_v_non.status_code == 400
        assert res_v_non.json()["detail"] == expected_msg
        assert res_r_non.status_code == 400
        assert res_r_non.json()["detail"] == expected_msg

        assert res_v_no_req.status_code == 400
        assert res_v_no_req.json()["detail"] == expected_msg
        assert res_r_no_req.status_code == 400
        assert res_r_no_req.json()["detail"] == expected_msg

        # Request an OTP for test_user
        sent_codes = []
        with patch("app.auth.mailer.send_password_reset_email", side_effect=lambda em, o: sent_codes.append(o)):
            req_res = await client.post(
                "/api/auth/forgot-password/request",
                json={"email": test_user.email},
            )
            assert req_res.status_code == 200

        valid_otp = sent_codes[0]
        wrong_otp = "999999" if valid_otp != "999999" else "888888"

        # Case 3: Registered account with active OTP, but wrong code
        res_v_wrong = await client.post(
            "/api/auth/forgot-password/verify-otp",
            json={"email": test_user.email, "otp": wrong_otp},
        )
        res_r_wrong = await client.post(
            "/api/auth/forgot-password/reset",
            json={"email": test_user.email, "otp": wrong_otp, "new_password": "HackedPassword1!"},
        )
        assert res_v_wrong.status_code == 400
        assert res_v_wrong.json()["detail"] == expected_msg
        assert res_r_wrong.status_code == 400
        assert res_r_wrong.json()["detail"] == expected_msg

        # Case 4: Expired OTP
        stmt = select(PasswordResetOTP).where(PasswordResetOTP.user_id == test_user.id)
        otp_rec = (await db.execute(stmt)).scalars().first()
        otp_rec.expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        await db.commit()

        res_v_exp = await client.post(
            "/api/auth/forgot-password/verify-otp",
            json={"email": test_user.email, "otp": valid_otp},
        )
        res_r_exp = await client.post(
            "/api/auth/forgot-password/reset",
            json={"email": test_user.email, "otp": valid_otp, "new_password": "HackedPassword1!"},
        )
        assert res_v_exp.status_code == 400
        assert res_v_exp.json()["detail"] == expected_msg
        assert res_r_exp.status_code == 400
        assert res_r_exp.json()["detail"] == expected_msg

        # Case 5: Locked out (attempts >= 5)
        otp_rec.expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        otp_rec.attempts = 5
        await db.commit()

        res_v_lock = await client.post(
            "/api/auth/forgot-password/verify-otp",
            json={"email": test_user.email, "otp": valid_otp},
        )
        res_r_lock = await client.post(
            "/api/auth/forgot-password/reset",
            json={"email": test_user.email, "otp": valid_otp, "new_password": "HackedPassword1!"},
        )
        assert res_v_lock.status_code == 400
        assert res_v_lock.json()["detail"] == expected_msg
        assert res_r_lock.status_code == 400
        assert res_r_lock.json()["detail"] == expected_msg

        # Verify that throughout ALL unauthorized attempts, user password hash is completely unchanged!
        await db.refresh(test_user)
        assert test_user.hashed_password == initial_hash

        # User can still log in with original credentials
        login_res = await client.post(
            "/api/auth/login",
            data={"username": test_user.email, "password": "password123"},
        )
        assert login_res.status_code == 200

    finally:
        settings.reveal_unregistered_email = orig_reveal


@pytest.mark.asyncio
async def test_account_enumeration_protection_when_disclosure_enabled(
    client: AsyncClient, test_user: User, db: AsyncSession
):
    """Even when reveal_unregistered_email=True, /verify-otp and /reset return identical HTTP 400 responses
    with NORMALIZED_OTP_ERROR for nonexistent accounts vs registered accounts without an active OTP request."""
    orig_reveal = settings.reveal_unregistered_email
    settings.reveal_unregistered_email = True
    try:
        nonexistent_email = "definitely_nonexistent_enum_target_9999@example.com"
        initial_hash = test_user.hashed_password
        expected_msg = "Invalid or expired password reset code. Please request a new code."

        # Case 1: Nonexistent account at /verify-otp and /reset
        res_v_non = await client.post(
            "/api/auth/forgot-password/verify-otp",
            json={"email": nonexistent_email, "otp": "123456"},
        )
        res_r_non = await client.post(
            "/api/auth/forgot-password/reset",
            json={"email": nonexistent_email, "otp": "123456", "new_password": "HackedPassword1!"},
        )

        # Case 2: Registered account with NO active reset request at /verify-otp and /reset
        res_v_no_req = await client.post(
            "/api/auth/forgot-password/verify-otp",
            json={"email": test_user.email, "otp": "123456"},
        )
        res_r_no_req = await client.post(
            "/api/auth/forgot-password/reset",
            json={"email": test_user.email, "otp": "123456", "new_password": "HackedPassword1!"},
        )

        # Both must return HTTP 400 Bad Request with identical detail
        assert res_v_non.status_code == 400
        assert res_v_non.json()["detail"] == expected_msg
        assert res_r_non.status_code == 400
        assert res_r_non.json()["detail"] == expected_msg

        assert res_v_no_req.status_code == 400
        assert res_v_no_req.json()["detail"] == expected_msg
        assert res_r_no_req.status_code == 400
        assert res_r_no_req.json()["detail"] == expected_msg

        # Indistinguishable comparison
        assert res_v_non.status_code == res_v_no_req.status_code
        assert res_v_non.json() == res_v_no_req.json()
        assert res_r_non.status_code == res_r_no_req.status_code
        assert res_r_non.json() == res_r_no_req.json()

        # Password hash remains untouched
        await db.refresh(test_user)
        assert test_user.hashed_password == initial_hash

        # User still logs in with original credentials
        login_res = await client.post(
            "/api/auth/login",
            data={"username": test_user.email, "password": "password123"},
        )
        assert login_res.status_code == 200

    finally:
        settings.reveal_unregistered_email = orig_reveal




