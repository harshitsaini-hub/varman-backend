import re
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
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
    assert "No active password reset request found" in reused.json()["detail"]


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
