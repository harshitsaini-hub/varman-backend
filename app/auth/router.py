import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import mailer, schemas, security
from app.config import settings
from app.database import get_db
from app.models.password_reset import PasswordResetOTP
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime has UTC timezone."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _compute_otp_hash(otp: str) -> str:
    """Compute HMAC-SHA256 hash of OTP keyed with server secret."""
    server_secret = (settings.jwt_secret or "varman-otp-default-secret").encode("utf-8")
    return hmac.new(server_secret, otp.encode("utf-8"), hashlib.sha256).hexdigest()


@router.post("/register", response_model=schemas.Token, status_code=status.HTTP_201_CREATED)
async def register(
    user_in: schemas.UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user atomically and return an access token and vault salt."""
    # Check if email exists
    stmt = select(User).where(User.email == user_in.email)
    result = await db.execute(stmt)
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Create new user
    hashed_password = security.get_password_hash(user_in.password)
    vault_salt = os.urandom(32).hex()
    new_user = User(
        email=user_in.email,
        hashed_password=hashed_password,
        display_name=user_in.display_name,
        vault_salt=vault_salt,
    )
    
    db.add(new_user)
    try:
        await db.commit()
        await db.refresh(new_user)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    except Exception as e:
        await db.rollback()
        logger.error("Failed to register user: %s", e)
        raise HTTPException(status_code=500, detail="Database error during registration")
        
    access_token = security.create_access_token(subject=new_user.id)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "vault_salt": new_user.vault_salt,
    }


@router.post("/login", response_model=schemas.Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db)
):
    """Login to get an access token. Uses OAuth2 password flow."""
    # Find user by email (form_data.username is used for email in this flow)
    stmt = select(User).where(User.email == form_data.username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    # Generate vault salt for legacy users who were created before the zero-knowledge vault
    if not user.vault_salt:
        user.vault_salt = os.urandom(32).hex()
        await db.commit()

    # Generate token
    access_token = security.create_access_token(subject=user.id)
    return {"access_token": access_token, "token_type": "bearer", "vault_salt": user.vault_salt}


def get_client_ip(request: Request) -> str:
    """Derive client IP from X-Forwarded-For header or direct connection address.

    When running behind a trusted reverse proxy (such as the Hugging Face Spaces
    reverse proxy), the original client IP is provided as the first value in the
    comma-separated X-Forwarded-For header chain.

    SECURITY NOTE: Deriving client IP from X-Forwarded-For is ONLY trustworthy
    when the application is deployed behind a trusted reverse proxy that strips or
    sanitizes client-supplied headers. If exposed directly to the internet without
    a proxy, clients can spoof this header.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return ""


@router.post("/forgot-password", response_model=schemas.MessageResponse)
@router.post("/forgot-password/request", response_model=schemas.MessageResponse)
async def request_password_reset(
    payload: schemas.ForgotPasswordRequest,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Request a 6-digit password reset OTP with rate limiting."""
    client_ip = get_client_ip(request)

    now_utc = datetime.now(timezone.utc)

    # ── Rate Limit 1: 60-second cooldown per email ───────────────────────
    cooldown_cutoff = now_utc - timedelta(seconds=60)
    stmt_cd = select(PasswordResetOTP).where(
        PasswordResetOTP.email == payload.email,
        PasswordResetOTP.created_at >= cooldown_cutoff,
    )
    if (await db.execute(stmt_cd)).first() is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Please wait 60 seconds before requesting another reset code.",
        )

    # ── Rate Limit 2: Max 3 requests per email per hour ──────────────────
    hour_cutoff = now_utc - timedelta(hours=1)
    stmt_hr_email = select(func.count()).select_from(PasswordResetOTP).where(
        PasswordResetOTP.email == payload.email,
        PasswordResetOTP.created_at >= hour_cutoff,
    )
    email_requests = (await db.execute(stmt_hr_email)).scalar() or 0
    if email_requests >= 3:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many password reset requests for this email. Please try again later.",
        )

    # ── Rate Limit 3: Max 10 requests per IP per hour ────────────────────
    if client_ip:
        stmt_hr_ip = select(func.count()).select_from(PasswordResetOTP).where(
            PasswordResetOTP.request_ip == client_ip,
            PasswordResetOTP.created_at >= hour_cutoff,
        )
        ip_requests = (await db.execute(stmt_hr_ip)).scalar() or 0
        if ip_requests >= 10:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests from this network. Please try again later.",
            )

    # ── Account Lookup ───────────────────────────────────────────────────
    stmt_user = select(User).where(User.email == payload.email)
    user = (await db.execute(stmt_user)).scalar_one_or_none()
    if not user:
        if settings.reveal_unregistered_email:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No account found for this email. Please sign up first.",
            )
        return {"message": "If an account exists for this email, a password reset code has been sent."}

    # ── OTP Generation & HMAC Storage ────────────────────────────────────
    otp = f"{secrets.randbelow(1_000_000):06d}"
    otp_hash = _compute_otp_hash(otp)
    expires_at = now_utc + timedelta(minutes=10)

    otp_record = PasswordResetOTP(
        user_id=user.id,
        email=user.email,
        request_ip=client_ip,
        otp_hash=otp_hash,
        expires_at=expires_at,
        attempts=0,
    )
    db.add(otp_record)

    # ── Send Email ───────────────────────────────────────────────────────
    try:
        mailer.send_password_reset_email(user.email, otp)
    except Exception:
        await db.rollback()
        raise

    await db.commit()
    return {"message": "Password reset code sent successfully. Please check your inbox."}


@router.post("/forgot-password/verify-otp", response_model=schemas.MessageResponse)
async def verify_otp(
    payload: schemas.VerifyOTPRequest,
    db: AsyncSession = Depends(get_db)
):
    """Verify that the 6-digit OTP entered matches the active code sent to the email holder."""
    # Find user
    stmt_user = select(User).where(User.email == payload.email)
    user = (await db.execute(stmt_user)).scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found for this email.",
        )

    # Find latest active OTP for user
    stmt_otp = (
        select(PasswordResetOTP)
        .where(
            PasswordResetOTP.user_id == user.id,
            PasswordResetOTP.used_at.is_(None),
        )
        .order_by(PasswordResetOTP.created_at.desc())
    )
    otp_record = (await db.execute(stmt_otp)).scalars().first()
    if not otp_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active password reset request found. Please request a new code.",
        )

    # Check lock (>= 5 failed attempts)
    if otp_record.attempts >= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many failed attempts. This code has been invalidated. Please request a new code.",
        )

    # Check expiration
    now_utc = datetime.now(timezone.utc)
    if _ensure_utc(otp_record.expires_at) < now_utc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password reset code has expired. Please request a new code.",
        )

    # Constant-time HMAC comparison
    candidate_hash = _compute_otp_hash(payload.otp)
    if not hmac.compare_digest(otp_record.otp_hash, candidate_hash):
        otp_record.attempts += 1
        await db.commit()
        if otp_record.attempts >= 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many failed attempts. This code has been invalidated. Please request a new code.",
            )
        remaining = 5 - otp_record.attempts
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid password reset code. {remaining} attempt(s) remaining.",
        )

    return {"message": "Verification code confirmed successfully."}


@router.post("/forgot-password/reset", response_model=schemas.MessageResponse)
async def reset_password(
    payload: schemas.ResetPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    """Verify 6-digit OTP and reset account password."""
    # Find user
    stmt_user = select(User).where(User.email == payload.email)
    user = (await db.execute(stmt_user)).scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found for this email.",
        )

    # Find latest active OTP for user
    stmt_otp = (
        select(PasswordResetOTP)
        .where(
            PasswordResetOTP.user_id == user.id,
            PasswordResetOTP.used_at.is_(None),
        )
        .order_by(PasswordResetOTP.created_at.desc())
    )
    otp_record = (await db.execute(stmt_otp)).scalars().first()
    if not otp_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active password reset request found. Please request a new code.",
        )

    # Check lock (>= 5 failed attempts)
    if otp_record.attempts >= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many failed attempts. This code has been invalidated. Please request a new code.",
        )

    # Check expiration
    now_utc = datetime.now(timezone.utc)
    if _ensure_utc(otp_record.expires_at) < now_utc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password reset code has expired. Please request a new code.",
        )

    # Constant-time HMAC comparison
    candidate_hash = _compute_otp_hash(payload.otp)
    if not hmac.compare_digest(otp_record.otp_hash, candidate_hash):
        otp_record.attempts += 1
        await db.commit()
        if otp_record.attempts >= 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many failed attempts. This code has been invalidated. Please request a new code.",
            )
        remaining = 5 - otp_record.attempts
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid password reset code. {remaining} attempt(s) remaining.",
        )

    # Successful validation: update password & mark OTP used
    user.hashed_password = security.get_password_hash(payload.new_password)
    otp_record.used_at = now_utc
    await db.commit()

    return {"message": "Password has been reset successfully. You can now log in with your new password."}


@router.get("/me", response_model=schemas.UserResponse)
async def read_users_me(
    current_user: Annotated[User, Depends(security.get_current_user)]
):
    """Get current authenticated user profile."""
    return current_user


@router.delete("/terminate-account", status_code=status.HTTP_204_NO_CONTENT)
async def terminate_account(
    current_user: Annotated[User, Depends(security.get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Delete the authenticated user and clean up their files."""
    from app.models.protected_image import ProtectedImage
    import shutil
    import os
    from app.config import settings

    # Delete images in database
    stmt = select(ProtectedImage).where(ProtectedImage.user_id == current_user.id)
    result = await db.execute(stmt)
    images = result.scalars().all()
    for img in images:
        await db.delete(img)

    # Delete storage folder on disk
    user_dir = os.path.join(settings.storage_dir, str(current_user.id))
    if os.path.exists(user_dir):
        try:
            shutil.rmtree(user_dir)
        except OSError as e:
            logger.warning("Failed to remove user directory: %s", e)

    # Delete user database entry
    await db.delete(current_user)
    await db.commit()
