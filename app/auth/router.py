import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import schemas, security
from app.database import get_db
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_in: schemas.UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user."""
    # Check if email exists
    stmt = select(User).where(User.email == user_in.email)
    result = await db.execute(stmt)
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    hashed_password = security.get_password_hash(user_in.password)
    new_user = User(
        email=user_in.email,
        hashed_password=hashed_password,
        display_name=user_in.display_name
    )
    
    db.add(new_user)
    try:
        await db.commit()
        await db.refresh(new_user)
    except Exception as e:
        await db.rollback()
        logger.error("Failed to register user: %s", e)
        raise HTTPException(status_code=500, detail="Database error during registration")
        
    return new_user


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

    # Generate token
    access_token = security.create_access_token(subject=user.id)
    return {"access_token": access_token, "token_type": "bearer"}


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
