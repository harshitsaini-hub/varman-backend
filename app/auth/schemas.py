import uuid
from pydantic import BaseModel, EmailStr, Field

class Token(BaseModel):
    access_token: str
    token_type: str
    vault_salt: str = ""

class TokenData(BaseModel):
    user_id: str | None = None

class UserBase(BaseModel):
    email: EmailStr
    display_name: str = Field(default="", max_length=100)

class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=72)

class UserResponse(UserBase):
    id: uuid.UUID
    is_active: bool
    vault_salt: str = ""

    model_config = {
        "from_attributes": True
    }


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=6, max_length=6)
    new_password: str = Field(min_length=8, max_length=72)


class MessageResponse(BaseModel):
    message: str
