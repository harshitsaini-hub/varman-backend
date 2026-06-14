import uuid
from pydantic import BaseModel, EmailStr, Field

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: str | None = None

class UserBase(BaseModel):
    email: EmailStr
    display_name: str = Field(default="", max_length=100)

class UserCreate(UserBase):
    password: str = Field(min_length=8)

class UserResponse(UserBase):
    id: uuid.UUID
    is_active: bool

    model_config = {
        "from_attributes": True
    }
