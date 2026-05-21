from typing import Annotated

import jwt
from fastapi import Header, HTTPException

from core.config import API_KEY, JWT_ALGORITHM, JWT_SECRET


def require_service_auth(
    x_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> dict:
    if API_KEY and x_api_key == API_KEY:
        return {"auth_type": "api_key"}

    if JWT_SECRET and authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return {"auth_type": "jwt", "sub": payload.get("sub")}
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=401, detail="Invalid token") from exc

    raise HTTPException(status_code=401, detail="Authentication required")


def require_owned_user_id(requested_user_id: str, auth_payload: dict) -> None:
    if auth_payload.get("auth_type") == "jwt":
        token_sub = auth_payload.get("sub")
        if not token_sub or token_sub != requested_user_id:
            raise HTTPException(status_code=403, detail="User ownership check failed")
