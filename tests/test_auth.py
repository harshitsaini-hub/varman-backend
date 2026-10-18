import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_register_success(client: AsyncClient):
    response = await client.post(
        "/api/auth/register",
        json={"email": "new@example.com", "password": "securepassword123", "display_name": "New"}
    )
    assert response.status_code == 201
    assert "access_token" in response.json()
    assert "vault_salt" in response.json()

@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient, test_user):
    response = await client.post(
        "/api/auth/register",
        json={"email": test_user.email, "password": "password123"}
    )
    assert response.status_code == 409
    assert "already registered" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient, test_user):
    response = await client.post(
        "/api/auth/login",
        data={"username": test_user.email, "password": "password123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

@pytest.mark.asyncio
async def test_login_failure(client: AsyncClient, test_user):
    response = await client.post(
        "/api/auth/login",
        data={"username": test_user.email, "password": "wrongpassword"}
    )
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_me_protected_route(client: AsyncClient, test_token: str):
    response = await client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {test_token}"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"

@pytest.mark.asyncio
async def test_me_unauthorized(client: AsyncClient):
    response = await client.get("/api/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_register_validation_error_structure_and_no_value_leakage(client: AsyncClient, caplog):
    """Validation errors must return 422 with string detail, sanitized errors, and never leak values."""
    caplog.clear()
    invalid_email = "invalid-email-address"
    short_pw = "pw123"

    response = await client.post(
        "/api/auth/register",
        json={"email": invalid_email, "password": short_pw}
    )
    assert response.status_code == 422
    data = response.json()

    # Detail must be a string naming the field
    assert isinstance(data["detail"], str)
    assert any(data["detail"].startswith(prefix) for prefix in ("email:", "password:"))

    # Errors must be a list of {field, message} dicts
    assert isinstance(data["errors"], list)
    assert len(data["errors"]) >= 1
    fields = [e["field"] for e in data["errors"]]
    assert "email" in fields or "password" in fields

    # NEVER include submitted values in response
    resp_text = response.text
    assert invalid_email not in resp_text
    assert short_pw not in resp_text
    for err in data["errors"]:
        assert "input" not in err

    # NEVER log submitted values
    assert invalid_email not in caplog.text
    assert short_pw not in caplog.text


@pytest.mark.asyncio
async def test_register_password_boundary_validation(client: AsyncClient):
    """Password must be between 8 and 72 characters (bcrypt limit)."""
    # 7 characters (too short)
    res_short = await client.post(
        "/api/auth/register",
        json={"email": "valid_boundary@example.com", "password": "a" * 7}
    )
    assert res_short.status_code == 422
    assert "at least 8 characters" in res_short.json()["detail"]

    # 73 characters (too long for bcrypt)
    res_long = await client.post(
        "/api/auth/register",
        json={"email": "valid_boundary@example.com", "password": "a" * 73}
    )
    assert res_long.status_code == 422
    assert "at most 72 characters" in res_long.json()["detail"]

    # Display name too long (> 100)
    res_name = await client.post(
        "/api/auth/register",
        json={"email": "valid_boundary@example.com", "password": "validPassword123", "display_name": "x" * 101}
    )
    assert res_name.status_code == 422
    assert "display_name" in res_name.json()["detail"]
