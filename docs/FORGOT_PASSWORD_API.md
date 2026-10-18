# Varman Authentication & Password Reset API Specification

## 1. Overview & Security Architecture

The Varman Authentication service provides secure user identity management, credential hashing, and self-service password recovery via one-time passwords (OTP).

### Key Security & Cryptographic Safeguards

1. **HMAC-SHA256 OTP Storage:**
   - OTPs are generated using `secrets.randbelow(1000000)` formatted as a 6-digit zero-padded string.
   - Raw OTPs are **never stored in plaintext** in the database.
   - OTPs are hashed using HMAC-SHA256 keyed with the application's `JWT_SECRET`:
     $$\text{OTP\_Hash} = \text{HMAC-SHA256}(\text{key}=\text{JWT\_SECRET}, \text{msg}=\text{otp})$$
   - Verification uses constant-time comparison (`hmac.compare_digest`) to prevent timing attacks.

2. **Expiration & Attempt Lockout:**
   - Every OTP expires strictly after **10 minutes** (`expires_at = created_at + 10m`).
   - A maximum of **5 failed attempts** are permitted per OTP. If exceeded, the record is immediately voided and subsequent submissions return `HTTP 400 Bad Request`.

3. **Multi-Tier Rate Limiting:**
   - **Cooldown:** Minimum 60-second cooldown between successive OTP requests for the same email address (`HTTP 429`).
   - **Email Rate Limit:** Maximum 3 OTP requests per hour per email (`HTTP 429`).
   - **IP Rate Limit:** Maximum 10 OTP requests per hour per client IP (`HTTP 429`). The client IP is derived from the first value of the `X-Forwarded-For` header when present (accommodating the Hugging Face Spaces reverse proxy), falling back to the direct socket connection address.
     > [!NOTE]
     > Deriving the client IP from `X-Forwarded-For` is **only trustworthy** when the service is deployed behind a trusted reverse proxy (e.g. Hugging Face Spaces) that scrubs or overrides incoming client headers. Without a trusted proxy, clients could spoof this header.

4. **Service Availability & Unconfigured SMTP:**
   - If SMTP is not configured (`SMTP_HOST`, `SMTP_USER`, or `SMTP_PASSWORD` missing) and (`APP_ENV` is `production` OR `DEV_PRINT_OTP` is not explicitly `True`), the endpoint immediately raises `HTTP 503 Service Unavailable` with `detail="Email service not configured"`.
   - The OTP is **never printed or logged** in production environments.
   - In non-production environments with `DEV_PRINT_OTP=True` explicitly enabled, the OTP is written to local development server logs for testing.

5. **Account Enumeration Defense:**
   - Controlled via the `REVEAL_UNREGISTERED_EMAIL` configuration flag:
     - `False` (default for production): Returns `HTTP 200 OK` with a generic message (*"If this email is registered, a password reset code has been sent."*), preventing attackers from scraping valid user accounts.
     - `True` (development / testing): Returns `HTTP 404 Not Found` (*"No account found with this email."*).

6. **Client-Side Encrypted Vault Impact:**
   - User encrypted image vaults use client-derived keys via PBKDF2 (600,000 iterations, SHA-256) from the user's password and persistent `vault_salt`.
   - The backend preserves the user's `vault_salt` during password resets.
   - **Warning to Users:** Because vault content is encrypted with keys derived from the old password, changing the password renders previously encrypted images in the local vault unreadable unless the user retains or restores old credentials.

---

## 2. API Endpoints

### 2.1 Register Identity

Create a new operator account and immediately receive JWT session credentials and vault salt.

- **Method:** `POST`
- **Path:** `/api/auth/register`
- **Content-Type:** `application/json`

#### Request Body
```json
{
  "email": "operator@example.com",
  "password": "strongPassword123",
  "display_name": "Agent Smith"
}
```
*Constraints:* `email` must be a valid email format. `password` must be at least 8 characters.

#### Responses

- **`201 Created`** (Success)
  ```json
  {
    "access_token": "eyJhbGciOi...",
    "token_type": "bearer",
    "vault_salt": "64_hex_character_salt..."
  }
  ```
- **`409 Conflict`** (Duplicate Identity)
  ```json
  {
    "detail": "Email already registered"
  }
  ```
- **`422 Unprocessable Entity`** (Validation Failure)
  ```json
  {
    "detail": [
      {
        "loc": ["body", "password"],
        "msg": "String should have at least 8 characters",
        "type": "string_too_short"
      }
    ]
  }
  ```

---

### 2.2 Request Password Reset OTP

Initiates the password reset workflow by generating and emailing a 6-digit OTP.

- **Method:** `POST`
- **Path:** `/api/auth/forgot-password/request`
- **Content-Type:** `application/json`

#### Request Body
```json
{
  "email": "operator@example.com"
}
```

#### Responses

- **`200 OK`** (OTP Generated & Sent)
  ```json
  {
    "message": "If this email is registered, a password reset code has been sent."
  }
  ```
- **`404 Not Found`** (Only when `REVEAL_UNREGISTERED_EMAIL=True`)
  ```json
  {
    "detail": "No account found with this email."
  }
  ```
- **`429 Too Many Requests`** (Rate Limit Exceeded)
  - Cooldown:
    ```json
    {
      "detail": "Please wait 54 seconds before requesting another code."
    }
    ```
  - Hourly Limit:
    ```json
    {
      "detail": "Too many reset requests for this email. Please try again in an hour."
    }
    ```
- **`503 Service Unavailable`** (SMTP Relay Delivery Failure)
  ```json
  {
    "detail": "Failed to send reset email. Please try again later."
  }
  ```

---

### 2.3 Verify OTP & Reset Password

Validates the OTP and updates the account password.

- **Method:** `POST`
- **Path:** `/api/auth/forgot-password/reset`
- **Content-Type:** `application/json`

#### Request Body
```json
{
  "email": "operator@example.com",
  "otp": "481029",
  "new_password": "newSecurePassword456"
}
```
*Constraints:* `otp` must be exactly 6 numeric digits. `new_password` must be at least 8 characters.

#### Responses

- **`200 OK`** (Password Successfully Reset)
  ```json
  {
    "message": "Password successfully reset. Please log in with your new password."
  }
  ```
- **`400 Bad Request`** (Invalid / Expired / Locked OTP)
  - Invalid OTP:
    ```json
    {
      "detail": "Invalid or expired reset code."
    }
    ```
  - Lockout:
    ```json
    {
      "detail": "Too many failed attempts. Please request a new reset code."
    }
    ```
- **`404 Not Found`**
  ```json
  {
    "detail": "User not found."
  }
  ```
- **`422 Unprocessable Entity`** (Schema / Formatting Error)
  ```json
  {
    "detail": [
      {
        "loc": ["body", "otp"],
        "msg": "String should match pattern '^[0-9]{6}$'",
        "type": "string_pattern_mismatch"
      }
    ]
  }
  ```

---

## 3. Environment Configuration Reference

| Environment Variable | Type | Default | Description |
|---|---|---|---|
| `SMTP_HOST` | `str` | `""` | Hostname of SMTP relay (e.g. `smtp.sendgrid.net`, `smtp.gmail.com`). |
| `SMTP_PORT` | `int` | `587` | Port for SMTP service (`587` for STARTTLS, `465` for SSL). |
| `SMTP_USER` | `str` | `""` | SMTP authentication username. |
| `SMTP_PASSWORD` | `str` | `""` | SMTP authentication password / app key. |
| `SMTP_FROM` | `str` | `noreply@varman.ai` | Sender address visible on outgoing emails. Note: `SMTP_FROM` should equal `SMTP_USER` (Gmail and other authenticated relays override other From addresses). |
| `APP_ENV` | `str` | `production` | Deployment environment (`production` or `development`). |
| `DEV_PRINT_OTP` | `bool` | `False` | When `True`, prints OTP to server console (dev only). |
| `REVEAL_UNREGISTERED_EMAIL`| `bool` | `False` | When `False`, returns generic 200 on missing emails. |
