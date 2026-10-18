# Security and Privacy Rules for Varman

This repository enforces strict security and privacy standards to protect developer identity, runtime secrets, and user privacy.

## Mandatory Rules

1. **Zero Secret & Credential Commits:**
   - **Never commit** real passwords, password hashes (bcrypt, argon2, pbkdf2), cryptographic salts, API keys, JWT secrets, private keys (`.pem`, `.key`), or credentials.
   - All runtime secrets and operator credentials must be supplied strictly via environment variables (e.g. `VARMAN_ADMIN_EMAIL`, `VARMAN_ADMIN_PASSWORD`, `JWT_SECRET`).
   - All `.env` and secret config files must remain untracked (enforced in `.gitignore`).

2. **No Database Files in Version Control:**
   - **Never commit** SQLite database files (`*.db`, `*.sqlite`, `*.sqlite3`).
   - SQLite databases generated during development or testing must remain local and untracked. Application startup automatically creates tables via SQLAlchemy metadata if the database is absent.

3. **Identity & Email Privacy:**
   - **Never commit or log** personal email addresses or local host identifiers in code, documentation, benchmarks, or git commit history.
   - All git commits must use the GitHub no-reply address (`@users.noreply.github.com`).
   - Test fixtures and mock schemas must use synthetic RFC 2606 example domains (e.g. `@example.com`).

4. **Never Print or Paste Secrets in Logs or Tooling:**
   - Automated scripts, CI workflows, and AI assistants must never echo, log, print, or paste credentials, salts, tokens, or hashes in terminal output or documentation reports.

5. **Password Reset & OTP Security:**
   - Raw OTP codes must **never** be stored in plaintext in the database; all OTPs must be hashed via HMAC-SHA256 keyed with `JWT_SECRET`.
   - Verification must strictly employ constant-time comparison (`hmac.compare_digest`) to prevent timing side-channel attacks.
   - Enforce multi-tier rate limits (60-second cooldown, 3/hr/email, 10/hr/IP) and a strict 5-attempt brute-force lockout threshold.
   - In production environments, generic `HTTP 200` responses must be returned for unregistered addresses (`REVEAL_UNREGISTERED_EMAIL=False`) to thwart user enumeration.
   - Never echo or log OTP values or SMTP credentials in production logs or automated reports.

## Pre-Push Verification

Before pushing any commit or branch to remote, verify repository hygiene:

```bash
pytest tests/test_repo_hygiene.py
```

This test enforces:
- No tracked database files, `.env`, `.pem`, or `.key` files in git index.
- No bcrypt password hash patterns (`$2a$/$2b$/$2y$`) in any tracked text file.
- No personal email addresses outside allowed domains (`@users.noreply.github.com` and `@example.com`).
