# Varman Project Handover

## Security & Privacy Rules

- **Zero Credentials in Git:** Never commit or print passwords, password hashes (bcrypt, argon2, pbkdf2), cryptographic salts, tokens, database files (`*.db`, `*.sqlite`, `*.sqlite3`), or personal email addresses.
- **Password Reset & OTP Security:** OTPs are stored strictly as HMAC-SHA256 hashes, verified in constant time, and rate-limited. Never log or commit raw OTPs or SMTP credentials. Consult [docs/FORGOT_PASSWORD_API.md](docs/FORGOT_PASSWORD_API.md).
- **Environment & Author Identity:** All secrets and credentials must be supplied exclusively via environment variables. All commits must use the official GitHub no-reply address (`@users.noreply.github.com`).
- **Pre-Push Gate:** Always run the repository hygiene test before every push:
  ```bash
  pytest tests/test_repo_hygiene.py
  ```
- For complete security protocols and policies, consult [docs/SECURITY_AND_PRIVACY_RULES.md](docs/SECURITY_AND_PRIVACY_RULES.md).

## Production & Deployment Reference

- **Default API Engine:** The deployed FastAPI service (`/images/protect`) defaults to the fast CPU-based `"Balanced"` engine (`warp_scale=0.015`, `dct_strength=10.0`).
- **F5 & Strong Mode Engine:** Strong PGD protection (`mode="strong"`, preset `"F5"`) requires a CUDA-capable GPU (`torch.cuda.is_available()`) and local surrogate weights cached at `~/.cache/varman/arcface_r50_backbone.pth`.
- **API Boundary:** The `"F5"` preset is research-validated and not selectable through the `/images/protect` web endpoint.
