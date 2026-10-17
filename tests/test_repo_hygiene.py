"""
Repository Hygiene and Security Guardrail Tests
===============================================
Ensures that no database files, credentials, private keys, bcrypt password hashes,
or unauthorized personal email addresses are ever committed to the repository index.
"""

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Allowed email domains / placeholder addresses
ALLOWED_EMAIL_DOMAINS = (
    "users.noreply.github.com",
    "example.com",
    "varman.ai",  # Placeholder domain in .env.example
)

FORBIDDEN_EXTENSIONS = (
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pem",
    ".key",
)

BINARY_EXTENSIONS = (
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".onnx",
    ".pth",
    ".safetensors",
    ".bin",
)


def get_tracked_files():
    """Retrieve all tracked files from git index."""
    res = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def test_no_forbidden_files_tracked():
    """Fail if git index contains any database, private key, or forbidden environment files."""
    tracked = get_tracked_files()
    forbidden_found = []

    for f in tracked:
        lower = f.lower()
        # Check forbidden extensions
        if any(lower.endswith(ext) for ext in FORBIDDEN_EXTENSIONS):
            forbidden_found.append(f)
        # Check forbidden environment files (allow .env.example only)
        if lower == ".env" or (lower.startswith(".env.") and lower != ".env.example"):
            forbidden_found.append(f)

    assert not forbidden_found, f"Forbidden sensitive files tracked in git: {forbidden_found}"


def test_no_bcrypt_hashes_in_tracked_text_files():
    """Fail if any tracked text file contains a bcrypt password hash."""
    tracked = get_tracked_files()
    bcrypt_pattern = re.compile(r"\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}")
    violating_files = []

    for f in tracked:
        if any(f.lower().endswith(ext) for ext in BINARY_EXTENSIONS):
            continue
        full_path = REPO_ROOT / f
        if not full_path.is_file():
            continue
        try:
            with open(full_path, "rb") as fp:
                raw = fp.read()
            if b"\x00" in raw:
                continue  # Skip binary file
            text = raw.decode("utf-8", errors="ignore")
            if bcrypt_pattern.search(text):
                violating_files.append(f)
        except Exception as e:
            violating_files.append(f"{f} (read error: {e})")

    assert not violating_files, f"Bcrypt password hash found in tracked files: {violating_files}"


def test_no_unauthorized_emails_in_tracked_text_files():
    """Fail if any tracked text file contains personal email addresses outside allowed domains."""
    tracked = get_tracked_files()
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
    violating_emails = []

    for f in tracked:
        if any(f.lower().endswith(ext) for ext in BINARY_EXTENSIONS):
            continue
        full_path = REPO_ROOT / f
        if not full_path.is_file():
            continue
        try:
            with open(full_path, "rb") as fp:
                raw = fp.read()
            if b"\x00" in raw:
                continue  # Skip binary file
            text = raw.decode("utf-8", errors="ignore")
            for match in email_pattern.finditer(text):
                domain = match.group(1).lower()
                if not any(domain == d or domain.endswith("." + d) for d in ALLOWED_EMAIL_DOMAINS):
                    violating_emails.append((f, f"***@{domain}"))
        except Exception as e:
            violating_emails.append((f, f"read error: {e}"))

    assert not violating_emails, f"Unauthorized personal emails found in tracked files: {violating_emails}"
