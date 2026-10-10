"""Database migration utility for Varman schema updates.
Respects settings.database_url across SQLite, MySQL, and PostgreSQL.
"""

import asyncio
from sqlalchemy import text
from app.config import settings
from app.database import engine


async def run_migration() -> None:
    print(f"[Migration] Connecting to database: {settings.database_url}")
    try:
        async with engine.begin() as conn:
            try:
                await conn.execute(text('ALTER TABLE users ADD COLUMN vault_salt VARCHAR(64) NOT NULL DEFAULT ""'))
                print("[Migration] Added vault_salt to users.")
            except Exception as e:
                print(f"[Migration] Note on users table: {e}")

            try:
                await conn.execute(text('ALTER TABLE protected_images ADD COLUMN vault_sealed BOOLEAN NOT NULL DEFAULT 0'))
                print("[Migration] Added vault_sealed to protected_images.")
            except Exception as e:
                print(f"[Migration] Note on protected_images table: {e}")

        print("[Migration] Migration finished.")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_migration())
