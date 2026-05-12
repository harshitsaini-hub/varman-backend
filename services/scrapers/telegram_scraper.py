import logging

from telethon import TelegramClient, events

from core.config import TELEGRAM_DANGER_CHANNELS
from services import db_service, notification_service
from utils.hashing import compute_phash_from_bytes

logger = logging.getLogger(__name__)
DANGER_CHANNELS = TELEGRAM_DANGER_CHANNELS


async def start_watcher(client: TelegramClient, db) -> None:
    """Watch configured Telegram channels for media matching protected pHashes."""
    if not DANGER_CHANNELS:
        logger.warning("[TELEGRAM] No channels configured; set TELEGRAM_DANGER_CHANNELS.")
        await client.run_until_disconnected()
        return

    logger.info("[TELEGRAM] Watching %s channels", len(DANGER_CHANNELS))

    @client.on(events.NewMessage(chats=list(DANGER_CHANNELS)))
    async def handler(event):
        try:
            if not event.message.photo and not event.message.document:
                return

            media_bytes = await event.download_media(file=bytes)
            if media_bytes is None:
                return

            phash = compute_phash_from_bytes(media_bytes)
            if phash is None:
                return

            match = db_service.lookup_phash_global(db, phash, threshold=10)
            if not match:
                return

            chat = await event.get_chat()
            notification_service.send_radar_alert(
                user_id=match.user_id,
                suspect_url=f"Telegram channel: {getattr(chat, 'title', 'Unknown')}",
                image_url="[Telegram — no direct URL]",
                platform="Telegram",
                context=f"Message ID: {event.message.id}",
            )
            logger.warning("[TELEGRAM HIT] Match found. User %s notified.", match.user_id)
        except Exception:
            logger.exception("[TELEGRAM ERROR] Failed to process Telegram event")

    await client.run_until_disconnected()
