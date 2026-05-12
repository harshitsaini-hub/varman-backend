from telethon import TelegramClient, events

from services import db_service, notification_service
from utils.hashing import compute_phash_from_bytes

DANGER_CHANNELS = [
    "your_channel_slug_1",
    "your_channel_slug_2",
]


async def start_watcher(client: TelegramClient, db):

    @client.on(events.NewMessage(chats=DANGER_CHANNELS))
    async def handler(event):
        try:
            if not event.message.photo and not event.message.document:
                return

            media_bytes = await event.download_media(bytes)

            if not isinstance(media_bytes, bytes):
                return

            phash = compute_phash_from_bytes(media_bytes)

            if phash is None:
                return

            match = db_service.lookup_phash_global(db, phash, threshold=10)
            if match:
                chat = await event.get_chat()
                chat_title = getattr(chat, "title", "Unknown")
                notification_service.send_radar_alert(
                    user_id=match.user_id,
                    suspect_url=f"Telegram: {chat_title}",
                    image_url="[Telegram — no direct URL]",
                    platform="Telegram",
                    context=f"Message ID: {event.message.id}",
                )

        except Exception as e:
            print(f"[TELEGRAM ERROR] {e}")

    await client.run_until_disconnected()
